import re

from os import getenv
from typing import Any, Iterable

import orjson

from attrs import define
from loguru import logger
from openai import APIError
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessage
from tenacity import retry
from tenacity import stop_after_attempt

from src.datagen.types import GenerationArgs
from src.utils import get_short_uid


MAX_TRIES = int(getenv("MAX_TRIES", "3"))
DECODING_BACKEND = getenv("DECODING_BACKEND", "outlines")
TOOL_REGEX = re.compile(
  r"<tool_call>\s*(?P<content>.*?)\s*(?:</tool_call>|(?=<tool_call>)|$)",
  flags=re.MULTILINE | re.DOTALL,
)


@define
class APIArgs:
  model: str = "default"
  key: str = "EMPTY"
  base_url: str = "http://localhost:3000/v1"
  timeout: float | None = 10


@define
class FunctionCall:
  id: str
  name: str
  args: dict[str, Any]
  message: str | None


def get_client(api: APIArgs) -> AsyncClient:
  return AsyncClient(api_key=api.key, base_url=api.base_url, timeout=api.timeout)


def get_fn_call_from_message(message: str) -> Iterable[FunctionCall]:
  for m in TOOL_REGEX.finditer(message):
    try:
      d = orjson.loads(m.group().strip())
    except orjson.JSONDecodeError as err:
      logger.error(f"Error while decoding a tool call: {err}")
      continue
    msg = TOOL_REGEX.sub("", message).strip()
    yield FunctionCall(id=get_short_uid(), name=d["name"], args=d["args"], message=msg or None)


def get_fn_call_from_client(message: ChatCompletionMessage) -> Iterable[FunctionCall]:
  if not message.tool_calls:
    return
  msg = message.content
  for call in message.tool_calls:
    try:
      args = orjson.loads(call.function.arguments)
    except orjson.JSONDecodeError as err:
      logger.error(f"Error while decoding a tool call: {err}")
      continue
    yield FunctionCall(id=call.id, name=call.function.name, args=args, message=msg or None)
    msg = None


@logger.catch(APIError)
@retry(stop=stop_after_attempt(MAX_TRIES), reraise=True)
async def generate(
  messages: list[dict],
  llm: AsyncClient,
  *,
  gen: GenerationArgs,
  api: APIArgs,
  json_schema: dict | None = None,
  tools: list[str] | None = None,
) -> str | list[FunctionCall]:
  extra = {}
  extra.update(
    top_k=gen.top_k,
    min_p=gen.min_p,
    min_tokens=gen.min_tokens,
    repetition_penalty=gen.repetition_penalty,
  )
  if json_schema:
    extra.update(guided_json=json_schema, guided_decoding_backend=DECODING_BACKEND)
  response = await llm.chat.completions.create(
    messages=messages,
    model=api.model,
    top_p=gen.top_p,
    temperature=gen.temperature,
    frequency_penalty=gen.frequency_penalty,
    presence_penalty=gen.presence_penalty,
    max_tokens=gen.max_tokens,
    seed=gen.seed,
    stop=gen.stop,
    extra_body=extra,
    tools=tools,
  )
  result = response.choices[0]
  logger.debug(f"Returned choice:\n{result}")
  if not tools:
    return result.message.content
  if calls := list(filter(bool, get_fn_call_from_client(result.message))):
    return calls
  if list(filter(bool, get_fn_call_from_message(result.message.content))):
    logger.warning("Tool call found inside the message, the parsing maybe incorrect!")
    return calls
  return result.message.content
