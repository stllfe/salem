import re

from os import getenv
from typing import Any

import orjson

from attrs import define
from loguru import logger
from openai import APIError
from openai import AsyncClient
from tenacity import retry
from tenacity import stop_after_attempt

from src.datagen.types import GenerationArgs
from src.utils import get_short_uid


MAX_TRIES = int(getenv("MAX_TRIES", "3"))
DECODING_BACKEND = getenv("DECODING_BACKEND", "outlines")
TOOL_REGEX = re.compile("<tool_call>(.*)</?tool_call>")


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


def get_fn_call_from_message(message: str) -> FunctionCall | None:
  for m in TOOL_REGEX.finditer(message):
    d = orjson.loads(m.group().strip())
    return FunctionCall(id=get_short_uid(), name=d["name"], args=d["args"], message=TOOL_REGEX.sub("", message).strip())
  return None


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
) -> str | FunctionCall:
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
  # TODO: add support for parallel tool calls
  if result.message.tool_calls:
    call = result.message.tool_calls[0]
    return FunctionCall(
      id=call.id, name=call.function.name, args=orjson.loads(call.function.arguments), message=result.message.content
    )
  if call := get_fn_call_from_message(result.message.content):
    logger.warning("Tool call found inside the message, the parsing maybe incorrect!")
    return call
  return result.message.content
