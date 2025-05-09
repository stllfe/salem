import re

from os import getenv
from typing import Any, Iterable

import orjson

from attrs import define
from openai import APIError
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageToolCall
from tenacity import retry
from tenacity import stop_after_attempt

from salem.datagen.types import GenerationParams
from salem.utils import get_logger
from salem.utils import get_short_uid


logger = get_logger()

MAX_TRIES = int(getenv("MAX_TRIES", "3"))
DECODING_BACKEND = getenv("DECODING_BACKEND", "outlines")
TOOL_REGEX = re.compile(
  r"<tool(_call|_response)?s?>\s*(?P<content>.*?)\s*(?:</tool(_call|_response)?s?>|(?=<tool(_call|_response)?s?>)|$)",
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

  def dump(self) -> dict[str, Any]:
    return {
      "id": self.id,
      "type": "function",
      "function": {"arguments": orjson.dumps(self.args).decode("utf-8"), "name": self.name},
    }


def get_client(api: APIArgs) -> AsyncClient:
  return AsyncClient(api_key=api.key, base_url=api.base_url, timeout=api.timeout)


def get_fn_call_from_message(message: str) -> Iterable[FunctionCall]:
  for m in TOOL_REGEX.finditer(message):
    c = None
    try:
      c = m.group("content").strip()
      d = orjson.loads(c)
    except orjson.JSONDecodeError as err:
      logger.error(f"Error while decoding a tool call: {err} -> {c!r}")
      continue
    yield FunctionCall(id=get_short_uid(8), name=d["name"], args=d["arguments"])


def remove_fn_call_from_message(message: str) -> str:
  return TOOL_REGEX.sub("", message)


def get_fn_call_from_openai(tool_calls: list[ChatCompletionMessageToolCall]) -> Iterable[FunctionCall]:
  if not tool_calls:
    return
  for call in tool_calls:
    try:
      args = orjson.loads(call.function.arguments)
    except orjson.JSONDecodeError as err:
      logger.error(f"Error while decoding a tool call: {err}")
      continue
    yield FunctionCall(id=call.id, name=call.function.name, args=args)


@logger.catch(APIError)
@retry(stop=stop_after_attempt(MAX_TRIES), reraise=True)
async def generate(
  messages: list[dict],
  client: AsyncClient,
  *,
  params: GenerationParams,
  api: APIArgs,
  json_schema: dict | None = None,
  tools: list[str] | None = None,
) -> str | None | tuple[str | None, list[FunctionCall]]:
  extra = {}
  is_openai_call = "api.openai.com" in api.base_url
  if is_openai_call:
    logger.warning("OpenAI endpoint detected, ommiting extra generation params!", once=True)
  else:
    extra.update(
      top_k=params.top_k,
      min_p=params.min_p,
      min_tokens=params.min_tokens,
      repetition_penalty=params.repetition_penalty,
    )
  if json_schema and not is_openai_call:
    extra.update(guided_json=json_schema, guided_decoding_backend=DECODING_BACKEND)
  # TODO: make adjustments for OpenAI API to utilize all the features possible
  response = await client.chat.completions.create(
    messages=messages,
    model=api.model,
    top_p=params.top_p,
    temperature=params.temperature,
    frequency_penalty=params.frequency_penalty,
    presence_penalty=params.presence_penalty,
    max_tokens=params.max_tokens,
    seed=params.seed,
    stop=params.stop,
    extra_body=extra,
    tools=tools,
  )
  result = response.choices[0]
  logger.debug(f"Returned choice:\n{result}")

  message = result.message.content
  if not tools:
    return message
  if calls := list(get_fn_call_from_openai(result.message.tool_calls)):
    return message, calls
  if calls := list(get_fn_call_from_message(message)):
    logger.warning("Tool call found inside the message, the parsing maybe incorrect!")
    return remove_fn_call_from_message(message), calls
  return message, []
