from os import getenv

from attrs import define
from loguru import logger
from openai import APIError
from openai import AsyncClient
from tenacity import retry
from tenacity import stop_after_attempt

from src.datagen.types import GenerationArgs


MAX_TRIES = int(getenv("MAX_TRIES", "3"))
DECODING_BACKEND = getenv("DECODING_BACKEND", "outlines")


@define
class APIArgs:
  model: str = "default"
  key: str = "EMPTY"
  base_url: str = "http://localhost:3000/v1"
  timeout: float | None = 10


def get_client(api: APIArgs) -> AsyncClient:
  return AsyncClient(api_key=api.key, base_url=api.base_url)


@logger.catch(APIError)
@retry(stop=stop_after_attempt(MAX_TRIES), reraise=True)
async def generate(
  messages: list[dict],
  llm: AsyncClient,
  *,
  gen: GenerationArgs,
  api: APIArgs,
  json_schema: dict | None = None,
) -> str:
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
  )
  result = response.choices[0]
  return result.message.content
