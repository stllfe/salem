# noqa: A005

import re

from functools import cache
from functools import cached_property
from os import getenv
from typing import Literal

import jinja2 as j2

from attrs import define
from attrs import field
from attrs import validators as V

from salem.utils import get_logger


SEED = int(getenv("SEED", "25512"))
MODEL = getenv("MODEL", "qwen/Qwen2.5-32B-Instruct")
SYSTEM = getenv("SYSTEM", "You are a helpful assistant.")
MAX_TOKENS = int(getenv("MAX_TOKENS", "2048"))

Language = Literal["russian", "english"]

logger = get_logger()


@define
class GenerationParams:
  """Generation parameters for LLM."""

  min_tokens: int = 0
  max_tokens: int = MAX_TOKENS
  temperature: float = 1.0
  min_p: float = 0.0
  top_p: float = 1.0
  top_k: int = -1
  frequency_penalty: float = 0.0
  presence_penalty: float = 0.0
  repetition_penalty: float = 1.0
  stop: str | list[str] | None = None
  seed: int | None = SEED


@define
class ModelDefaults:
  gen: GenerationParams = field(factory=GenerationParams)
  sys: str = field(default=SYSTEM)


DEFAULTS: dict[str | re.Pattern, ModelDefaults] = {
  re.compile(r"qwen/Qwen2.5-(?:Coder-)?\d{1,2}B-(?:Instruct)?"): ModelDefaults(
    gen=GenerationParams(
      temperature=0.7,
      top_p=0.8,
      min_p=0.15,
      repetition_penalty=1.05,
    )
  ),
}


@cache
def get_defaults(model: str | None = None) -> ModelDefaults:
  model = model or MODEL
  for tag, defs in DEFAULTS.items():
    if re.match(tag, model):
      logger.info(f"Found preconfigured model config: {model}", once=True)
      logger.info(f"{defs!r}", once=True)
      return defs
  logger.info(f"Using common defaults for unknown model: {model}", once=True)
  return ModelDefaults()


def get_default_generation_params(model: str | None = None) -> GenerationParams:
  defs = get_defaults(model)
  return defs.gen


def get_default_system_prompt(model: str | None = None) -> str:
  defs = get_defaults(model)
  return defs.sys


@define
class Instruction:
  """A one-off prompt to generate a single-turn chat completion response."""

  prompt: str = field(validator=V.min_len(1))
  system: str | None = field(factory=get_default_system_prompt)

  @cached_property
  def template(self) -> j2.Template:
    return j2.Template(self.prompt, undefined=j2.StrictUndefined)

  def prepare(self, **context) -> list[dict[str, str]]:
    from salem.datagen.utils import dumps

    history = []
    context.update(dumps=dumps)  # special utility for jsons
    if self.system:
      history.append({"role": "system", "content": self.system})
    history.append({"role": "user", "content": self.template.render(context)})
    return history
