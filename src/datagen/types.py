# noqa: A005

import json
import re

from functools import cache
from functools import cached_property
from os import getenv
from typing import Any, Literal

import jinja2 as j2

from attrs import define
from attrs import field
from attrs import validators as V

from src.utils import get_logger


SEED = int(getenv("SEED", "25512"))
MODEL = getenv("MODEL", "qwen/Qwen2.5-32B-Instruct")
SYSTEM = getenv("SYSTEM", "You are a helpful assistant.")
MAX_TOKENS = int(getenv("MAX_TOKENS", "2048"))

Language = Literal["russian", "english"]

logger = get_logger()


@define
class GenerationArgs:
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
  gen: GenerationArgs = field(factory=GenerationArgs)
  system: str = field(default=SYSTEM)


DEFAULTS: dict[str | re.Pattern, ModelDefaults] = {
  re.compile(r"qwen/Qwen2.5-(?:Coder-)?\d{1,2}B-(?:Instruct)?"): ModelDefaults(
    gen=GenerationArgs(
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


def get_default_generation(model: str | None = None) -> GenerationArgs:
  defs = get_defaults(model)
  return defs.gen


def get_default_system_prompt(model: str | None = None) -> str:
  defs = get_defaults(model)
  return defs.system


def dumps(v: Any) -> str:
  if isinstance(v, dict | list | tuple | set):
    return json.dumps(v, ensure_ascii=False, indent=2)
  return str(v)


@define
class Instruction:
  prompt: str = field(validator=V.min_len(1))
  system: str | None = field(factory=get_default_system_prompt)

  @cached_property
  def template(self) -> j2.Template:
    return j2.Template(self.prompt)

  def prepare(self, **context) -> list[dict[str, str]]:
    history = []
    context.update(dumps=dumps)  # special utility for jsons
    if self.system:
      history.append({"role": "system", "content": self.system})
    history.append({"role": "user", "content": self.template.render(context)})
    return history
