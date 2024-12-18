# noqa: A005
from functools import cached_property
from os import getenv
from typing import Literal

import jinja2 as j2

from attrs import define
from attrs import field
from attrs import validators as V


SEED = int(getenv("SEED", "25512"))
MAX_TOKENS = int(getenv("MAX_TOKENS", "2048"))


Language = Literal["russian", "english"]


@define
class Instruction:
  prompt: str = field(validator=V.min_len(1))
  system: str | None = None

  @cached_property
  def template(self) -> j2.Template:
    return j2.Template(self.prompt)

  def prepare(self, **context) -> list[dict[str, str]]:
    chat = []
    if self.system:
      chat.append({"role": "system", "content": self.system})
    chat.append({"role": "user", "content": self.template.render(context)})
    return chat


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
