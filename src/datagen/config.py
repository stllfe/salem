import re

from os import getenv

from src.datagen.types import GenerationArgs
from src.utils import get_logger


MODEL = getenv("MODEL", "qwen/Qwen2.5-32B-Instruct")

DEFAULTS: dict[str | re.Pattern, GenerationArgs] = {
  re.compile(r"qwen/Qwen2.5-\d{1,2}B-Instruct"): GenerationArgs(
    temperature=0.7,
    top_p=0.8,
    min_p=0.15,
    repetition_penalty=1.05,
  ),
}


logger = get_logger()


def genargs_from_env(model: str | None = None) -> GenerationArgs:
  model = model or MODEL
  for tag, args in DEFAULTS.items():
    if re.match(tag, model):
      logger.info(f"Found preconfigured generation args: {model}", once=True)
      logger.info(f"{args!r}", once=True)
      return args
  logger.info(f"Using default generation args for model: {model}", once=True)
  args = GenerationArgs()
  logger.info(f"{args!r}", once=True)
  return args
