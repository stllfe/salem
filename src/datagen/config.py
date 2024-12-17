import re

from os import getenv

from loguru import logger

from src.datagen.types import GenerationArgs


DEFAULTS: dict[str | re.Pattern, GenerationArgs] = {
  re.compile(r"qwen/Qwen2.5-\d{1,2}B-Instruct"): GenerationArgs(
    temperature=0.7,
    top_p=0.8,
    min_p=0.15,
    repetition_penalty=1.05,
  ),
}


def genargs_from_env() -> GenerationArgs:
  model = getenv("MODEL")
  args = GenerationArgs()
  for tag in DEFAULTS:
    if re.match(tag, model):
      logger.info(f"Found preconfigured generation args: {model}")
      args = DEFAULTS[tag]
      break
  else:
    logger.info(f"Using default generation args for model: {model}")
  logger.info(f"{args!r}")
  return args
