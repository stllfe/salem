from collections.abc import Iterable
from pathlib import Path

import loguru
import orjsonl
import yaml

from src.datagen.types import Instruction


LOGGER_FORMAT = (
  "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
  "<level>{level: <8}</level> | "
  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
  "<bold><level>{message}</level></bold>"
)


def read_prompt(p: Path) -> Instruction:
  raw = yaml.safe_load(p.read_bytes())
  if not isinstance(raw, dict):
    raise ValueError(f"incorrect instruction prompt: {p}")
  system = raw.pop("system", None)
  prompt = raw.pop("prompt")
  if len(raw) > 0:
    raise ValueError(f"unknown prompt keys: {raw}")
  return Instruction(prompt, system)


def read_jsonl(p: Path) -> Iterable[dict]:
  return orjsonl.stream(p)


def get_logger() -> "loguru.Logger":
  from loguru import logger
  from tqdm import tqdm

  logger.remove()
  logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=LOGGER_FORMAT)
  return logger
