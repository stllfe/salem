from collections import defaultdict
from uuid import uuid4

import loguru


_LOGGER_FORMAT = (
  "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
  "<level>{level: <8}</level> | "
  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
  "<bold><level>{message}</level></bold>"
)

_LOGGER_HISTORY = defaultdict(set)


def _log_filter(record: dict) -> bool:
  if "once" in record["extra"]:
    level = record["level"].no
    message = record["message"]
    if message in _LOGGER_HISTORY[level]:
      return False
    _LOGGER_HISTORY[level].add(message)
  return True


def get_logger() -> "loguru.Logger":
  from loguru import logger
  from tqdm import tqdm

  logger.remove()
  logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=_LOGGER_FORMAT, filter=_log_filter)
  return logger


def get_short_uid(n: int = 6) -> str:
  uuid = uuid4()
  return uuid.hex[:n]
