import json

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjsonl
import yaml

from salem.datagen.types import Instruction


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


def dumps(v: Any) -> str:
  if isinstance(v, dict | list | tuple | set):
    return json.dumps(v, ensure_ascii=False, indent=2)
  return str(v)
