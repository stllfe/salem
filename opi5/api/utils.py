from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import msgspec

from msgspec import Struct
from msgspec.json import decode as decode_json
from msgspec.json import encode as encode_json
from msgspec.yaml import decode as decode_yaml
from msgspec.yaml import encode as encode_yaml
from omegaconf import OmegaConf as oc


StrOrPath = str | Path


# hotfix for Path:
# https://github.com/jcrist/msgspec/issues/530
def dec_hook(c: type, obj: Any) -> Any:
  if c is Path:
    return Path(obj).resolve()
  return obj


class Serializable(Struct):
  """Adds serialization methods to a basic `msgspec.Struct`"""

  @classmethod
  def from_json(cls, j: StrOrPath, strict: bool = True) -> Self:
    j_str = j.read_text("utf-8") if isinstance(j, Path) else j
    return decode_json(j_str, type=cls, strict=strict, dec_hook=dec_hook)

  @classmethod
  def from_yaml(cls, y: StrOrPath, strict: bool = True) -> Self:
    conf = oc.load(y)
    y_str = oc.to_yaml(conf, resolve=True)
    return msgspec.convert(decode_yaml(y_str), cls, strict=strict, dec_hook=dec_hook)

  @classmethod
  def from_dict(cls, d: dict[str, Any], strict: bool = True) -> Self:
    return msgspec.convert(d, cls, strict=strict, dec_hook=dec_hook)

  def to_yaml(self) -> bytes:
    return encode_yaml(self)

  def to_json(self) -> bytes:
    return encode_json(self)

  def to_dict(self) -> dict[str, Any]:
    return msgspec.to_builtins(self)
