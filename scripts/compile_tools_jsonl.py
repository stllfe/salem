import importlib.util
import inspect
import json
import sys

from pathlib import Path
from types import ModuleType

from tools.utils import get_tool_schema


ROOT = Path(__file__).parent.parent
CORE = ROOT.joinpath("tools").joinpath("core")
DEST = ROOT.joinpath("data").joinpath("tools")


def should_export(m: ModuleType, name: str) -> bool:
  if name.startswith("_"):
    return False
  member = getattr(m, name)
  return inspect.isfunction(member) and member.__module__ == m.__name__


def convert_module_to_jsonl(module: ModuleType, fpath: Path) -> None:
  functions = [getattr(module, name) for name in dir(module) if should_export(module, name)]
  tool_schemas = [get_tool_schema(fn) for fn in functions]

  if not tool_schemas:
    return

  with open(fpath, mode="w") as fd:
    for schema in tool_schemas:
      fd.write(json.dumps(schema, ensure_ascii=False) + "\n")


def main(root: Path = CORE, dest: Path = DEST, module: str | None = None) -> None:
  """Convert all functions in a module to a JSONL file for function calling."""

  if not root.is_dir():
    raise NotADirectoryError(root)

  if module:
    modules = [root.joinpath(module).with_suffix(".py")]
  else:
    modules = root.glob("*.py")

  dest.mkdir(parents=True, exist_ok=True)
  sys.path.append(root.as_posix())
  for path in modules:
    # skip init modules
    if path.stem == "__init__":
      continue
    print(f"Convering {path.stem!r} ...", end=" ")
    s = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    p = dest.joinpath(path.name).with_suffix(".jsonl")
    try:
      convert_module_to_jsonl(m, p)
    except Exception as err:
      print(f"error! {err}")
    else:
      print(f"ok! -> {p}")


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
