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


def include(module: ModuleType, name: str) -> bool:
  if name.startswith("_"):
    return False
  member = getattr(module, name)
  return inspect.isfunction(member) and member.__module__ == module.__name__


def load_module(mpath: Path) -> ModuleType:
  s = importlib.util.spec_from_file_location(mpath.stem, mpath)
  m = importlib.util.module_from_spec(s)
  s.loader.exec_module(m)
  return m


def module_to_jsonl(module: ModuleType, fpath: Path) -> None:
  functions = [getattr(module, name) for name in dir(module) if include(module, name)]
  schemas = [get_tool_schema(fn) for fn in functions]

  if not schemas:
    return

  with fpath.with_suffix(".jsonl").open(mode="w") as fd:
    for schema in schemas:
      fd.write(json.dumps(schema, ensure_ascii=False) + "\n")


def main(root: Path = CORE, dest: Path = DEST, module: str | None = None) -> None:
  """Convert all functions in a module to a JSONL schema file for function calling."""

  if not root.is_dir():
    raise NotADirectoryError(root)

  if module:
    modules = [root.joinpath(module).with_suffix(".py")]
  else:
    modules = root.glob("*.py")

  dest.mkdir(parents=True, exist_ok=True)
  sys.path.append(root.as_posix())
  for path in modules:
    if path.stem == "__init__":  # skip init modules
      continue
    print(f"Compiling '{path.stem}'", end=" ", flush=True)
    m = load_module(path)
    p = dest.joinpath(path.stem).with_suffix(".jsonl")
    try:
      module_to_jsonl(m, p)
    except Exception as err:
      print(f"error ! {err}")
    else:
      print(f"ok → {p}")
  print("Done!")


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
