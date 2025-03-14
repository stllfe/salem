import inspect
import json
import re

from collections.abc import Callable
from types import ModuleType
from typing import Any, Type, Union

from docstring_parser import parse


NoneType: Type = type(None)


class DateTimeJsonEncoder(json.JSONEncoder):
  def default(self, obj: Any) -> Any:
    import datetime as dt

    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
      return obj.isoformat()
    elif isinstance(obj, dt.timedelta):
      return (dt.datetime.min + obj).time().isoformat()
    return super(DateTimeJsonEncoder, self).default(obj)


def flatten_string(s: str) -> str:
  s = s.replace("\n", " ")
  return re.sub(r"\s+", " ", s)


def format_docstring(doc: str) -> str:
  for m in re.finditer(r"(^\s*(Args?|Examples?|Raises?|Returns?):\s*$)", doc, flags=re.MULTILINE):
    doc = doc[: m.start()]
    break
  return flatten_string(doc.strip())


def extract_type_info(annotation: Any) -> dict[str, Any]:
  """Convert Python type annotations to JSON Schema type representations.

  Supports basic types, Optional, Union, and custom types.
  """
  # handle optional types
  if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
    # check if None is one of the Union types (indicates Optional)
    if NoneType in annotation.__args__:
      # remove None and get the actual type
      non_none_types = [t for t in annotation.__args__ if t is not NoneType]
      if len(non_none_types) == 1:
        annotation = non_none_types[0]

  type_map = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
  }

  # handle special string-like types (like CURRENT enum)
  if hasattr(annotation, "__name__") and annotation.__name__ == "str":
    return {"type": "string"}

  # check for string-based literal types
  if str(annotation).startswith("typing.Literal"):
    # extract enum values
    values = annotation.__args__
    return {"type": "string", "enum": list(values)}

  # check if the type is in our type map
  if annotation in type_map:
    return type_map[annotation]

  # fallback to string for custom types
  return {"type": "string"}


def get_func_name(fn: Callable, sep: str = ".") -> str:
  fnname = fn.__name__.strip()
  module = fn.__module__.split(".")[-1]
  return f"{module}{sep}{fnname}"


# TODO: refactor and cleanup this function, looks messy
def get_tool_schema(fn: Callable, *, openai: bool = False) -> dict[str, Any]:
  """Convert a Python function to a function calling compatible schema."""

  sig = inspect.signature(fn)
  doc = inspect.getdoc(fn) or ""

  parameters = {"type": "object", "properties": {}, "required": []}

  for name, param in sig.parameters.items():
    if name in ["self", "cls"]:
      continue

    required = param.default is inspect.Parameter.empty
    default = param.default

    try:
      type_info = extract_type_info(param.annotation)
    except Exception:
      type_info = {"type": "string"}

    # try to extract parameter description from docstring
    desc = ""
    if doc:
      parsed_doc = parse(doc)
      for p in parsed_doc.params:
        if p.arg_name == name:
          desc = flatten_string(p.description or "")
          break

    if required:
      parameters["required"].append(name)
    else:
      desc += f" (default: {default})" if default else ""

    parameters["properties"][name] = {**type_info, "description": desc}

  returns_desc = extract_type_info(sig.return_annotation).get("type", "")
  if doc:
    parsed_doc = parse(doc)
    if parsed_doc.returns and parsed_doc.returns.description:
      returns_desc = parsed_doc.returns.description
  returns_desc = flatten_string(returns_desc)

  schema = {
    "name": get_func_name(fn, sep="_" if openai else "."),
    "description": format_docstring(doc.strip()),  # used to be -> doc.split("\n")[0] if doc else ""
    "parameters": parameters,
  }
  if not openai:
    schema.update(returns=returns_desc)
    return schema

  schema = {"type": "function", "function": schema}
  return schema


def get_public_functions(module: ModuleType) -> list[Callable]:
  def include(name: str) -> bool:
    if name.startswith("_"):
      return False
    member = getattr(module, name)
    return inspect.isfunction(member) and member.__module__ == module.__name__

  return [getattr(module, name) for name in dir(module) if include(name)]


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
  if len(content) <= max_length:
    return content
  else:
    return (
      content[: max_length // 2]
      + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
      + content[-max_length // 2 :]
    )
