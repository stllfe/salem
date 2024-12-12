import inspect

from collections.abc import Callable
from typing import Any, Type, Union

from docstring_parser import parse


NoneType: Type = type(None)


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


def get_tool_schema(fn: Callable) -> dict[str, Any]:
  """Convert a Python function to a function calling compatible schema."""

  sig = inspect.signature(fn)
  doc = inspect.getdoc(fn) or ""

  fnname = fn.__name__
  module = fn.__module__.split(".")[-1]

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
          desc = p.description or ""
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

  return {
    "name": f"{module}.{fnname}",
    "description": doc.split("\n")[0] if doc else "",
    "parameters": parameters,
    "returns": returns_desc,
  }
