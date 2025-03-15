from collections.abc import Callable
from typing import Any

from smolagents import Tool

from salem.tools.runtime import call
from salem.tools.runtime import runtime


def convert_to_tool(fn: Callable) -> Tool:
  from functools import wraps

  from smolagents import tool

  @tool
  @wraps(fn)
  def wrapper(*args, **kwargs) -> Any:
    if hasattr(fn, "__runtime_callable__"):
      return fn(*args, **kwargs)
    return call(fn, runtime, *args, **kwargs)

  return wrapper
