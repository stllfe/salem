from collections.abc import Callable
from typing import Any

from tools.runtime import call
from tools.runtime import runtime


def convert_to_tool(fn: Callable) -> Callable:
  from functools import wraps

  from smolagents import tool

  @tool
  @wraps(fn)
  def wrapper(*args, **kwargs) -> Any:
    return call(fn, runtime, *args, **kwargs)

  return wrapper
