from collections.abc import Callable
from typing import Any

from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from smolagents.prompts import CODE_SYSTEM_PROMPT

from tools.core import calendar
from tools.core import weather
from tools.core import web
from tools.runtime import CURRENT
from tools.runtime import call
from tools.runtime import runtime
from tools.utils import get_public_functions


def convert_to_tool(fn: Callable) -> Callable:
  from functools import wraps

  from smolagents import tool

  @tool
  @wraps(fn)
  def wrapper(*args, **kwargs) -> Any:
    return call(fn, runtime, *args, **kwargs)

  return wrapper


MODULES = [calendar, web, weather]
TOOLS = [convert_to_tool(fn) for module in MODULES for fn in get_public_functions(module)]


SYSTEM_PROMPT_ADD = (
  runtime.resolve("""
<%text>
Use the same language as a user messages or provided input. For example, russian, for russian inputs or user requests.

---
<%text>
## SYSTEM STATUS
</%text>
User name: Олег
Current date: {CURRENT.DATE} (ISO 8061)
Current time: {CURRENT.TIME} (ISO 8061)
Current language: {CURRENT.LANGUAGE}
Current location: {CURRENT.LOCATION}
""")
  + f"""
---
You can use these dynamic variables as strings when you want to reference some current values:
{",\n".join(CURRENT)}
"""
)

agent = CodeAgent(
  tools=TOOLS,
  model=OpenAIServerModel("default", api_base="http://localhost:3001/v1", api_key="[EMPTY]"),
  system_prompt=f"{CODE_SYSTEM_PROMPT}\n{SYSTEM_PROMPT_ADD}",
)

while True:
  try:
    user = input("$> ").strip()
    answer = agent.run(user)
  except KeyboardInterrupt:
    break
