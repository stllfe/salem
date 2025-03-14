# noqa: A005
from smolagents import ToolCallingAgent

from smol.config import model
from smol.utils import convert_to_tool
from tools.core import calendar
from tools.utils import get_public_functions


tools = [convert_to_tool(fn) for fn in get_public_functions(calendar)]


calendar = ToolCallingAgent(
  tools=tools,
  model=model,
  max_steps=10,
  name="calendar",
  description="Manages your events and reminders. Give it your query as an argument.",
)
