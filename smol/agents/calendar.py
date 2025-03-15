# noqa: A005
from smolagents import ToolCallingAgent

from salem.tools.core import calendar
from salem.tools.utils import get_public_functions
from smol import config
from smol.utils import convert_to_tool


tools = [convert_to_tool(fn) for fn in get_public_functions(calendar)]


calendar = ToolCallingAgent(
  tools=tools,
  model=config.model,
  max_steps=config.MAX_CALENDAR_STEPS,
  name="calendar",
  description="Manages your events and reminders. Give it your query as an argument.",
)
