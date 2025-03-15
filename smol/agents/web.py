from smolagents import ToolCallingAgent

from salem.tools.core import web
from salem.tools.utils import get_public_functions
from smol import config
from smol.utils import convert_to_tool


tools = [convert_to_tool(fn) for fn in get_public_functions(web)]


browser = ToolCallingAgent(
  tools=tools,
  model=config.model,
  max_steps=config.MAX_WEB_STEPS,
  name="browser",
  description="Runs web searches for you. Give it your query as an argument.",
)
