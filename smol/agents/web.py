from smolagents import ToolCallingAgent

from smol.config import model
from smol.utils import convert_to_tool
from tools.core import web
from tools.utils import get_public_functions


tools = [convert_to_tool(fn) for fn in get_public_functions(web)]


browser = ToolCallingAgent(
  tools=tools,
  model=model,
  max_steps=10,
  name="browser",
  description="Runs web searches for you. Give it your query as an argument.",
)
