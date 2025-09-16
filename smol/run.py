import importlib
import importlib.resources

import yaml

from smolagents import CodeAgent

from salem.tools.runtime import CURRENT
from salem.tools.runtime import runtime
from smol.agents.calendar import calendar
from smol.agents.web import browser
from smol.config import model


CODE_AGENT_PROMPTS = yaml.safe_load(
  importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
)
SYSTEM_PROMPT_ADD = f"""
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
"""

DYNAMIC_VARIABLES_HINT = f"""
---
You can use these dynamic variables as strings when you want to reference some current values:
{",\n".join(CURRENT)}
"""

CODE_AGENT_PROMPTS["system_prompt"] = (
  f"{CODE_AGENT_PROMPTS['system_prompt']}\n{runtime.resolve(SYSTEM_PROMPT_ADD)}\n{DYNAMIC_VARIABLES_HINT}"
)

agent = CodeAgent(tools=[], model=model, prompt_templates=CODE_AGENT_PROMPTS, managed_agents=[calendar, browser])

while True:
  try:
    # user = input("$> ").strip()
    # if not user or user == "/exit":
    #   break
    user = "who's david laid"
    answer = agent.run(user)
  except KeyboardInterrupt:
    break
