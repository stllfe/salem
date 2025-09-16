import os

from smolagents.models import LiteLLMModel
from smolagents.models import OpenAIServerModel

import smol.monitor  # noqa

from smol.rknn_model import RKLLMModel


match os.getenv("MODEL"):
  case "gpt-4o":
    model = OpenAIServerModel("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
  case "claude":
    model = LiteLLMModel(model_id="anthropic/claude-3-7-sonnet-latest", api_key=os.environ["ANTHROPIC_API_KEY"])
  case "rkllm":
    import os
    import resource

    os.environ["RKLLM_LOG_LEVEL"] = "2"  # will print token stats
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    model = RKLLMModel("qwen2.5:3B")
  case _:
    model = OpenAIServerModel("default", api_base="http://localhost:3000/v1", api_key="[EMPTY]", tool_choice="auto")


MAX_WEB_STEPS = 5
MAX_CALENDAR_STEPS = 3
