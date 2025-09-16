from smolagents.models import ChatMessage
from smolagents.models import MessageRole
from smolagents.models import Model
from smolagents.models import TokenUsage
from smolagents.models import Tool


class RKLLMModel(Model):
  def __init__(self, model_id: str, **kwargs) -> None:
    from opi5.api.rkllm import RKLLMModel as RKLLMModelAPI

    self.rkllm = RKLLMModelAPI(model_id)
    super().__init__(**kwargs)

  def generate(
    self,
    messages: list[ChatMessage],
    stop_sequences: list[str] | None = None,
    response_format: dict[str, str] | None = None,
    tools_to_call_from: list[Tool] | None = None,
    **kwargs,
  ) -> ChatMessage:
    if response_format is not None:
      raise ValueError("RKLLM does not support structured outputs.")

    completion_kwargs = self._prepare_completion_kwargs(
      messages=messages,
      stop_sequences=stop_sequences,
      tools_to_call_from=tools_to_call_from,
      flatten_messages_as_text=True,
      **kwargs,
    )

    messages = completion_kwargs.pop("messages")
    prepared_stop_sequences = completion_kwargs.pop("stop", [])
    tools = completion_kwargs.pop("tools", None)
    completion_kwargs.pop("tool_choice", None)

    # prepare messages for RKLLM
    response = self.rkllm.generate(messages, tools=tools, stop_sequences=prepared_stop_sequences)

    self._last_input_token_count = response.usage.prompt_tokens
    self._last_output_token_count = response.usage.completion_tokens

    text = response.message.content
    return ChatMessage(
      role=MessageRole.ASSISTANT,
      content=text,
      raw={"out": text, "completion_kwargs": completion_kwargs},
      token_usage=TokenUsage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
      ),
    )
