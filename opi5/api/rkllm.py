# TODO: make a higher API available here like the following:
# https://github.com/c0zaut/RKLLM-Gradio/blob/main/model_class.py
import enum
import threading
import time

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import msgspec

from loguru import logger

from api.binding import RKLLM


API_POLL_DELAY = 0.005

StrOrPath = str | Path


class Serializable(msgspec.Struct):
  """Adds serialization methods to a basic `msgspec.Struct`"""

  def asjson(self) -> bytes:
    from msgspec.json import encode

    return encode(self)

  def asdict(self) -> dict[str, Any]:
    return msgspec.to_builtins(self)


class GenerationParams(Serializable):
  max_context_len: int
  max_new_tokens: int
  top_k: int
  top_p: float
  temperature: float
  repeat_penalty: float
  frequency_penalty: float
  system_prompt: str


class ModelConfig(Serializable):
  name: str
  size: float
  filepath: StrOrPath
  tokenizer: StrOrPath
  params: GenerationParams | None = None


class ChatRole(enum.StrEnum):
  SYSTEM = enum.auto()
  USER = enum.auto()
  ASSISTANT = enum.auto()


class StopReason(enum.StrEnum):
  MAX_LENGTH = enum.auto()
  STOP_TOKEN = enum.auto()
  ERROR = enum.auto()


class ChatMessage(Serializable):
  role: ChatRole
  content: str
  stop_reason: StopReason = msgspec.field(default=StopReason.STOP_TOKEN)


StreamCallback = Callable[[str], None]


class RKLLMModel:
  def __init__(self, rkllm: RKLLM, config: ModelConfig) -> None:
    from transformers import AutoTokenizer
    from transformers import PreTrainedTokenizer

    self.rkllm = rkllm
    self.config = config
    self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

  @classmethod
  def from_config(cls, config: ModelConfig) -> "RKLLMModel":
    rkllm = RKLLM(config.filepath)
    return cls(rkllm, config)

  def generate(
    self,
    messages: list[ChatMessage | dict[str, str]],
    stop_sequences: Sequence[str] | None = None,
    stream_callback: StreamCallback | None = None,
    debug: bool = False,
    **kwargs,
  ) -> ChatMessage:
    """Process the input messages and return the model's response.

    Args:
      messages: A list of message dictionaries to be processed. Each dictionary should have the structure:
        `{"role": "user/system", "content": "message content"}`.
      stop_sequences: A list of strings that will stop the generation if encountered in the model's output.
      stream_callback: A callback to apply on each new token when generating.
      **kwargs: Additional keyword arguments to be passed to the underlying model.
    Returns:
      A chat message object containing the model's response.
    """
    from api.binding import global_text

    prompt = self.tokenizer.apply_chat_template(
      [message.asdict() if isinstance(message, ChatMessage) else message for message in messages],
      tokenize=True,
      add_generation_prompt=True,
    )

    if debug:
      logger.debug("Detokenized prompt:\n{}", self.tokenizer.decode(prompt))

    output: list[str] = []
    stops = set(stop_sequences or [])
    try:
      gen_thread = threading.Thread(target=self.rkllm.run, args=(prompt,))
      gen_thread.start()
      gen_finished = False
      while not gen_finished:
        while len(global_text) > 0:
          token = global_text.pop(0)
          output.append(token)
          if stream_callback:
            stream_callback(token)
          if token in stops:
            break
          time.sleep(API_POLL_DELAY)
        gen_thread.join(timeout=API_POLL_DELAY)
        gen_finished = not gen_thread.is_alive()
      content = "".join(output)
      return ChatMessage(ChatRole.ASSISTANT, content)
    finally:
      self.rkllm.release()

  def __call__(self, *args, **kwargs):
    return self.generate(*args, **kwargs)
