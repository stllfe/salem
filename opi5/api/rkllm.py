# TODO: make a higher API available here like the following:
# https://github.com/c0zaut/RKLLM-Gradio/blob/main/model_class.py
import enum
import threading
import time

from collections.abc import Callable, Sequence
from pathlib import Path

import msgspec

from loguru import logger

from api.binding import RKLLM
from api.utils import ModelInfo
from api.utils import Serializable
from api.utils import model_registry


CAPI_POLL_DELAY = 0.005
MAX_SUPPORTED_CONTEXT_LENGTH = 16384

StrOrPath = str | Path


class GenerationConfig(Serializable):
  max_length: int = 4096
  # FIXME: MAX_SUPPORTED_CONTEXT_LEN doesn't work if the context_len in .rknn model is smaller
  max_new_tokens: int = -1  # is it actually present in HF generation configs?
  top_k: int = 1
  top_p: float = 0.9
  temperature: float = 0.7
  repetition_penalty: float = 1.1
  frequency_penalty: float = 0.0


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
  def __init__(self, model: str | ModelInfo) -> None:
    from transformers import AutoTokenizer
    from transformers import PreTrainedTokenizer

    if isinstance(model, str):
      model = model_registry.get_model(model)

    model_dir = model_registry.get_model_dir(model)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_dir)

    gen_config = GenerationConfig()

    hf_gen_config = model_dir.joinpath("generation_config.json")
    if hf_gen_config.exists():
      gen_config = GenerationConfig.from_json(hf_gen_config, strict=False)

    self.rkllm = RKLLM(
      model_dir / model.filename,
      temperature=gen_config.temperature,
      top_k=gen_config.top_k,
      top_p=gen_config.top_p,
      max_context_len=gen_config.max_length,
      repeat_penalty=gen_config.repetition_penalty,
      frequency_penalty=gen_config.frequency_penalty,
    )
    # TODO: how to get the init status? Cause it can silently fail on some params mismatch
    self.tokenizer = tokenizer
    self.model_info = model
    self.gen_config = gen_config

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
      [message.to_dict() if isinstance(message, ChatMessage) else message for message in messages],
      tokenize=True,
      add_generation_prompt=True,
    )

    if debug:
      logger.debug("Input prompt:\n{}", self.tokenizer.decode(prompt))

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
          time.sleep(CAPI_POLL_DELAY)
        gen_thread.join(timeout=CAPI_POLL_DELAY)
        gen_finished = not gen_thread.is_alive()
      content = "".join(output)
      return ChatMessage(ChatRole.ASSISTANT, content)
    finally:
      self.rkllm.release()

  def __call__(self, *args, **kwargs):
    return self.generate(*args, **kwargs)
