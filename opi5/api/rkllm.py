# TODO: make a higher API available here like the following:
# https://github.com/c0zaut/RKLLM-Gradio/blob/main/model_class.py
import enum
import threading
import time

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import msgspec

from api.binding import RKLLM
from api.models import ModelInfo
from api.models import model_registry
from api.utils import Serializable
from loguru import logger


RKNN_API_POLL_DELAY = 0.005
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


class TokenUsage(Serializable):
  prompt_tokens: int | None = None
  completion_tokens: int | None = None


class ChatMessage(Serializable):
  role: ChatRole
  content: str


class ChatResponse(Serializable):
  message: ChatMessage
  usage: TokenUsage = msgspec.field(default_factory=TokenUsage)
  stop_reason: StopReason = msgspec.field(default=StopReason.STOP_TOKEN)


StreamCallback = Callable[[str], None]


class RKLLMModel:
  def __init__(self, model: str | ModelInfo, prompt_cache_path: StrOrPath | None = None) -> None:
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
      prompt_cache_path=prompt_cache_path,
      temperature=gen_config.temperature,
      top_k=gen_config.top_k,
      top_p=gen_config.top_p,
      max_context_len=gen_config.max_length,
      repeat_penalty=gen_config.repetition_penalty,
      frequency_penalty=gen_config.frequency_penalty,
      keep_history=True,
    )
    # TODO: how to get the init status? Cause it can silently fail on some params mismatch
    self.tokenizer = tokenizer
    self.model_info = model
    self.gen_config = gen_config

    self.last_token_usage: TokenUsage | None = None
    self.last_stop_reason: StopReason | None = None

  def generate(
    self,
    messages: list[ChatMessage | dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    stop_sequences: Sequence[str] | None = None,
    stream_callback: StreamCallback | None = None,
    debug: bool = False,
    **kwargs,
  ) -> ChatResponse:
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

    from api.binding import global_stats
    from api.binding import global_text

    def prepare_message(m: ChatMessage | dict[str, Any]) -> dict[str, Any]:
      assert isinstance(m, (ChatMessage, dict))  # noqa
      if isinstance(m, dict):
        cm = ChatMessage.from_dict(m)
        return cm.to_dict()
      return m.to_dict()

    conversation = list(map(prepare_message, messages))
    # NOTE: smolagents write a huge system prompt with all tools listed already
    prompt = self.tokenizer.apply_chat_template(
      conversation,
      tools=tools,
      tokenize=True,
      add_generation_prompt=True,
    )

    if debug:
      logger.debug("Input prompt:\n{}", self.tokenizer.decode(prompt))

    # TODO: I don't understand what's the reason to do that anyways if we use our tokenizer here
    # if tools is not None:
    #   import json

    #   system_prompt = next((m["content"] for m in conversation if str(m["role"]) == "system"), "")
    #   self.rkllm.set_function_tools(system_prompt, json.dumps(tools), "tool_response")

    output: list[str] = []
    stops = set(stop_sequences or [])

    gen_thread = threading.Thread(target=self.rkllm.run, args=(prompt,), kwargs=kwargs)
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
        time.sleep(RKNN_API_POLL_DELAY)
      gen_thread.join(timeout=RKNN_API_POLL_DELAY)
      gen_finished = not gen_thread.is_alive()
    content = "".join(output)
    usage = TokenUsage(
      prompt_tokens=global_stats.get("prefill_tokens"),
      completion_tokens=global_stats.get("generate_tokens"),
    )
    message = ChatMessage(ChatRole.ASSISTANT, content)
    response = ChatResponse(message, usage)

    self.last_token_usage = usage
    self.last_stop_reason = response.stop_reason
    return response

  def __call__(self, *args, **kwargs):
    return self.generate(*args, **kwargs)

  def generate_stream(
    self,
    messages: list[ChatMessage | dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    stop_sequences: Sequence[str] | None = None,
    debug: bool = False,
    **kwargs,
  ) -> Iterable[str]:
    import queue

    q: queue.Queue[str] = queue.Queue()

    generate = partial(
      self.generate,
      tools=tools,
      stop_sequences=stop_sequences,
      stream_callback=q.put,
      debug=debug,
      **kwargs,
    )
    producer = threading.Thread(target=generate, args=(messages,))

    def consumer() -> Iterable[str]:
      while producer.is_alive():
        try:
          yield q.get(block=True, timeout=RKNN_API_POLL_DELAY)
        except queue.Empty:
          continue

    try:
      producer.start()
      yield from consumer()
    finally:
      producer.join()

  def release(self) -> None:
    self.rkllm.release()
