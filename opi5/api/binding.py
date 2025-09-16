# modified from
# https://github.com/airockchip/rknn-llm/blob/8a4962842f2acf73a0f6f994a6c2e94a2cdfa075/examples/rkllm_server_demo/rkllm_server/flask_server.py
# https://github.com/c0zaut/RKLLM-Gradio/blob/4e01042e8f0fdf2e57d83525750a44370ad91a21/model_class.py

import ctypes
import os
import sys

from pathlib import Path


MAX_SUPPORTED_CONTEXT_LENGTH = 16384

LIB_PATH = os.path.join(os.path.dirname(__file__), "lib", "librkllmrt.so")

# Set the dynamic library path
RKLLM_LIB = ctypes.CDLL(LIB_PATH)

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInputType.RKLLM_INPUT_TOKEN = 1
RKLLMInputType.RKLLM_INPUT_EMBED = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
  _fields_ = [
    ("base_domain_id", ctypes.c_int32),
    ("embed_flash", ctypes.c_int8),
    ("enabled_cpus_num", ctypes.c_int8),
    ("enabled_cpus_mask", ctypes.c_uint32),
    ("n_batch", ctypes.c_uint8),
    ("use_cross_attn", ctypes.c_int8),
    ("reserved", ctypes.c_uint8 * 104),
  ]


class RKLLMParam(ctypes.Structure):
  _fields_ = [
    ("model_path", ctypes.c_char_p),
    ("max_context_len", ctypes.c_int32),
    ("max_new_tokens", ctypes.c_int32),
    ("top_k", ctypes.c_int32),
    ("n_keep", ctypes.c_int32),
    ("top_p", ctypes.c_float),
    ("temperature", ctypes.c_float),
    ("repeat_penalty", ctypes.c_float),
    ("frequency_penalty", ctypes.c_float),
    ("presence_penalty", ctypes.c_float),
    ("mirostat", ctypes.c_int32),
    ("mirostat_tau", ctypes.c_float),
    ("mirostat_eta", ctypes.c_float),
    ("skip_special_token", ctypes.c_bool),
    ("is_async", ctypes.c_bool),
    ("img_start", ctypes.c_char_p),
    ("img_end", ctypes.c_char_p),
    ("img_content", ctypes.c_char_p),
    ("extend_param", RKLLMExtendParam),
  ]


class RKLLMLoraAdapter(ctypes.Structure):
  _fields_ = [("lora_adapter_path", ctypes.c_char_p), ("lora_adapter_name", ctypes.c_char_p), ("scale", ctypes.c_float)]


class RKLLMEmbedInput(ctypes.Structure):
  _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]


class RKLLMTokenInput(ctypes.Structure):
  _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]


class RKLLMMultiModelInput(ctypes.Structure):
  _fields_ = [
    ("prompt", ctypes.c_char_p),
    ("image_embed", ctypes.POINTER(ctypes.c_float)),
    ("n_image_tokens", ctypes.c_size_t),
    ("n_image", ctypes.c_size_t),
    ("image_width", ctypes.c_size_t),
    ("image_height", ctypes.c_size_t),
  ]


class RKLLMInputUnion(ctypes.Union):
  _fields_ = [
    ("prompt_input", ctypes.c_char_p),
    ("embed_input", RKLLMEmbedInput),
    ("token_input", RKLLMTokenInput),
    ("multimodal_input", RKLLMMultiModelInput),
  ]


class RKLLMInput(ctypes.Structure):
  _fields_ = [
    ("role", ctypes.c_char_p),
    ("enable_thinking", ctypes.c_bool),
    ("input_type", RKLLMInputType),
    ("input_data", RKLLMInputUnion),
  ]


class RKLLMLoraParam(ctypes.Structure):
  _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
  _fields_ = [("save_prompt_cache", ctypes.c_int), ("prompt_cache_path", ctypes.c_char_p)]


class RKLLMInferParam(ctypes.Structure):
  _fields_ = [
    ("mode", RKLLMInferMode),
    ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
    ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
    ("keep_history", ctypes.c_int),
  ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
  _fields_ = [
    ("hidden_states", ctypes.POINTER(ctypes.c_float)),
    ("embd_size", ctypes.c_int),
    ("num_tokens", ctypes.c_int),
  ]


class RKLLMResultLogits(ctypes.Structure):
  _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)), ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMPerfStat(ctypes.Structure):
  _fields_ = [
    ("prefill_time_ms", ctypes.c_float),
    ("prefill_tokens", ctypes.c_int),
    ("generate_time_ms", ctypes.c_float),
    ("generate_tokens", ctypes.c_int),
    ("memory_usage_mb", ctypes.c_float),
  ]


class RKLLMResult(ctypes.Structure):
  _fields_ = [
    ("text", ctypes.c_char_p),
    ("token_id", ctypes.c_int),
    ("last_hidden_layer", RKLLMResultLastHiddenLayer),
    ("logits", RKLLMResultLogits),
    ("perf", RKLLMPerfStat),
  ]


# Define global variables to store the callback function output for displaying in the Gradio interface
global_text: list[str] = []
global_state: int = -1
global_stats: dict[str, int | float | None] = {}
split_byte_data = bytes(b"")  # Used to store the segmented byte data


def convert_perf_stats(perf: RKLLMPerfStat) -> dict[str, int | float | None]:
  d = {}
  for f, _ in perf._fields_:
    d[f] = getattr(perf, f, None)
  return d


# Define the callback function
def callback_impl(result: RKLLMResult, userdata: dict | None, state: int) -> int:
  global global_text, global_state, global_stats, split_byte_data
  if state == LLMCallState.RKLLM_RUN_FINISH:
    global_state = state
    global_stats |= convert_perf_stats(result.contents.perf)
    print("\n")
    sys.stdout.flush()
  elif state == LLMCallState.RKLLM_RUN_ERROR:
    global_state = state
    print("run error")
    sys.stdout.flush()
  elif state == LLMCallState.RKLLM_RUN_NORMAL:
    global_state = state
    global_text += result.contents.text.decode("utf-8")
  return 0


# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)


StrOrPath = str | Path


# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
  def __init__(
    self,
    model_path: StrOrPath,
    lora_model_path: StrOrPath | None = None,
    prompt_cache_path: StrOrPath | None = None,
    max_context_len: int = 4096,
    max_new_tokens: int = -1,
    top_k: int = 1,
    top_p: float = 0.9,
    temperature: float = 0.8,
    repeat_penalty: float = 1.1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    keep_history: bool = False,
  ) -> None:
    rkllm_param = RKLLMParam()
    rkllm_param.model_path = bytes(str(model_path), "utf-8")

    rkllm_param.max_context_len = 4096
    rkllm_param.max_new_tokens = 4096
    rkllm_param.skip_special_token = True
    rkllm_param.n_keep = -1
    rkllm_param.top_k = 1
    rkllm_param.top_p = 0.9
    rkllm_param.temperature = 0.8
    rkllm_param.repeat_penalty = 1.1
    rkllm_param.frequency_penalty = 0.0
    rkllm_param.presence_penalty = 0.0

    rkllm_param.mirostat = 0
    rkllm_param.mirostat_tau = 5.0
    rkllm_param.mirostat_eta = 0.1

    rkllm_param.is_async = False

    rkllm_param.img_start = "".encode("utf-8")
    rkllm_param.img_end = "".encode("utf-8")
    rkllm_param.img_content = "".encode("utf-8")

    rkllm_param.extend_param.base_domain_id = 0
    rkllm_param.extend_param.embed_flash = 1
    rkllm_param.extend_param.n_batch = 1
    rkllm_param.extend_param.use_cross_attn = 0
    rkllm_param.extend_param.enabled_cpus_num = 4
    rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

    self.handle = RKLLM_Handle_t()

    self.rkllm_init = RKLLM_LIB.rkllm_init
    self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
    self.rkllm_init.restype = ctypes.c_int
    self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

    self.rkllm_run = RKLLM_LIB.rkllm_run
    self.rkllm_run.argtypes = [
      RKLLM_Handle_t,
      ctypes.POINTER(RKLLMInput),
      ctypes.POINTER(RKLLMInferParam),
      ctypes.c_void_p,
    ]
    self.rkllm_run.restype = ctypes.c_int

    self.set_chat_template = RKLLM_LIB.rkllm_set_chat_template
    self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    self.set_chat_template.restype = ctypes.c_int

    self.set_function_tools_ = RKLLM_LIB.rkllm_set_function_tools
    self.set_function_tools_.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    self.set_function_tools_.restype = ctypes.c_int

    # system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
    # prompt_prefix = "<|im_start|>user"
    # prompt_postfix = "<|im_end|><|im_start|>assistant"
    # self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

    self.rkllm_destroy = RKLLM_LIB.rkllm_destroy
    self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
    self.rkllm_destroy.restype = ctypes.c_int

    self.rkllm_abort = RKLLM_LIB.rkllm_abort

    rkllm_lora_params = None
    if lora_model_path:
      lora_adapter_name = "test"
      lora_adapter = RKLLMLoraAdapter()
      ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
      lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode("utf-8"))
      lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode("utf-8"))
      lora_adapter.scale = 1.0

      rkllm_load_lora = RKLLM_LIB.rkllm_load_lora
      rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
      rkllm_load_lora.restype = ctypes.c_int
      rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
      rkllm_lora_params = RKLLMLoraParam()
      rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode("utf-8"))

    self.rkllm_infer_params = RKLLMInferParam()
    ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
    self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
    self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
    self.rkllm_infer_params.keep_history = int(keep_history)

    self.prompt_cache_path = None
    if prompt_cache_path:
      self.prompt_cache_path = prompt_cache_path

      rkllm_load_prompt_cache = RKLLM_LIB.rkllm_load_prompt_cache
      rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
      rkllm_load_prompt_cache.restype = ctypes.c_int
      rkllm_load_prompt_cache(self.handle, ctypes.c_char_p(str(prompt_cache_path).encode("utf-8")))

    self.tools = None

  def set_function_tools(self, system_prompt: str, tools: str, tool_response_str: str) -> None:
    # tool_response_str: Identifier tag for tool function call results, used to distinguish them
    #   from regular conversation content.
    if self.tools is None or self.tools != tools:
      self.tools = tools
      self.set_function_tools_(
        self.handle,
        ctypes.c_char_p(system_prompt.encode("utf-8")),
        ctypes.c_char_p(tools.encode("utf-8")),
        ctypes.c_char_p(tool_response_str.encode("utf-8")),
      )

  # TODO: add type hints
  def tokens_to_ctypes_array(self, tokens, ctype) -> type:
    # Converts a Python list to a ctypes array.
    # The tokenizer outputs as a Python list.
    return (ctype * len(tokens))(*tokens)

  # def run(self, *param):
  #   role, enable_thinking, prompt = param
  #   rkllm_input = RKLLMInput()
  #   rkllm_input.role = role.encode("utf-8") if role is not None else "user".encode("utf-8")
  #   rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking if enable_thinking is not None else False)
  #   rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
  #   rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))
  #   self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
  #   return

  def run(self, prompt: str | list[int], role: str | None = None, enable_thinking: bool = False) -> None:
    rkllm_input = RKLLMInput()
    rkllm_input.role = role.encode("utf-8") if role is not None else "user".encode("utf-8")
    rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)

    if isinstance(prompt, str):
      rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
      rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))
    else:
      rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_TOKEN
      # Create ctypes array and keep a reference to prevent garbage collection
      token_array = self.tokens_to_ctypes_array(prompt, ctypes.c_int32)
      # Assign pointer to the array
      rkllm_input.input_data.token_input.input_ids = ctypes.cast(token_array, ctypes.POINTER(ctypes.c_int32))
      rkllm_input.input_data.token_input.n_tokens = ctypes.c_size_t(len(prompt))
      # Keep reference to prevent garbage collection
      self._token_array_ref = token_array

    self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

  def abort(self):
    return self.rkllm_abort(self.handle)

  def release(self):
    self.rkllm_destroy(self.handle)

  # NOTE: this is moved to a higher-level API in rkllm.py
  # def get_RKLLM_output(self, message, history):
  #   # Link global variables to retrieve the output information from the callback function
  #   global global_text, global_state
  #   global_text = []
  #   global_state = -1
  #   user_prompt = {"role": "user", "content": message}
  #   history.append(user_prompt)
  #   # Gemma 2 does not support system prompt.
  #   if self.system_prompt == "":
  #     prompt = [user_prompt]
  #   else:
  #     prompt = [{"role": "system", "content": self.system_prompt}, user_prompt]
  #   # print(prompt)
  #   TOKENIZER_PATH = "%s/%s" % (MODEL_PATH, self.st_model_id.replace("/", "-"))
  #   if not os.path.exists(TOKENIZER_PATH):
  #     print("Tokenizer not cached locally, downloading to %s" % TOKENIZER_PATH)
  #     os.mkdir(TOKENIZER_PATH)
  #     tokenizer = AutoTokenizer.from_pretrained(self.st_model_id, trust_remote_code=True)
  #     tokenizer.save_pretrained(TOKENIZER_PATH)
  #   else:
  #     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
  #   prompt = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
  #   # response = {"role": "assistant", "content": "Loading..."}
  #   response = {"role": "assistant", "content": ""}
  #   history.append(response)
  #   model_thread = threading.Thread(target=self.run, args=(prompt,))
  #   model_thread.start()
  #   model_thread_finished = False
  #   while not model_thread_finished:
  #     while len(global_text) > 0:
  #       response["content"] += global_text.pop(0)
  #       # Marco-o1
  #       response["content"] = str(response["content"]).replace("<Thought>", "\\<Thought\\>")
  #       response["content"] = str(response["content"]).replace("</Thought>", "\\<\\/Thought\\>")
  #       response["content"] = str(response["content"]).replace("<Output>", "\\<Output\\>")
  #       response["content"] = str(response["content"]).replace("</Output>", "\\<\\/Output\\>")
  #       time.sleep(0.005)
  #       # Gradio automatically pushes the result returned by the yield statement when calling the then method
  #       yield response
  #     model_thread.join(timeout=0.005)
  #     model_thread_finished = not model_thread.is_alive()
