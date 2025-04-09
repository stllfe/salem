# copied from
# https://github.com/airockchip/rknn-llm/blob/cb5b341364311065fd19eddd631a79a9f0c5afe1/examples/rkllm_server_demo/rkllm_server/flask_server.py#L231

import ctypes
import sys


# Set the dynamic library path
rkllm_lib = ctypes.CDLL("lib/librkllmrt.so")

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT = 0
RKLLMInputMode.RKLLM_INPUT_TOKEN = 1
RKLLMInputMode.RKLLM_INPUT_EMBED = 2
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL = 3

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
    ("reserved", ctypes.c_uint8 * 106),
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
  ]


class RKLLMInputUnion(ctypes.Union):
  _fields_ = [
    ("prompt_input", ctypes.c_char_p),
    ("embed_input", RKLLMEmbedInput),
    ("token_input", RKLLMTokenInput),
    ("multimodal_input", RKLLMMultiModelInput),
  ]


class RKLLMInput(ctypes.Structure):
  _fields_ = [("input_mode", ctypes.c_int), ("input_data", RKLLMInputUnion)]


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


class RKLLMResult(ctypes.Structure):
  _fields_ = [
    ("text", ctypes.c_char_p),
    ("token_id", ctypes.c_int),
    ("last_hidden_layer", RKLLMResultLastHiddenLayer),
    ("logits", RKLLMResultLogits),
  ]


# Define global variables to store the callback function output for displaying in the Gradio interface
global_text = ""
global_state = -1
split_byte_data = bytes(b"")  # Used to store the segmented byte data


# Define the callback function
def callback_impl(result, userdata, state):
  global global_text, global_state, split_byte_data
  if state == LLMCallState.RKLLM_RUN_FINISH:
    global_state = state
    print("\n")
    sys.stdout.flush()
  elif state == LLMCallState.RKLLM_RUN_ERROR:
    global_state = state
    print("run error")
    sys.stdout.flush()
  elif state == LLMCallState.RKLLM_RUN_NORMAL:
    global_state = state
    global_text += result.contents.text.decode("utf-8")


# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)


# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
  def __init__(self, model_path, lora_model_path=None, prompt_cache_path=None):
    rkllm_param = RKLLMParam()
    rkllm_param.model_path = bytes(model_path, "utf-8")

    rkllm_param.max_context_len = 4096
    rkllm_param.max_new_tokens = -1
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
    rkllm_param.extend_param.enabled_cpus_num = 4
    rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

    self.handle = RKLLM_Handle_t()

    self.rkllm_init = rkllm_lib.rkllm_init
    self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
    self.rkllm_init.restype = ctypes.c_int
    self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

    self.rkllm_run = rkllm_lib.rkllm_run
    self.rkllm_run.argtypes = [
      RKLLM_Handle_t,
      ctypes.POINTER(RKLLMInput),
      ctypes.POINTER(RKLLMInferParam),
      ctypes.c_void_p,
    ]
    self.rkllm_run.restype = ctypes.c_int

    self.set_chat_template = rkllm_lib.rkllm_set_chat_template
    self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    self.set_chat_template.restype = ctypes.c_int

    # FIXME: why it's ommited?
    system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
    prompt_prefix = "<|im_start|>user"
    prompt_postfix = "<|im_end|><|im_start|>assistant"
    # self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

    self.rkllm_destroy = rkllm_lib.rkllm_destroy
    self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
    self.rkllm_destroy.restype = ctypes.c_int

    rkllm_lora_params = None
    if lora_model_path:
      lora_adapter_name = "test"
      lora_adapter = RKLLMLoraAdapter()
      ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
      lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode("utf-8"))
      lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode("utf-8"))
      lora_adapter.scale = 1.0

      rkllm_load_lora = rkllm_lib.rkllm_load_lora
      rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
      rkllm_load_lora.restype = ctypes.c_int
      rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
      rkllm_lora_params = RKLLMLoraParam()
      rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode("utf-8"))

    self.rkllm_infer_params = RKLLMInferParam()
    ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
    self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
    self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
    self.rkllm_infer_params.keep_history = 0

    self.prompt_cache_path = None
    if prompt_cache_path:
      self.prompt_cache_path = prompt_cache_path

      rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
      rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
      rkllm_load_prompt_cache.restype = ctypes.c_int
      rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode("utf-8")))

  def run(self, prompt):
    rkllm_input = RKLLMInput()
    rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
    rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))
    self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
    return

  def release(self):
    self.rkllm_destroy(self.handle)
