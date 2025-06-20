# copied from
# https://github.com/airockchip/rknn-llm/blob/8a4962842f2acf73a0f6f994a6c2e94a2cdfa075/examples/rkllm_server_demo/rkllm_server/flask_server.py

import ctypes
import os
import sys


lib_path = os.path.join(os.path.dirname(__file__), "lib", "librkllmrt.so")

# Set the dynamic library path
rkllm_lib = ctypes.CDLL(lib_path)

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
global_text = []
global_state = -1
split_byte_data = bytes(b"")  # Used to store the segmented byte data


# Define the callback function
def callback_impl(result, userdata, state):
  global global_text, global_state, split_byte_data
  # print(f"{state=}")
  if state == LLMCallState.RKLLM_RUN_FINISH:
    global_state = state
    print("\n")  # helpful when RKLLM_LOG_LEVEL is set and stats are printed right after the response
    sys.stdout.flush()
  elif state == LLMCallState.RKLLM_RUN_ERROR:
    global_state = state
    print("run error")
    sys.stdout.flush()
  # elif state == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
  #   """
  #       If using the GET_LAST_HIDDEN_LAYER function, the callback interface will return the memory pointer: last_hidden_layer, the number of tokens: num_tokens, and the size of the hidden layer: embd_size.
  #       With these three parameters, you can retrieve the data from last_hidden_layer.
  #       Note: The data needs to be retrieved during the current callback; if not obtained in time, the pointer will be released by the next callback.
  #       """
  #   if result.last_hidden_layer.embd_size != 0 and result.last_hidden_layer.num_tokens != 0:
  #     data_size = (
  #       result.last_hidden_layer.embd_size * result.last_hidden_layer.num_tokens * ctypes.sizeof(ctypes.c_float)
  #     )
  #     print(f"data_size: {data_size}")
  #     global_text.append(f"data_size: {data_size}\n")
  #     output_path = os.getcwd() + "/last_hidden_layer.bin"
  #     with open(output_path, "wb") as outFile:
  #       data = ctypes.cast(result.last_hidden_layer.hidden_states, ctypes.POINTER(ctypes.c_float))
  #       float_array_type = ctypes.c_float * (data_size // ctypes.sizeof(ctypes.c_float))
  #       float_array = float_array_type.from_address(ctypes.addressof(data.contents))
  #       outFile.write(bytearray(float_array))
  #       print(f"Data saved to {output_path} successfully!")
  #       global_text.append(f"Data saved to {output_path} successfully!")
  #   else:
  #     print("Invalid hidden layer data.")
  #     global_text.append("Invalid hidden layer data.")
  #   global_state = state
  #   time.sleep(0.05)
  #   sys.stdout.flush()
  else:
    # Save the output token text and the RKLLM running state
    global_state = state
    # Monitor if the current byte data is complete; if incomplete, record it for later parsing
    try:
      if split_byte_data is None or split_byte_data == "":
        global_text.append((b"" + result.contents.text).decode("utf-8"))
        # print((split_byte_data + result.contents.text).decode("utf-8"), end="")
        split_byte_data = bytes(b"")
      else:
        global_text.append((split_byte_data + result.contents.text).decode("utf-8"))
        # print((split_byte_data + result.contents.text).decode("utf-8"), end="")
        split_byte_data = bytes(b"")
    except:
      if result.contents.text is not None:
        split_byte_data += result.contents.text
    sys.stdout.flush()


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

    # system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
    # prompt_prefix = "<|im_start|>user"
    # prompt_postfix = "<|im_end|><|im_start|>assistant"
    # self.set_chat_template(self.handle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "/no_think<|im_end|>\n<|im_start|>assistant\n");
    # self.set_chat_template(
    #   self.handle,
    #   ctypes.c_char_p(system_prompt.encode("utf-8")),
    #   ctypes.c_char_p(prompt_prefix.encode("utf-8")),
    #   ctypes.c_char_p(prompt_postfix.encode("utf-8")),
    # )

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

  def run(self, prompt: str) -> None:
    rkllm_input = RKLLMInput()
    rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
    rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))
    self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
    return

  def release(self) -> None:
    self.rkllm_destroy(self.handle)

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
