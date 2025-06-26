import os
import resource
import subprocess

from functools import partial
from pathlib import Path


os.environ["RKLLM_LOG_LEVEL"] = "1"  # will print token stats

from api.rkllm import ModelConfig
from api.rkllm import RKLLMModel


root_dir = Path(__file__).parent
models_dir = root_dir.joinpath("models")
scripts_dir = root_dir.joinpath("scripts")

script = "fix_freq_rk3588.sh"
command = f"sudo bash {scripts_dir.joinpath(script)}"
subprocess.run(command, shell=True)
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

filename = (
  # "Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"
  # "Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"
  "Qwen2.5-3B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"
)

# config = ModelConfig(
#   name="qwen3:4B",
#   size=4,
#   filepath=models_dir.joinpath(filename),
#   tokenizer="Qwen/Qwen3-4B",
# )
config = ModelConfig(
  name="qwen2.5:3B",
  size=3,  # TODO: make possible to infer it from HF files or somehow
  filepath=models_dir.joinpath(filename),
  tokenizer="Qwen/Qwen2.5-3B-Instruct",
)
# TODO: now just put it into config yaml files and load automatically

model = RKLLMModel.from_config(config)
callback = partial(print, end="", flush=True)

# TODO: get generation config from HuggingFace as well, and set it correctly in RKNN class
# TODO: make possible to change generation params on-the-fly
response = model.generate([{"role": "user", "content": "Hello, what's up?"}], stream_callback=callback, debug=True)
print(response)
