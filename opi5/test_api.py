import os
import resource
import subprocess

from functools import partial
from pathlib import Path


os.environ["RKLLM_LOG_LEVEL"] = "1"  # will print token stats

from api.rkllm import RKLLMModel


root_dir = Path(__file__).parent
models_dir = root_dir.joinpath("models")
scripts_dir = root_dir.joinpath("scripts")

script = "fix_freq_rk3588.sh"
command = f"sudo bash {scripts_dir.joinpath(script)}"
subprocess.run(command, shell=True)
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

model = RKLLMModel("qwen2.5:1.5B")
callback = partial(print, end="", flush=True)

# TODO: make possible to change generation params on-the-fly
# -> looks like not possible since rknn-llm initializes a runtime once with a set of params
response = model.generate([{"role": "user", "content": "Hello, what's up?"}], stream_callback=callback, debug=True)
print(response)
