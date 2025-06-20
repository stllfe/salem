import os
import resource
import subprocess
import threading
import time

from pathlib import Path


os.environ["RKLLM_LOG_LEVEL"] = "1"

from api.binding import RKLLM
from api.binding import global_text


root_dir = Path(__file__).parent
models_dir = root_dir.joinpath("models")
scripts_dir = root_dir.joinpath("scripts")

script = "fix_freq_rk3588.sh"
command = f"sudo bash {scripts_dir.joinpath(script)}"
subprocess.run(command, shell=True)
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

model_name = (
  "Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"  # TODO: пока только эта моделька работает корректно
  # "Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"
  # "Qwen2.5-3B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"
)
model = RKLLM(model_path=models_dir.joinpath(model_name).as_posix())

prompt = "Hello! What's up?"
output = ""

try:
  model_thread = threading.Thread(target=model.run, args=(prompt,))
  model_thread.start()
  model_thread_finished = False
  while not model_thread_finished:
    while len(global_text) > 0:
      diff = global_text.pop(0)
      output += diff
      time.sleep(0.005)
      print(diff, end="", flush=True)
    model_thread.join(timeout=0.005)
    model_thread_finished = not model_thread.is_alive()
finally:
  model.release()
  print("\n")
