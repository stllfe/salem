import os
import resource
import subprocess

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

model = RKLLMModel("qwen2.5:3B", prompt_cache_path="./rkllm_cache")

# TODO: make possible to change generation params on-the-fly
# NOTE: looks like not possible since rknn-llm initializes a runtime once with a set of params
messages = []
try:
  while True:
    message = input("$> ")
    if not message or message == "/exit":
      break
    messages.append({"role": "user", "content": message})
    reply = ""
    for token in model.generate_stream(messages, debug=True):
      print(token, end="", flush=True)
      reply += token
    messages.append({"role": "assistant", "content": reply})
except KeyboardInterrupt:
  pass
finally:
  print("\nGracefull shutdown...")
  model.release()
