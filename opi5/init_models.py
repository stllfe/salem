from pathlib import Path

from api.utils import CONFIG_PATH
from api.utils import init_models
from api.utils import load_model_registry


def main(config_path: Path = CONFIG_PATH, force_download: bool = False) -> None:
  """Downloads and validates all RKLLM models from the given config."""

  config = load_model_registry(config_path)
  init_models(config, force_download=force_download)


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
