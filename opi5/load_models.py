from pathlib import Path

from api.models import CONFIG_PATH
from api.models import init_models
from api.models import load_model_registry


def main(config_path: Path = CONFIG_PATH, force_download: bool = False) -> None:
  """Downloads and validates all RKLLM models from the given config."""

  config = load_model_registry(config_path)
  init_models(config, force_download=force_download)


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
