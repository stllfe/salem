import os

from functools import cache
from pathlib import Path

from loguru import logger
from msgspec import field

from api.utils import Serializable


PROJECT_DIR = Path(__file__).parent.parent.parent
OPI5_DIR = PROJECT_DIR.joinpath("opi5")

# set the default RKLLM_MODELS_DIR relative to opi5 subfolder root by default
os.environ["RKLLM_MODELS_DIR"] = os.getenv("RKLLM_MODELS_DIR", str(OPI5_DIR.joinpath("models")))

CONFIG_PATH = OPI5_DIR.joinpath("config.yaml")

MODEL_FILES = (
  "config.json",
  "README.md",
  "generation_config.json",
)
TOKENIZER_FILES = (
  "tokenizer_config.json",
  "tokenizer.json",
  "vocab.json",
  "merges.txt",
)


class ModelInfo(Serializable):
  family: str
  size: str
  hf_path: str
  filename: str

  @property
  def slag(self) -> str:
    return f"{self.family}:{self.size}"


class ModelRegistry(Serializable):
  models_dir: Path
  cache_dir: Path
  models: list[ModelInfo]

  # helper index for quick model retrieval
  _index: dict[str, ModelInfo] = field(default_factory=dict)

  def __post_init__(self) -> None:
    for m in self.models:
      self._index[m.slag] = m
      self._index[m.filename] = m

  def get_model_dir(self, m: ModelInfo) -> Path:
    return self.models_dir / m.family / m.size

  def get_model(self, model_name: str) -> ModelInfo:
    return self._index[model_name]


@cache
def load_model_registry(config_path: str | Path = CONFIG_PATH) -> ModelRegistry:
  config_path = Path(config_path)
  assert config_path.exists(), f"Registry config file not found: {config_path}"
  return ModelRegistry.from_yaml(config_path)


def init_models(registry: ModelRegistry, force_download: bool = False) -> None:
  """Loads all necessary model files from the given model registry."""

  from tempfile import TemporaryDirectory

  from huggingface_hub import get_paths_info
  from huggingface_hub import hf_hub_download
  from transformers import AutoTokenizer

  models_dir = registry.models_dir
  models_dir.mkdir(parents=True, exist_ok=True)

  cache_dir = registry.cache_dir
  cache_dir.mkdir(parents=True, exist_ok=True)

  def init_model(m: ModelInfo) -> None:
    model_dir = registry.get_model_dir(m)
    model_dir.mkdir(parents=True, exist_ok=True)
    # get the main model file
    # FIXME: cache_dir doesn't behave as expected anyways
    hf_hub_download(m.hf_path, m.filename, cache_dir=cache_dir, local_dir=model_dir, force_download=force_download)
    # get tokenizer files if not present
    try:
      AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    except (EnvironmentError, ValueError):  # partial downloads or non-existen files
      with TemporaryDirectory() as tmp:
        # we omit cache_dir here since tokenizer files are typically lightweight
        tokenizer = AutoTokenizer.from_pretrained(m.hf_path, cache_dir=tmp, force_download=True)
        tokenizer.save_pretrained(model_dir)
    # get additional model files if they exist on the remote url
    for f in get_paths_info(m.hf_path, list(MODEL_FILES)):
      hf_hub_download(m.hf_path, f.path, cache_dir=cache_dir, local_dir=model_dir, force_download=force_download)

  for model in registry.models:
    try:
      init_model(model)
    except Exception:
      logger.exception(f"Can't initialize the model: {model.slag}")


model_registry = load_model_registry()
