"""Runs batch generation with an OpenAI compatible API endpoint."""

import asyncio
import enum

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import orjson

from aiopath import AsyncPath
from attrs import define
from attrs import field
from attrs import validators as V
from pydantic import BaseModel
from pydantic import TypeAdapter

from src.datagen import GenerationArgs
from src.datagen import Language
from src.datagen import utils
from src.datagen.config import get_default_generation
from src.datagen.openai import APIArgs
from src.datagen.openai import generate
from src.datagen.openai import get_client
from src.utils import get_logger


ROOT = Path(__file__).parent.parent

DATA = ROOT.joinpath("data")
TOOLS = DATA.joinpath("tools")
INTERIM = DATA.joinpath("interim")

PROMPTS = Path(__file__).parent.joinpath("prompts")

logger = get_logger()


@define
class BaseArgs:
  prompt: Path = field()
  """a path to a prompt for generation in YAML format."""

  dest: Path = field()
  """a path to save the resulting JSONL generations."""

  njobs: int = field(default=8)
  """how many requests to run in parallel."""

  api: APIArgs = field(factory=APIArgs)
  gen: GenerationArgs = field(factory=get_default_generation)


@define
class CasesArgs(BaseArgs):
  """Generate cases for a tool use."""

  num_cases: int = field(default=10, validator=V.gt(1))
  """how many cases to generate per tool function."""

  language: Language = "english"

  tools: Path = TOOLS
  prompt: Path = PROMPTS / "cases.yaml"
  dest: Path = INTERIM / "cases.jsonl"


@define
class ConvosArgs(BaseArgs):
  """Generate conversations based on the synthetic cases."""

  prompt: Path = PROMPTS / "convos.yaml"

  language: Language = "russian"

  tools: Path = TOOLS
  prompt: Path = PROMPTS / "convos.yaml"
  dest: Path = INTERIM / "convos.jsonl"


async def append_jsonl(r: dict[str, Any], f: Path, *, lock: asyncio.Lock | None = None) -> None:
  json = orjson.dumps(r).decode()
  lock = lock or nullcontext()
  async with lock:
    async with AsyncPath(f).open(mode="a+", encoding="utf-8") as fd:
      await fd.write(json + "\n")


class Communication(enum.StrEnum):
  CASUAL = enum.auto()
  FORMAL = enum.auto()
  UNCLEAR = enum.auto()


class Interaction(enum.StrEnum):
  SINGLE_HOP = enum.auto()
  MULTI_HOP = enum.auto()


class Case(BaseModel):
  context: str
  communication: Communication
  interaction: Interaction
  challenges: str


Cases = TypeAdapter(list[Case])


def generate_cases(args: CasesArgs) -> None:
  """Generate cases for tool use."""

  from tqdm.asyncio import tqdm

  llm = get_client(args.api)

  njobs = asyncio.BoundedSemaphore(args.njobs)
  lock = asyncio.Lock()

  args.dest.parent.mkdir(exist_ok=True, parents=True)
  logger.info(f"Saving results to {args.dest}")

  schema = Cases.json_schema()
  logger.info(f"Requested output schema: {Cases!r}")

  tools: list[dict] = []
  for p in args.tools.glob("*.jsonl"):
    logger.info(f"Adding '{p.stem}' tools ...")
    tools.extend(utils.read_jsonl(p))

  prompt = utils.read_prompt(args.prompt)
  names = [tool["name"] for tool in tools]

  messages = prompt.prepare(**tools[0], tools=names, num_cases=args.num_cases, language=args.language)
  logger.info(f"Current system prompt: {prompt.system}")
  logger.info(f"Example user message:\n{messages[-1]['content']}")

  async def job(tool: dict) -> None:
    name = tool["name"]
    messages = prompt.prepare(**tool, tools=names, num_cases=args.num_cases, language=args.language)
    async with njobs:
      try:
        result = await generate(messages, llm, gen=args.gen, api=args.api, json_schema=schema)
        parsed = list(map(Case.model_dump, Cases.validate_json(result)))
        await append_jsonl({"name": name, "cases": parsed}, args.dest, lock=lock)
      except Exception as err:
        logger.error(f"Error while processing {name}: {err}")

  werk = tqdm.gather(*map(job, tools), desc="generating cases", unit="tool")
  asyncio.run(werk)
  logger.info("All done!")


def generate_convos(args: ConvosArgs) -> None:
  # TODO: implement this one
  pass


if __name__ == "__main__":
  import tyro

  args = tyro.extras.subcommand_cli_from_dict({"cases": CasesArgs, "convos": ConvosArgs})
  if isinstance(args, CasesArgs):
    generate_cases(args)
  else:
    generate_convos(args)
