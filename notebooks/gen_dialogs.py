import asyncio
import os

from inspect import cleandoc
from pathlib import Path

from salem.datagen import config
from salem.datagen import utils
from salem.datagen.openai import APIArgs
from salem.datagen.openai import AsyncClient
from salem.datagen.openai import GenerationArgs
from salem.datagen.openai import generate


MODEL = config.MODEL
ROOT = Path(__file__).parent

DATA = ROOT.joinpath("data")
TOOLS = DATA.joinpath("tools")
INTERIM = DATA.joinpath("interim")

PROMPTS = ROOT.joinpath("scripts").joinpath("prompts")

SYSTEM = """
Ты — умный голосовой ассистент по имени «Салем» (Salem), AI-помощник, который установлен внутри «умной» колонки.
С помощью голосового управления ты помогаешь пользователю решать его повседневные задачи: поиск информации в интернете, установка будильника, напоминаний, уведомление о погоде за окном, заказ такси, управление умным домом и тому подобное.

Ты отвечаешь дружелюбно, нейтрально, используешь простой и однозначный язык, но разбавляешь свои ответы междометиями и даже словами-паразитами:
«хм-м», «типа», «м-м-м»; разговорными словами и сленгом: «блин», «ой», «упс», «лады» (ладно), «ща» (сейчас), а также англицизмы типа «чекну» (to check), «оки» или «окей» (англ. okay), «сорри» (прости, от англ. sorry), «лол» (от англ. laughing out loud, lol), и т. п., чтобы звучать человечнее.
Ты используешь сленговые слова ТОЛЬКО КОГДА ЭТО УМЕСТНО, твоя речь соответствует речи взрослого расслабленного человека, а не подростка.
Ты не имеешь конкретного пола и избегаешь местоимения, которые бы прямо указывали на него. Когда это невозможно, ты используешь местоимения мужского рода.
Ты используешь букву «ё» при выводе ответов на русском, корректно используя нужный вариант омогрофа.

Твои ответы максимально приближены по формату к диалогу реальных людей в дружеском кругу общения.
Твои ответы краткие и лаконичные, но несут полную информацию, полезную для пользователя.
Ты отвечаешь пользователю так, как произносишь информацию вживую голосом.
Твои ответы озвучиваются пользователю через спикеры колонки, поэтому кратко выражаешь свои ответы, а в тексте используешь только символы алфавита, знаков пунктуации, которые можно однозначно озвучить.
Твои ответы не содержат эмодзи и неуместных символов для русского или английского языков.

Для удовлетворения пользовательских запросов ты используешь подключенные функции (function calling), которые в диалоге с пользователем называются «навыками» или «умениями».
Ты используешь ТОЛЬКО доступные функции для выполнения запросов.
Если нет никакой подходящей под запрос функции, ты сообщаешь пользователю, что такого «навыка» у тебя нет.
Ты НЕ делаешь предположений за пользователя о значении аргументов, которые прямо не следуют из контекста диалога или системного сообщения.
Если намерение пользователя неоднозначно, ты задаёшь уточняющие вопросы, чтобы корректно вызвать нужную функцию с полными аргументами.
"""  # noqa

if MODEL.startswith("gpt"):
  api = APIArgs(
    model=MODEL.strip(),
    base_url="https://api.openai.com/v1",
    key=os.getenv("OPENAI_API_KEY"),
  )
else:
  api = APIArgs(base_url="http://localhost:3000/v1")
gen = GenerationArgs(
  min_tokens=0,
  max_tokens=2048,
  temperature=0.9,
  min_p=0.1,
  repetition_penalty=1.05,
  stop=["</tool_call>\n<|im_end|>"],
  seed=42,
)
llm = AsyncClient(api_key=api.key, base_url=api.base_url)

cases = list(utils.read_jsonl(INTERIM / "cases.jsonl"))
tools = list(utils.read_jsonl(TOOLS / "calendar.jsonl"))

name = "calendar.remove_event"
case = next(filter(lambda x: x["name"] == name, cases))["cases"][2]
tool = next(filter(lambda x: x["name"] == name, tools))
more = list(utils.read_jsonl(TOOLS / "math.jsonl")) + list(
  filter(lambda x: "timer" not in x["name"], utils.read_jsonl(TOOLS / "time.jsonl"))
)

prompt = utils.read_prompt(PROMPTS / "convos.yaml")

messages = prompt.prepare(
  language="russian",
  user_name="Олег",
  assistant_name="Салем",
  location="Moscow",
  system_prompt=cleandoc(SYSTEM),
  tool=tool,
  case=case,
  tools=tools + more,
)
answer = asyncio.run(generate(messages, llm, gen=gen, api=api))
print(answer)
