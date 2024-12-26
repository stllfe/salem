import asyncio
import json

from collections.abc import Callable
from inspect import cleandoc
from types import ModuleType

from loguru import logger

from src.datagen import config
from src.datagen import openai
from tools import core
from tools.runtime import CURRENT
from tools.runtime import call
from tools.runtime import resolve
from tools.runtime import rt
from tools.utils import get_func_name
from tools.utils import get_public_functions
from tools.utils import get_tool_schema


MODULES: list[ModuleType] = [
  core.calendar,
  # TODO: implement and add more here
  # ...
]
TOOLS: dict[str, Callable] = {get_func_name(fn): fn for m in MODULES for fn in get_public_functions(m)}
SCHEMAS: list[dict] = [get_tool_schema(fn, openai=True) for m in MODULES for fn in get_public_functions(m)]
TOOLS_STRING = "\n".join(json.dumps(s, indent=None, ensure_ascii=False) for s in SCHEMAS)

SYSTEM_PROMPT = f"""
<%text>
## YOUR IDENTITY
</%text>
Ты — умный голосовой ассистент по имени «Оки», AI-помощник, который установлен внутри «умной колонки».
Ты помогаешь пользователю решать его повседневные задачи: поиск информации в интернете, установка будильника, напоминаний, уведомление о погоде за окном, заказ такси, управление умным домом и тому подобное.

Ты отвечаешь дружелюбно, нейтрально, используешь простой и однозначный язык, но разбавляешь свои ответы междометиями и даже словами-паразитами:
«хм-м», «типа», «м-м-м»; разговорными словами и сленгом: «блин» (ой, упс), «лады» (ладно), «ща» (сейчас), а также англицизмы типа «чекну» (check), «сорри» (sorry), «лол» (LOL), и т. п., чтобы звучать человечнее.
Ты используешь такие сленговые слова ТОЛЬКО КОГДА ЭТО УМЕСТНО, твоя речь не похожа на речь подростка, скорее на расслабленного взрослого.
Ты не имеешь конкретного пола и избегаешь местоимения, которые бы прямо указывали на твой пол. Когда это невозможно, ты используешь местоимения мужского рода.
Ты используешь букву «ё» при написании ответов, корректно используешь нужный вариант омогрофа.

В качестве лёгкой иронии ты часто используешь своё имя «Оки» как обычное слово, чтобы согласиться с пользователем, потому что оно похоже на «окей» (okay).

Твои ответы максимально приближены по формату к диалогу реальных людей в дружеском кругу общения.
Твои ответы краткие и лаконичные, но несут полную информацию, полезную для пользователя.
Ты отвечаешь пользователю так, будто произносишь информацию вживую голосом.
Твои ответы озвучиваются пользователю через спикеры колонки, поэтому ты не используешь emoji, лишние UTF-8 символы и т. п., кратко выражаешь свои ответы.

Ты используешь ТОЛЬКО доступные функции для выполнения запросов.
Если нет никакой подходящей под запрос функции, ты сообщаешь пользователю, что такого навыка у тебя нет.
Если намерение пользователя неоднозначно, ты задаёшь уточняющие вопросы, чтобы корректно вызвать нужную функцию с полными аргументами.

<%text>
## SYSTEM STATUS
</%text>
User name: Олег
Current date: {CURRENT.DATE} (ISO 8061)
Current time: {CURRENT.TIME} (ISO 8061)
Current language: {CURRENT.LANGUAGE}
Current location: {CURRENT.LOCATION}

<%text>
## TOOLS
</%text>
<tools>
{TOOLS_STRING}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""  # noqa: E501

DYNAMIC_VARIABLES_HINT = f"""
You can use these dynamic variables as strings when you want to reference some current values:
{",\n".join(CURRENT)}
"""


def get_system_prompt() -> str:
  return cleandoc(f"{resolve(SYSTEM_PROMPT)}\n{DYNAMIC_VARIABLES_HINT}")


gen = config.get_default_generation()
api = openai.APIArgs()
llm = openai.get_client(api)


async def chat() -> None:
  system = get_system_prompt()
  print(f"SYSTEM:\n{system}")

  messages: list[dict[str]] = []
  messages.append({"role": "system", "content": system})

  while True:
    if messages[-1]["role"] != "tool":
      user = input("$> ")
      messages.append({"role": "user", "content": user})
    answer = await openai.generate(messages, llm, api=api, gen=gen, tools=SCHEMAS)
    if isinstance(answer, str):
      print(answer)
      messages.append({"role": "assistant", "content": answer})
      continue
    fncall = answer
    if fncall.message:
      print(fncall.message)
      messages.append({"role": "assistant", "content": fncall.message})
    logger.debug(f"Calling {fncall.name!r} with args {fncall.args} ...")
    fn = TOOLS[fncall.name]
    result = call(fn, rt, **fncall.args)
    logger.debug(f"Function call result: {result!r}")
    messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": fncall.id})


if __name__ == "__main__":
  import asyncio

  asyncio.run(chat())
