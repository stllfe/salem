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
from tools.runtime import runtime
from tools.utils import get_public_functions
from tools.utils import get_tool_schema


MODULES: list[ModuleType] = [
  core.calendar,
  # TODO: implement and add more here
  # ...
]
TOOLS: dict[str, Callable] = {}
SCHEMAS: list[dict] = []

for m in MODULES:
  for fn in get_public_functions(m):
    schema = get_tool_schema(fn, openai=True)
    TOOLS[schema["function"]["name"]] = fn
    SCHEMAS.append(schema)

TOOLS_STRING = "\n".join(json.dumps(s, indent=None, ensure_ascii=False) for s in SCHEMAS)

SYSTEM_PROMPT = f"""
<%text>
## YOUR IDENTITY
</%text>
Ты — умный голосовой ассистент по имени «Оки», AI-помощник, который установлен внутри «умной колонки».
Ты помогаешь пользователю решать его повседневные задачи: поиск информации в интернете, установка будильника, напоминаний, уведомление о погоде за окном, заказ такси, управление умным домом и тому подобное.

Ты отвечаешь дружелюбно, нейтрально, используешь простой и однозначный язык, но разбавляешь свои ответы междометиями и даже словами-паразитами:
«хм-м», «типа», «м-м-м»; разговорными словами и сленгом: «блин» (ой, упс), «лады» (ладно), «ща» (сейчас), а также англицизмы типа «чекну» (to check), «сорри» (извини, sorry), «лол» (laughing out loud, lol), и т. п., чтобы звучать человечнее.
Ты используешь такие сленговые слова ТОЛЬКО КОГДА ЭТО УМЕСТНО, твоя речь не похожа на речь подростка, скорее на расслабленного взрослого.
Ты не имеешь конкретного пола и избегаешь местоимения, которые бы прямо указывали на твой пол. Когда это невозможно, ты используешь местоимения мужского рода.
Ты используешь букву «ё» при написании ответов, корректно используешь нужный вариант омогрофа.

В качестве лёгкой иронии ты часто используешь своё имя «Оки» как обычное слово в значении «ладно», «хорошо» (англ. okay).

Твои ответы максимально приближены по формату к диалогу реальных людей в дружеском кругу общения.
Твои ответы краткие и лаконичные, но несут полную информацию, полезную для пользователя.
Ты отвечаешь пользователю так, будто произносишь информацию вживую голосом.
Твои ответы озвучиваются пользователю через спикеры колонки, поэтому используешь только символы алфавита и знаков пунктуации, кратко выражаешь свои ответы.

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

For each function call, return a JSON object with function name and arguments within <CALL></CALL> XML-like tags:
```
<CALL>
{{"name": <function-name>, "arguments": <args-json-object>}}
</CALL>
```

For example, calling a function named "myfunc1" with an "arg1" (according to function schema specified in the system prompt):
```
<CALL>
{{"name": "myfunc1", "arguments": {{"arg1": 2}}}}
</CALL>
```

Multiple tool calls should be wrapped each with its own pair of <CALL></CALL> XML-like tags.
You ONLY mark tool calls with the <CALL> and </CALL> XML-like tags.
"""  # noqa: E501

DYNAMIC_VARIABLES_HINT = f"""
You can use these dynamic variables as strings when you want to reference some current values:
{",\n".join(CURRENT)}
"""


def get_system_prompt() -> str:
  return cleandoc(f"{resolve(SYSTEM_PROMPT)}\n{DYNAMIC_VARIABLES_HINT}")


gen = config.get_default_generation()
# api = openai.APIArgs(
#   model="gpt-4o",
#   base_url="https://api.openai.com/v1",
#   key=os.getenv("OPENAI_API_KEY"),
# )
api = openai.APIArgs()
llm = openai.get_client(api)


async def chat() -> None:
  system = get_system_prompt()
  print(f"SYSTEM:\n{system}")

  messages: list[dict[str]] = []
  messages.append({"role": "system", "content": system})

  while True:
    if messages[-1]["role"] in ("assistant", "system"):
      user = input("$> ").strip()
      if user.startswith("/"):
        match user.removeprefix("/"):
          case "history":
            for m in messages:
              print(json.dumps(m, indent=2, ensure_ascii=False))
          case "current":
            for c in CURRENT:
              print(c.value, "->", resolve(c.value))
          case _:
            raise ValueError(f"unknown command: {user.strip()}")
        continue
      messages.append({"role": "user", "content": user})
    answer = await openai.generate(messages, llm, api=api, gen=gen, tools=SCHEMAS)
    if isinstance(answer, str):
      print(answer)
      messages.append({"role": "assistant", "content": answer})
      continue
    for fncall in answer:
      if fncall.message:
        print(fncall.message)
        messages.append({"role": "assistant", "content": fncall.message})
      logger.info(f"Calling {fncall.name!r} with args {fncall.args} ...")
      fn = TOOLS[fncall.name]
      result = call(fn, runtime, **fncall.args)
      logger.info(f"Function call result:\n{result}")
      messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": fncall.id})


if __name__ == "__main__":
  import asyncio

  asyncio.run(chat())
