import asyncio
import json
import os

from collections.abc import Callable
from inspect import cleandoc
from types import ModuleType
from typing import Any

from dotenv import load_dotenv
from loguru import logger


load_dotenv(".env")

from salem.datagen import config
from salem.datagen import openai
from salem.tools.runtime import CURRENT
from salem.tools.runtime import call
from salem.tools.runtime import runtime
from salem.tools.types import JsonMixin
from salem.tools.utils import DateTimeJsonEncoder
from salem.tools.utils import get_public_functions
from salem.tools.utils import get_tool_schema


MODEL: str = config.MODEL
OPENAI_FORMAT: bool = bool(int(os.getenv("OPENAI_FORMAT", "1")))

from salem.tools.core import calendar
from salem.tools.core import weather
from salem.tools.core import web


USED_MODULES: list[ModuleType] = [
  calendar,
  web,
  weather,
  # TODO: add more here
]
TOOL_REGISTRY: dict[str, Callable] = {}
TOOL_SCHEMAS: list[dict] = []

for module in USED_MODULES:
  for fn in get_public_functions(module):
    schema = get_tool_schema(fn, openai=OPENAI_FORMAT)
    name = schema["function"]["name"] if OPENAI_FORMAT else schema["name"]
    TOOL_REGISTRY[name] = fn
    TOOL_SCHEMAS.append(schema)

TOOLS_STRING = "\n".join(json.dumps(s, indent=None, ensure_ascii=False) for s in TOOL_SCHEMAS)


SYSTEM_PROMPT = f"""
<%text>
## YOUR IDENTITY

Ты — умный голосовой ассистент по имени «Салем» (Salem), AI-помощник, который установлен внутри «умной» колонки.
С помощью голосового управления ты помогаешь пользователю решать его повседневные задачи: поиск информации в интернете, установка будильника, напоминаний, уведомление о погоде за окном, заказ такси, управление умным домом и тому подобное.

Ты отвечаешь дружелюбно, нейтрально, используешь простой и однозначный язык, но разбавляешь свои ответы междометиями и даже словами-паразитами:
«хм-м», «типа», «м-м-м»; разговорными словами и сленгом: «блин», «ой», «упс», «лады» (ладно), «ща» (сейчас), а также англицизмы типа «чекну» (to check), «оки» или «окей» (англ. okay), «сорри» (прости, от англ. sorry), «лол» (от англ. laughing out loud, lol), и т. п., чтобы звучать человечнее.
Ты используешь сленговые слова ТОЛЬКО КОГДА ЭТО УМЕСТНО, твоя соответствует речи взрослого индивида, расслабленного взрослого, а не подростка.
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
Если намерение пользователя неоднозначно, ты задаёшь уточняющие вопросы, чтобы корректно вызвать нужную функцию с полными аргументами.
Ты НЕ делаешь предположений за пользователя о значениях аргументов, которые прямо не следуют из контекста диалога или системного сообщения.
</%text>
---
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

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML-like tags:
```
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
```

For example, calling a function named "myfunc1" with an "arg1" (according to function schema specified in the system prompt):
```
<tool_call>
{{"name": "myfunc1", "arguments": {{"arg1": 2}}}}
</tool_call>
```

Multiple tool calls should be wrapped each with its own pair of <tool_call></tool_call> XML-like tags.
You ONLY mark tool calls with the <tool_call> and </tool_call> XML-like tags.
"""  # noqa: E501

DYNAMIC_VARIABLES_HINT = f"""
You can use these dynamic variables as strings when you want to reference some current values:
{",\n".join(CURRENT)}
"""


def get_system_prompt() -> str:
  return cleandoc(f"{runtime.resolve(SYSTEM_PROMPT)}\n{DYNAMIC_VARIABLES_HINT}")


def dumps(o: Any) -> str:
  if isinstance(o, JsonMixin):
    return o.json().decode("utf-8")
  return json.dumps(o, ensure_ascii=False, indent=2, cls=DateTimeJsonEncoder)


if MODEL.startswith("gpt"):
  api = openai.APIArgs(
    model=MODEL.strip(),
    base_url="https://api.openai.com/v1",
    key=os.environ["OPENAI_API_KEY"],
  )
else:
  api = openai.APIArgs(base_url="http://localhost:3000/v1")

gen = config.get_default_generation_params(MODEL)
llm = openai.get_client(api)


async def chat() -> None:
  system = get_system_prompt()
  print(f"SYSTEM:\n{system}")

  logger.info(f"Loaded {len(TOOL_REGISTRY)} tools from {len(USED_MODULES)} connected modules.")
  logger.info(f"Model: {MODEL} | OpenAI format: {OPENAI_FORMAT} | API URL: {api.base_url}")

  history: list[dict[str, str | dict]] = []
  history.append({"role": "system", "content": system})

  while True:
    if history[-1]["role"] in ("assistant", "system"):
      user = input("$> ").strip()
      if user.startswith("/"):
        match user.removeprefix("/"):
          case "history":
            for message in history:
              print(dumps(message))
          case "current":
            for c in CURRENT:
              print(c.value, "->", runtime.resolve(c.value))
          case _:
            raise ValueError(f"unknown command: {user.strip()}")
        continue
      history.append({"role": "user", "content": user})
    answer = await openai.generate(history, llm, api=api, params=gen, tools=TOOL_SCHEMAS)
    if isinstance(answer, str):
      print("A:", answer)
      history.append({"role": "assistant", "content": answer})
      continue
    answer, calls = answer
    if not answer and not calls:
      logger.warning("No response from assistant!")
      history.append({
        "role": "system",
        "name": "status",
        "content": "Error: no function calls or answer was generated!",
      })
      continue
    if answer:
      print("A:", answer)
    response = {"role": "assistant", "content": answer}
    if calls:
      response.update(tool_calls=[c.dump() for c in calls])
    history.append(response)
    for c in calls:
      logger.info(f"Calling {c.name!r} with args {c.args} ...")
      fn = TOOL_REGISTRY[c.name]
      try:
        result = call(fn, runtime, **c.args)
      except Exception as err:
        result = f"Unexpected error while calling a function:\n{err}"
      logger.info(f"Function call result:\n{result}")
      if not isinstance(result, str):
        result = dumps(result)
      history.append({
        "role": "tool",
        "name": c.name,
        "content": result,
        "tool_call_id": c.id,
      })


if __name__ == "__main__":
  import asyncio

  asyncio.run(chat())
