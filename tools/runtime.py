import enum
import inspect

from collections.abc import Callable
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar, cast
from zoneinfo import ZoneInfo

import punq

from attrs import define
from attrs import field
from mako.template import Template

from tools.core.backend import calendar
from tools.core.backend import weather
from tools.core.backend import web


ISO8061_DATE = "%Y-%m-%d"
ISO8061_TIME = "%H:%M:%S"
ISO8061 = f"{ISO8061_DATE} {ISO8061_TIME}"

MAKO_PREFIX = "$"

T = TypeVar("T")


def _get_current_var_name(name: str) -> str:
  return f"CURRENT_{name.upper()}"


class CURRENT(enum.StrEnum):
  """Simple placeholders for the current dynamic runtime values."""

  @staticmethod
  def _generate_next_value_(name: str, *_) -> str:
    return MAKO_PREFIX + "{%s}" % _get_current_var_name(name)

  TIME: str = enum.auto()
  DATE: str = enum.auto()
  DATETIME: str = enum.auto()
  LOCATION: str = enum.auto()
  LANGUAGE: str = enum.auto()

  @enum.property
  def alias(self) -> str:
    return _get_current_var_name(self.name)


@define
class Runtime:
  timezone: str
  location: str
  language: str

  backends: punq.Container = field(factory=punq.Container, init=False)

  @cached_property
  def tz(self) -> ZoneInfo:
    return ZoneInfo(self.timezone)

  @property
  def date(self) -> str:
    """Returns a current date as a simple separated ISO8601 string <YYYY-MM-DD> (always local timezone)."""

    return datetime.now(self.tz).strftime(ISO8061_DATE)

  @property
  def time(self) -> str:
    """Returns only a current time as an ISO8601 compatible <hours:minutes:seconds> (always local timezone)."""

    return datetime.now(self.tz).strftime(ISO8061_TIME)

  @property
  def datetime(self) -> str:
    """Returns a current datetime as a simple separated ISO8601 string (always local timezone)."""

    return f"{self.date} {self.time}"

  def resolve(self, value: str) -> str:
    env_map = {
      CURRENT.TIME: self.time,
      CURRENT.DATE: self.date,
      CURRENT.DATETIME: self.datetime,
      CURRENT.LOCATION: self.location,
      CURRENT.LANGUAGE: self.language,
    }
    ctx = {e.alias: v for e, v in env_map.items()}
    return Template(value).render(**ctx)

  def get_backend(self, backend: type[T]) -> T:
    tool = self.backends.resolve(backend)
    return cast(T, tool)

  def set_backend(self, backend: type[T], instance: T) -> None:
    self.backends.register(backend, instance=instance)


def call(fn: Callable, rt: Runtime, *args, **kwargs) -> Any:
  """Runs the given tool function, interpolating context variables from the runtime."""

  sig = inspect.signature(fn)
  kws = {}

  args = sig.bind_partial(*args, **kwargs)
  args.apply_defaults()

  for key, val in args.arguments.items():
    # if the value is a string, attempt to render it as a Mako template
    if isinstance(val, str) and val.startswith(MAKO_PREFIX):
      try:
        kws[key] = rt.resolve(val)
      except Exception:
        # TODO: log error
        kws[key] = val
    else:
      # keep non-string values as-is
      kws[key] = val

  return fn(**kws)


def runtime_callable(fn: Callable) -> Callable:
  global runtime

  from functools import wraps

  @wraps(fn)
  def wrapper(*args, **kwargs) -> Any:
    return call(fn, runtime, *args, **kwargs)

  return wrapper


# @@@ Configuration Section

root_dir = Path("./.rt")
root_dir.mkdir(exist_ok=True)

runtime = Runtime(
  timezone="Europe/Moscow",
  location="Moscow",
  language="ru",
)


runtime.set_backend(
  calendar.Calendar,
  instance=calendar.JsonBasedCalendar.from_path(root_dir / "calendar.json"),
)
runtime.set_backend(
  web.Browser,
  instance=web.Browser(
    web=web.DuckDuckGoSearch(),
    wiki=web.WikiChatSearch(language=runtime.language),
  ),
)
runtime.set_backend(
  weather.WeatherProvider,
  instance=weather.OpenMeteoWeatherProvider(language=runtime.language),
)
