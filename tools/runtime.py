import enum
import inspect

from collections.abc import Callable
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from attrs import define
from mako.template import Template

from tools.core.backend import CalendarBackend
from tools.core.backend import JsonBasedCalendar


ISO8061_DATE = "%Y-%m-%d"
ISO8061_TIME = "%H:%M:%S"
ISO8061 = f"{ISO8061_DATE} {ISO8061_TIME}"

Callback = Callable[..., None]


@define
class Runtime:
  timezone: str
  location: str
  language: str

  # actual implementations
  # TODO: add support for proper dependency injection, using punc or something similar
  calendar: CalendarBackend
  on_event: Callback | None = None

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


def _get_current_var_name(name: str) -> str:
  return f"CURRENT_{name.upper()}"


class CURRENT(enum.StrEnum):
  """Simple placeholders for the current dynamic runtime values."""

  @staticmethod
  def _generate_next_value_(name: str, *_) -> str:
    return "${%s}" % _get_current_var_name(name)

  TIME: str = enum.auto()
  DATE: str = enum.auto()
  DATETIME: str = enum.auto()
  LOCATION: str = enum.auto()
  LANGUAGE: str = enum.auto()

  @enum.property
  def alias(self) -> str:
    return _get_current_var_name(self.name)


def resolve(val: str) -> str:
  env_map = {
    CURRENT.TIME: rt.time,
    CURRENT.DATE: rt.date,
    CURRENT.DATETIME: rt.datetime,
    CURRENT.LOCATION: rt.location,
    CURRENT.LANGUAGE: rt.language,
  }
  ctx = {e.alias: v for e, v in env_map.items()}
  return Template(val).render(**ctx)


def call(fn: Callable, rt: Runtime, *args, **kwargs) -> Any:
  """Runs the given tool function, interpolating context variables from the runtime."""

  sig = inspect.signature(fn)
  kws = {}

  args = sig.bind_partial(*args, **kwargs)
  args.apply_defaults()

  for key, val in args.arguments.items():
    # if the value is a string, attempt to render it as a Mako template
    if isinstance(val, str):
      try:
        kws[key] = resolve(val)
      except Exception:
        kws[key] = val
    else:
      # keep non-string values as-is
      kws[key] = val

  return fn(**kws)


rt = Runtime(
  timezone="Europe/Moscow",
  location="Moscow",
  language="ru",
  calendar=JsonBasedCalendar.from_path(Path(".rt/calendar.json").expanduser()),
)
