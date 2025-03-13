# noqa: A005
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

import orjson

from attrs import asdict
from attrs import field
from attrs import frozen

from src.utils import get_short_uid


Language = Literal["ru", "en"]  # only these are supported
TimeUnit = Literal["hour", "minute", "second", "day", "week", "month", "year"]


class TempUnit(StrEnum):
  C = "celsius"
  F = "fahrenheit"


class JsonMixin:
  def dump(self) -> dict:
    return asdict(self)

  def json(self) -> bytes:
    return orjson.dumps(self.dump(), option=orjson.OPT_OMIT_MICROSECONDS)

  @classmethod
  def load(cls, data: dict | bytes, strict=True) -> Self:
    if isinstance(data, bytes):
      data = orjson.loads(data)
      assert isinstance(data, dict), f"Unknown raw format: {type(dict)}"
    if strict:
      slots = set(cls.__slots__)
      data = {k: v for k, v in data.items() if k in slots}
    return cls(**data)


def convert_datetime(dt: datetime | str) -> datetime:
  if isinstance(dt, str):
    return datetime.fromisoformat(dt)
  return dt


@frozen
class Event(JsonMixin):
  name: str
  date: datetime = field(converter=convert_datetime)
  comment: str | None = None

  uid: str = field(factory=get_short_uid)
  timezone: str | None = None


@frozen
class Reminder(JsonMixin):
  message: str
  date: datetime = field(converter=convert_datetime)

  uid: str = field(factory=get_short_uid)
  timezone: str | None = None


@frozen
class WebLink(JsonMixin):
  url: str
  title: str | None = None
  caption: str | None = None
  uid: str = field(factory=get_short_uid)


@frozen
class WikiExtract(JsonMixin):
  url: str
  title: str
  content: str
  language: Language
  updated_at: datetime = field(converter=convert_datetime)
  section: str | None = None
  uid: str = field(factory=get_short_uid)


@frozen
class Weather(JsonMixin):
  location: str
  temperature: float

  feels_like: float
  """How actually a temperature feels like"""

  humidity: float
  """Humidity in %"""

  pressure: float
  """Pressure in mmHg"""

  wind_speed: float
  """Wind speed in ms (metres per second)"""

  date: datetime = field(converter=convert_datetime)

  units: TempUnit = field(default=TempUnit.C, converter=TempUnit)
  """Temperature units (celsius or fahrenheit)"""


@frozen
class WeatherForecast(JsonMixin):
  location: str
  forecast: list[Weather]

  @property
  def days(self) -> int:
    return len(self.forecast)

  def min(self, attr: str) -> float:
    pass

  def max(self, attr: str) -> float:
    pass

  def avg(self, attr: str) -> float:
    pass
