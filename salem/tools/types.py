# noqa: A005
import operator as op

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from itertools import chain
from typing import Generic, Iterator, Literal, NamedTuple, Self, TypeVar
from zoneinfo import ZoneInfo

import orjson

from attrs import asdict
from attrs import field
from attrs import frozen

from salem.utils import get_short_uid


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


def convert_datetime(dt: datetime | str | int) -> datetime:
  if isinstance(dt, int):
    return datetime.fromtimestamp(dt)
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
class LocationInfo(JsonMixin):
  name: str
  country: str
  latitude: float
  longitude: float
  timezone: str

  @property
  def tz(self) -> ZoneInfo:
    return ZoneInfo(self.timezone)

  def __str__(self) -> str:
    return f"{self.name}, {self.country} ({self.timezone})".title()


@frozen
class Weather(JsonMixin):
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


class DayWeather(NamedTuple):
  morning: Weather
  daytime: Weather
  evening: Weather


T = TypeVar("T")


@frozen
class Aggregate(Generic[T]):
  values: Sequence[T] = field(converter=list)

  def min(self) -> T:
    return min(self.values)

  def max(self) -> T:
    return max(self.values)

  def avg(self) -> T:
    return sum(self.values) / len(self.values)


@frozen
class WeatherForecast(JsonMixin):
  location: LocationInfo
  daily: list[DayWeather] = field(repr=lambda _: "[...]")

  def __iter__(self) -> Iterator[Weather]:
    return chain.from_iterable(map(iter, self.daily))

  def __len__(self) -> int:
    return len(self.daily)

  @property
  def num_days(self) -> int:
    dt = self.end_date - self.start_date
    return dt.days + 1  # including current

  @property
  def start_date(self) -> datetime:
    return self.daily[0].morning.date

  @property
  def end_date(self) -> datetime:
    return self.daily[-1].evening.date

  def get(self, prop: str, only: Literal["morning", "daytime", "evening"] | None = None) -> Aggregate[float]:
    if only:
      points = map(op.attrgetter(only), self.daily)
    else:
      points = iter(self)
    points = list(points)
    values = list(map(op.attrgetter(prop), points))
    types = set(map(type, values))
    assert len(types) == 1 and types.pop() is float, f"Expected prop argument to be a float value, got: {types[0]}"
    return Aggregate(values)
