# noqa: A005

import json

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import orjson

from attrs import asdict
from attrs import define
from attrs import field
from attrs import frozen
from tinydb import JSONStorage
from tinydb import Query
from tinydb import TinyDB


def get_short_uid() -> str:
  uuid = uuid4()
  return uuid.hex[:6]


def convert_datetime(dt: datetime | str) -> datetime:
  if isinstance(dt, str):
    return datetime.fromisoformat(dt)
  return dt


class DateTimeEncoder(json.JSONEncoder):
  def default(self, obj: Any) -> Any:
    import datetime as dt

    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
      return obj.isoformat()
    elif isinstance(obj, dt.timedelta):
      return (dt.datetime.min + obj).time().isoformat()
    return super(DateTimeEncoder, self).default(obj)


class JsonMixin:
  def dump(self) -> dict:
    return asdict(self)

  def json(self) -> bytes:
    return orjson.dumps(self.dump(), option=orjson.OPT_OMIT_MICROSECONDS)

  @classmethod
  def load(cls, data: dict | bytes) -> Self:
    if isinstance(data, bytes):
      data = orjson.loads(data)
    return cls(**data)


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


class CalendarBackend(ABC):
  @abstractmethod
  def add_event(self, event: Event) -> str:
    """Add a new event."""

  @abstractmethod
  def get_event(self, uid: str) -> Event:
    """Get an event by uid."""

  @abstractmethod
  def remove_event(self, uid: str) -> None:
    """Remove an event by uid."""

  @abstractmethod
  def edit_event(self, uid: str, *, date: datetime | None = None, comment: str | None = None) -> None:
    """Edit an existing event."""

  @abstractmethod
  def get_all_events(self, start: datetime, end: datetime, regex: str | None = None) -> list[Event]:
    """Get all events in the date range."""

  @abstractmethod
  def add_reminder(self, reminder: Reminder) -> str:
    """Add a new reminder."""

  @abstractmethod
  def remove_reminder(self, uid: str) -> None:
    """Remove a reminder by uid."""

  @abstractmethod
  def get_reminder(self, uid: str) -> Reminder:
    """Get a reminder by uid."""

  @abstractmethod
  def get_all_reminders(self, date: datetime) -> list[Reminder]:
    """Get all reminders for the date."""


@define
class JsonBasedCalendar(CalendarBackend):
  db: TinyDB

  def __attrs_post_init__(self) -> None:
    self._events = self.db.table("events")
    self._reminders = self.db.table("reminders")

  @classmethod
  def from_path(cls, path: str | Path) -> Self:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return cls(TinyDB(storage=JSONStorage, path=path.as_posix(), encoding="utf-8", cls=DateTimeEncoder))

  def add_event(self, event: Event) -> str:
    self._events.insert(event.to_dict())
    return event.uid

  def get_event(self, uid: str) -> Event:
    Q = Query()
    if doc := self._events.get(Q.uid == uid):
      return Event.load(doc)
    raise KeyError(f"Event {uid} not found")

  def remove_event(self, uid: str) -> None:
    Q = Query()
    if not self._events.remove(Q.uid == uid):
      raise KeyError(f"Event {uid} not found")

  def edit_event(self, uid: str, *, date: datetime | None = None, comment: str | None = None) -> None:
    Q = Query()
    updates = {}
    if date is not None:
      updates["date"] = date.isoformat()
    if comment is not None:
      updates["comment"] = comment
    if not updates:
      return
    if not self._events.update(updates, Q.uid == uid):
      raise KeyError(f"Event {uid} not found")

  def get_all_events(self, start: datetime, end: datetime, regex: str | None = None) -> list[Event]:
    Q = Query()
    start = start.isoformat()
    end = end.isoformat()

    # first filter by date range
    docs = self._events.search((Q.date >= start) & (Q.date <= end))

    # then optionally filter by pattern
    if regex is not None:
      import re

      regex = re.compile(regex)
      docs = [
        doc
        for doc in docs
        if regex.search(doc["name"]) or (doc["comment"] is not None and regex.search(doc["comment"]))
      ]

    return [Event.load(doc) for doc in docs]

  def add_reminder(self, reminder: Reminder) -> str:
    self._reminders.insert(reminder.dump())
    return reminder.uid

  def remove_reminder(self, uid: str) -> None:
    Q = Query()
    if not self._reminders.remove(Q.uid == uid):
      raise KeyError(f"Reminder {uid} not found")

  def get_reminder(self, uid: str) -> Reminder:
    Q = Query()
    if doc := self._reminders.get(Q.uid == uid):
      return Reminder.load(doc)
    raise KeyError(f"Reminder {uid} not found")

  def get_all_reminders(self, date: datetime) -> list[Reminder]:
    Q = Query()
    date, *_ = date.isoformat("T").partition("T")  # get just the date part
    docs = self._reminders.search(Q.date.test(lambda d: d.startswith(date)))
    return [Reminder.load(doc) for doc in docs]
