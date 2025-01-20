# noqa: A005

from datetime import datetime
from pathlib import Path
from typing import Self

from attrs import define
from tinydb import JSONStorage
from tinydb import Query
from tinydb import TinyDB

from tools.core.backend.calendar.base import Calendar
from tools.types import Event
from tools.types import Reminder
from tools.utils import DateTimeJsonEncoder


@define
class JsonBasedCalendar(Calendar):
  db: TinyDB

  def __attrs_post_init__(self) -> None:
    self._events = self.db.table("events")
    self._reminders = self.db.table("reminders")

  @classmethod
  def from_path(cls, path: str | Path) -> Self:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    db = TinyDB(storage=JSONStorage, path=path.as_posix(), encoding="utf-8", cls=DateTimeJsonEncoder)
    return cls(db)

  def add_event(self, event: Event) -> str:
    self._events.insert(event.dump())
    return event.uid

  def get_event(self, uid: str) -> Event:
    q = Query()
    if doc := self._events.get(q.uid == uid):
      return Event.load(doc)
    raise KeyError(f"Event {uid} not found")

  def remove_event(self, uid: str) -> None:
    q = Query()
    if not self._events.remove(q.uid == uid):
      raise KeyError(f"Event {uid} not found")

  def edit_event(self, uid: str, *, date: datetime | None = None, comment: str | None = None) -> None:
    q = Query()
    updates = {}
    if date is not None:
      updates["date"] = date.isoformat()
    if comment is not None:
      updates["comment"] = comment
    if not updates:
      return
    if not self._events.update(updates, q.uid == uid):
      raise KeyError(f"Event {uid} not found")

  def get_all_events(self, start: datetime, end: datetime, regex: str | None = None) -> list[Event]:
    q = Query()
    start = start.isoformat()
    end = end.isoformat()

    # first filter by date range
    docs = self._events.search((q.date >= start) & (q.date <= end))

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
    q = Query()
    if not self._reminders.remove(q.uid == uid):
      raise KeyError(f"Reminder {uid} not found")

  def get_reminder(self, uid: str) -> Reminder:
    q = Query()
    if doc := self._reminders.get(q.uid == uid):
      return Reminder.load(doc)
    raise KeyError(f"Reminder {uid} not found")

  def get_all_reminders(self, date: datetime) -> list[Reminder]:
    q = Query()
    date, *_ = date.isoformat("T").partition("T")  # get just the date part
    docs = self._reminders.search(q.date.test(lambda d: d.startswith(date)))
    return [Reminder.load(doc) for doc in docs]
