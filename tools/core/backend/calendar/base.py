# noqa: A005
from abc import ABC
from abc import abstractmethod
from datetime import datetime

from tools.types import Event
from tools.types import Reminder


class Calendar(ABC):
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
