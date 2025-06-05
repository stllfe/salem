# noqa: A005
from datetime import datetime

from salem.tools.core.backend.calendar import Calendar
from salem.tools.runtime import CURRENT
from salem.tools.runtime import ISO8061
from salem.tools.runtime import runtime
from salem.tools.runtime import runtime_callable
from salem.tools.types import Event
from salem.tools.types import Reminder


calendar = runtime.get_backend(Calendar)


def _format_event(e: Event) -> str:
  return f"Event [{e.uid}] @ {e.date.strftime(ISO8061)} - {e.name}\n\tComment: {e.comment or 'N/A'}"


def _format_reminder(r: Reminder) -> str:
  return f"Reminder [{r.uid}] @ {r.date.strftime(ISO8061)} - {r.message}"


@runtime_callable
def add_event(name: str, date: str, time: str, comment: str | None = None) -> str:
  """Add a new event to the user's calendar.

  Args:
    name: A name of this event
    date: ISO8601 YYYY-MM-DD format date string
    time: ISO8601 hours:minutes:seconds format time string
    comment: Optional comment for this event

  Returns:
    A uid of the added event
  """

  date = datetime.fromisoformat(f"{date}T{time}").astimezone(runtime.tz)
  e = Event(name=name, date=date, comment=comment, timezone=runtime.timezone)
  return calendar.add_event(e)


@runtime_callable
def get_event(uid: str) -> str:
  """Get an event by this uid.

  Args:
    uid: A unique identifier of the event needed.

  Raises:
    KeyError: If no events found by the given uid
  """

  e = calendar.get_event(uid)
  return _format_event(e)


@runtime_callable
def remove_event(uid: str) -> str:
  """Remove an event by this uid.

  Args:
    uid: A unique identifier of the event needed.

  Raises:
    KeyError: If no events found by the given uid

  Returns:
    A string status of the event removal
  """

  calendar.remove_event(uid)
  return f"Event [{uid}] removed."


@runtime_callable
def edit_event(uid: str, date: str | None = None, time: str | None = None, comment: str | None = None) -> str:
  """Change some properties of an existing event.

  Args:
    uid: The event's unique identifier
    date: A new event's date (ISO8601 YYYY-MM-DD format)
    time: A new event's time (ISO8601 hours:minutes:seconds format)
    comment: A new event's comment

  Returns:
    A string status of the event update
  """

  e = calendar.get_event(uid)
  old_date, old_time = e.date.isoformat(" ").split()
  date = datetime.fromisoformat(f"{date or old_date}T{time or old_time}")
  calendar.edit_event(uid, date=date, comment=comment)
  return f"Updated: {get_event(uid)}"


@runtime_callable
def get_all_events(start: str = CURRENT.DATE, end: str = CURRENT.DATE, regex: str | None = None) -> str:
  """Get all events from the user's calendar.

  Args:
    start: From which date to lookup the events (ISO8601 YYYY-MM-DD format)
    end: Until which date (inclusive) to lookup the events (ISO8601 YYYY-MM-DD format)
    regex: Optional regular expression to use for filtering events based on the name or comment properties

  Returns:
    A string result of events search
  """

  start = datetime.fromisoformat(start).astimezone(runtime.tz)
  end = datetime.fromisoformat(end).astimezone(runtime.tz)

  start, end = sorted((start, end))
  # if end <= start:
  #   raise ValueError(f"'end' date ({end}) should be later then 'start' ({start})")
  if events := calendar.get_all_events(start, end, regex):
    return "Events:\n- " + "- ".join([_format_event(e) + "\n" for e in events])
  return "No events found for the given query."


@runtime_callable
def add_reminder(time: str, msg: str, date: str = CURRENT.DATE) -> str:
  """Add a new reminder to the user's calendar.

  Args:
    time: A new reminder's time (ISO8601 hours:minutes:seconds format)
    msg: A new reminder's message string
    date: A new reminder's date (ISO8601 YYYY-MM-DD format)

  Returns:
    A uid of the added reminder
  """

  date = datetime.fromisoformat(f"{date}T{time}").astimezone(runtime.tz)
  r = Reminder(msg, date, timezone=runtime.timezone)
  return calendar.add_reminder(r)


@runtime_callable
def remove_reminder(uid: str) -> str:
  """Remove a reminder by this uid.

  Args:
    uid: A unique identifier of the event needed.

  Raises:
    KeyError: If no reminders found by the given uid

  Returns:
    A string status of the reminder removal
  """

  calendar.remove_reminder(uid)
  return f"Reminder [{uid}] removed."


@runtime_callable
def get_reminder(uid: str) -> str:
  """Get a reminder by this uid.

  Args:
    uid: A unique identifier of the event needed.

  Raises:
    KeyError: If no reminders found by the given uid
  """

  r = calendar.get_reminder(uid)
  return _format_reminder(r)


@runtime_callable
def get_all_reminders(date: str = CURRENT.DATE) -> str:
  """Get all reminders on the given date in the user's calendar.

  Args:
    date: The date to get all reminders for (ISO8601 YYYY-MM-DD format)
  """

  date = datetime.fromisoformat(date)
  if reminders := calendar.get_all_reminders(date):
    return "Reminders:\n- " + "- ".join([_format_reminder(r) + "\n" for r in reminders])
  return f"No reminders found @ {date.strftime(ISO8061)}"
