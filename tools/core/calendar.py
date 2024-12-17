# noqa: A005
from tools.runtime import CURRENT


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


def get_event(uid: str) -> str:
  """Get an event by this uid.

  Raises:
    KeyError: If no events found by the given uid
  """


def remove_event(uid: str) -> str:
  """Remove an event by this uid.

  Raises:
    KeyError: If no events found by the given uid

  Returns:
    A string status of the event removal
  """


def edit_event(uid: str, date: str | None = None, time: str | None = None, comment: str | None = None) -> str:
  """Change some properties of an existing event.

  Args:
    uid: The event's uid
    date: A new event's date (ISO8601 YYYY-MM-DD format)
    time: A new event's time (ISO8601 hours:minutes:seconds format)
    comment: A new event's comment

  Returns:
    A string status of the event update
  """


def get_all_events(start: str = CURRENT.DATE, end: str = CURRENT.DATE, regex: str | None = None) -> str:
  """Get all events from the user's calendar.

  Args:
    start: From which date to lookup the events (ISO8601 YYYY-MM-DD format)
    end: Until which date (inclusive) to lookup the events (ISO8601 YYYY-MM-DD format)
    regex: Optional regular expression to use for filtering events based on the name or comment properties

  Returns:
    A string result of events search
  """


def add_reminder(time: str, msg: str, date: str = CURRENT.DATE) -> int:
  """Add a new reminder to the user's calendar.

  Args:
    time: A new reminder's time (ISO8601 hours:minutes:seconds format)
    msg: A new reminder's message string
    date: A new reminder's date (ISO8601 YYYY-MM-DD format)

  Returns:
    A uid of the added reminder
  """


def remove_reminder(uid: str) -> str:
  """Remove a reminder by this uid.

  Raises:
    KeyError: If no reminders found by the given uid

  Returns:
    A string status of the reminder removal
  """


def get_reminder(uid: str) -> str:
  """Get a reminder by this uid.

  Raises:
    KeyError: If no reminders found by the given uid
  """


def get_all_reminders(date: str = CURRENT.DATE) -> str:
  """Get all reminders on the given date in the user's calendar."""
