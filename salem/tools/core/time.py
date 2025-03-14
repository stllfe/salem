# noqa: A005

from salem.tools.runtime import CURRENT
from salem.tools.runtime import runtime_callable
from salem.tools.types import TimeUnit


@runtime_callable
def get_now(location: str = CURRENT.LOCATION) -> str:
  """Get the current time in ISO8601 format: <hours:minutes:seconds>.

  Args:
    location: What location to get the time for (current by default)
  """


@runtime_callable
def add_time(dt: str, amount: int, unit: TimeUnit) -> str:
  """Add some time to the date to produce a new date.

  Args:
    dt: The date optinally with time in ISO8601 format: YYYY-MM-DD hours:minutes:seconds (like 2024-12-03 18:34:93).
      If no time provided, 00:00:00 is assumed
    amount: Amount of time to add to the given date (can be negative to subtract)
    unit: The unit of time added in human-readable format
  """


@runtime_callable
def set_timer(hours: int, minutes: int, seconds: int) -> int:
  """Set a timer for the given amount of time.

  Note: all values should be explicitly provided (e.g. `set_timer(0, 0, 10)` for a 10 seconds timer)
    and the total amount of time should not exceed the 24 hours limit

  Raises:
    ValueError: if the total amount of time exceeds 24 hours

  Returns:
    A uid of the timer added
  """


@runtime_callable
def get_timer(uid: str) -> str:
  """Get a running timer information by the given uid."""


@runtime_callable
def get_all_timers() -> str:
  """Get all currently running timers information."""


@runtime_callable
def remove_timer(uid: str) -> str:
  """Remove a timer by this uid.

  Raises:
    KeyError: If no timers found by the given uid

  Returns:
    A string status of the timer removal
  """
