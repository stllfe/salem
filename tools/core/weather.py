from tools.runtime import CURRENT


def get_weather(location: str = CURRENT.LOCATION) -> str:
  """Get the current weather conditions.

  Args:
    location: What location to get the weather for (current by default)
  """


def get_forecast(days: int, location: str = CURRENT.LOCATION) -> str:
  """Get a weather forcast for the next several days.

  Note: only up-to 21 days is supported, should be at least 1 (1 day ahead)

  Args:
    days: The amount of days ahead to get the forecast for (1..21, both ends inclusive)
    location: What location to get the forecast for (current by default)

  Raises:
    ValueError: If the amount of days is not between 1 and 21
  """
