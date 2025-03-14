from salem.tools.core.backend.weather import WeatherProvider
from salem.tools.runtime import CURRENT
from salem.tools.runtime import runtime
from salem.tools.runtime import runtime_callable
from salem.tools.types import Weather
from salem.utils import get_logger


weather = runtime.get_backend(WeatherProvider)
logger = get_logger()


def _format_weather(w: Weather) -> str:
  return (
    f"[{w.date}] @ {w.temperature:.2f} {w.units}°; humidity {w.humidity:.2f}%; wind: {w.wind_speed:.2f} metres/seconds"  # noqa
  )


@runtime_callable
def get_weather(location: str = CURRENT.LOCATION) -> str:
  """Get the current weather conditions.

  Args:
    location: What location to get the weather for (current by default)
  """

  w = weather.get_weather(location)
  logger.debug(f"weather call:\n{w!r}")

  return f"Weather @ {w.location}: {_format_weather(w)}"


@runtime_callable
def get_forecast(days: int, location: str = CURRENT.LOCATION) -> str:
  """Get a weather forcast for the next several days.

  Note: only up-to 21 days is supported, should be at least 1 (1 day ahead)

  Args:
    days: The amount of days ahead to get the forecast for (1..21, both ends inclusive)
    location: What location to get the forecast for (current by default)

  Raises:
    ValueError: If the amount of days is not between 1 and 21
  """

  if forecast := weather.get_forecast(days, location):
    title = f"Weather forcast for {days} days in {location}:\n- "
    return title + "- ".join([_format_weather(w) + "\n" for w in forecast])
  return "No events found for the given query."
