from salem.tools.core.backend.weather import WeatherProvider
from salem.tools.runtime import CURRENT
from salem.tools.runtime import ISO8061
from salem.tools.runtime import ISO8061_DATE
from salem.tools.runtime import runtime
from salem.tools.runtime import runtime_callable
from salem.tools.types import DayWeather
from salem.tools.types import Weather
from salem.utils import get_logger


weather = runtime.get_backend(WeatherProvider)
logger = get_logger()


def _format_weather(w: Weather) -> str:
  return f"[{w.date.strftime(ISO8061)}] @ temperature: {w.temperature:.2f} {w.units.name}°, humidity {w.humidity:.2f}%, wind: {w.wind_speed:.2f} m/s"  # noqa


def _format_daily(d: DayWeather) -> str:
  date = d.morning.date.strftime(ISO8061_DATE)
  return f"[{date}] @ temperature {d.daytime.units.name}° | morning: {d.morning.temperature:.2f}, daytime: {d.daytime.temperature:.2f}, evening: {d.daytime.temperature:.2f}"  # noqa


@runtime_callable
def get_weather(location: str = CURRENT.LOCATION) -> str:
  """Get the current weather conditions.

  Args:
    location: What location to get the weather for (current by default)
  """

  info = weather.get_location(location)
  w = weather.get_weather(info)
  return f"Weather @ {info}: {_format_weather(w)}"


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

  info = weather.get_location(location)
  if forecast := weather.get_forecast(info, days):
    title = f"Weather forcast for {forecast.num_days} days in {forecast.location}:\n- "
    return title + "- ".join([_format_daily(d) + "\n" for d in forecast.daily])
  return "No forecast found for the given query."
