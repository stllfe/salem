from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable

from tools.types import Weather


class WeatherProvider(ABC):
  @abstractmethod
  def get_weather(self, location: str) -> Weather:
    """Get the current weather conditions."""

  @abstractmethod
  def get_forecast(self, days: int, location: str) -> Iterable[Weather]:
    """Get a weather forecast for the next several days."""
