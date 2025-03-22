from abc import ABC
from abc import abstractmethod

from salem.tools.types import LocationInfo
from salem.tools.types import Weather
from salem.tools.types import WeatherForecast


class WeatherProvider(ABC):
  @abstractmethod
  def get_location(self, name: str) -> LocationInfo:
    """Get a full location info based on location name."""

  @abstractmethod
  def get_weather(self, location: LocationInfo) -> Weather:
    """Get the current weather conditions."""

  @abstractmethod
  def get_forecast(self, location: LocationInfo, days: int) -> WeatherForecast:
    """Get a weather forecast for the next several days."""
