from datetime import datetime
from typing import Any

import openmeteo_requests as omr
import orjson
import requests_cache as rc
import retry_requests as rr

from attrs import define
from attrs import field
from requests import Session

from salem.tools.core.backend.weather.base import WeatherProvider
from salem.tools.types import JsonMixin
from salem.tools.types import Language
from salem.tools.types import TempUnit
from salem.tools.types import Weather
from salem.tools.types import WeatherForecast


URLS = {
  "historical_weather": "https://archive-api.open-meteo.com/v1/archive",
  "geocoding": "https://geocoding-api.open-meteo.com/v1/search",
  "air_quality": "https://air-quality-api.open-meteo.com/v1/air-quality",
  "elevation": "https://api.open-meteo.com/v1/elevation",
  "weather_forecast": "https://api.open-meteo.com/v1/forecast",
}


@define
class LocationInfo(JsonMixin):
  name: str
  country: str
  latitude: float
  longitude: float
  timezone: str


# Setup the Open-Meteo API client with cache and retry on error
cache_session = rc.CachedSession(".cache", expire_after=3600)
retry_session = rr.retry(cache_session, retries=5, backoff_factor=0.2)
client = omr.Client(session=retry_session)


@define
class OpenMeteoWeatherProvider(WeatherProvider):
  language: Language = "ru"
  units: TempUnit = TempUnit.C
  session: Session = field(default=retry_session)
  openmeteo: omr.Client = field(default=client)

  def _send_request(self, url: str, params: dict[str, Any] | None = None) -> dict:
    # check for url shorthand
    url = URLS.get(url, url)

    response = self.session.get(url, params=params)
    response.raise_for_status()  # raise an exception for bad status codes

    return orjson.loads(response.content)

  def _get_info(self, location: str) -> LocationInfo:
    response = self._send_request(
      URLS["geocoding"],
      params={
        "name": location.strip().title(),
        "count": 1,  # take only a single top match
        "language": self.language,
        "format": "json",
      },
    )
    return LocationInfo.load(data=response["results"][0])

  def _get_weather(self, info: LocationInfo) -> Weather:
    params = {
      "latitude": info.latitude,
      "longitude": info.longitude,
      "current": [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "pressure_msl",
        "wind_speed_10m",
        "rain",
      ],
      "hourly": "temperature_2m",
      "temperature_unit": str(self.units),
      "wind_speed_unit": "ms",
    }
    responses = self.openmeteo.weather_api(URLS["weather_forecast"], params=params)
    response = responses[0]

    location = f"{info.country}/{info.name}"
    current = response.Current()
    assert current, f"Error getting weather info for location: {location}"

    return Weather(
      location=location,
      temperature=current.Variables(0).Value(),
      feels_like=current.Variables(1).Value(),
      humidity=current.Variables(2).Value(),
      pressure=current.Variables(3).Value(),
      wind_speed=current.Variables(4).Value(),
      date=datetime.now(),
      units=self.units,
    )

  def get_weather(self, location: str) -> Weather:
    info = self._get_info(location)
    return self._get_weather(info)

  def get_forecast(self, days: int, location: str) -> WeatherForecast:
    # TODO: implement this
    pass
