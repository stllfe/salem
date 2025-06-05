from datetime import datetime
from typing import Any

import openmeteo_requests as omr
import orjson
import requests_cache as rc
import retry_requests as rr

from attrs import define
from attrs import field
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse
from requests import Session

from salem.tools.core.backend.weather.base import WeatherProvider
from salem.tools.types import DayWeather
from salem.tools.types import Language
from salem.tools.types import LocationInfo
from salem.tools.types import TempUnit
from salem.tools.types import Weather
from salem.tools.types import WeatherForecast


MAX_DAYS = 14  # we can maybe use historical weather on anything beyond this threshold

# https://open-meteo.com/en/docs
URLS = {
  "historical_weather": "https://archive-api.open-meteo.com/v1/archive",
  "geocoding": "https://geocoding-api.open-meteo.com/v1/search",
  "air_quality": "https://air-quality-api.open-meteo.com/v1/air-quality",
  "elevation": "https://api.open-meteo.com/v1/elevation",
  "weather_forecast": "https://api.open-meteo.com/v1/forecast",
}


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

  def _get_forecast(self, location: LocationInfo, days: int = MAX_DAYS) -> WeatherApiResponse:
    values = [
      "temperature_2m",
      "apparent_temperature",
      "relative_humidity_2m",
      "surface_pressure",
      "wind_speed_10m",
      "rain",
    ]
    params = {
      "latitude": location.latitude,
      "longitude": location.longitude,
      "current": values,
      "hourly": values,
      "temperature_unit": str(self.units),
      "forecast_days": days,
      "wind_speed_unit": "ms",
    }
    responses = self.openmeteo.weather_api(URLS["weather_forecast"], params=params)
    return responses[0]  # response per location, so we always get the first

  def get_location(self, name: str) -> LocationInfo:
    response = self._send_request(
      URLS["geocoding"],
      params={
        "name": name.strip().title(),
        "count": 1,  # take only a single top match
        "language": self.language,
        "format": "json",
      },
    )
    return LocationInfo.load(data=response["results"][0])

  def get_weather(self, location: LocationInfo) -> Weather:
    resp = self._get_forecast(location)
    curr = resp.Current()
    assert curr, f"Error getting weather info for location: {location}"
    return Weather(
      temperature=curr.Variables(0).Value(),
      feels_like=curr.Variables(1).Value(),
      humidity=curr.Variables(2).Value(),
      pressure=curr.Variables(3).Value(),
      wind_speed=curr.Variables(4).Value(),
      date=datetime.now().astimezone(location.tz),
      units=self.units,
    )

  def get_forecast(self, location: LocationInfo, days: int = 14) -> WeatherForecast:
    import pandas as pd

    resp = self._get_forecast(location)
    hourly = resp.Hourly()
    assert hourly, f"Error getting weather info for location: {location}"

    data = pd.DataFrame({
      "units": self.units,
      "temperature": hourly.Variables(0).ValuesAsNumpy(),
      "feels_like": hourly.Variables(1).ValuesAsNumpy(),
      "humidity": hourly.Variables(2).ValuesAsNumpy(),
      "pressure": hourly.Variables(3).ValuesAsNumpy(),
      "wind_speed": hourly.Variables(4).ValuesAsNumpy(),
      "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
      ),
    })
    data = data[data.date.dt.hour.isin([6, 12, 18])]  # keep only: morning, daytime, evening measurements
    daily: list[DayWeather] = []
    for _, values in data.groupby(data.date.dt.date):
      values = values.sort_values("date")
      day: list[Weather] = []
      for v in values.itertuples(index=False):
        day.append(
          Weather(
            temperature=v.temperature,
            feels_like=v.feels_like,
            humidity=v.humidity,
            pressure=v.pressure,
            wind_speed=v.wind_speed,
            date=v.date.to_pydatetime().astimezone(location.tz),
            units=v.units,
          )
        )
      daily.append(DayWeather(*day))
    return WeatherForecast(location=location, daily=daily[:days])


if __name__ == "__main__":
  wp = OpenMeteoWeatherProvider()
  loc = wp.get_location("Moscow")
  foc = wp.get_forecast(loc)
  tmp = foc.get("temperature")
  print(tmp.avg())
