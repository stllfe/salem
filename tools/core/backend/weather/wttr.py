from datetime import datetime
from typing import Iterable

import orjson

from attrs import define
from attrs import field
from requests import Session

from tools.core.backend.weather.base import WeatherProvider
from tools.types import Language
from tools.types import Weather


URL = "https://wttr.in/{location}?Q?m&format=j1&lang={language}"


@define
class WttrWeatherProvider(WeatherProvider):
  language: Language = "ru"
  session: Session = field(factory=Session)

  def _send_request(self, location: str) -> dict:
    # send a GET request to the URL
    url = URL.format(location=location, language=self.language)

    response = self.session.get(url)  # cache this url response
    response.raise_for_status()  # raise an exception for bad status codes

    # print(response.json())
    return orjson.loads(response.content)

  def get_weather(self, location: str) -> Weather:
    data = self._send_request(location)
    current = data["current_condition"][0]

    # TODO: check irene parsing
    return Weather(
      location=data["nearest_area"][0]["areaName"][0]["value"],
      temperature=float(current["temp_C"]),
      feels_like=float(current["FeelsLikeC"]),
      humidity=float(current["humidity"]),
      pressure=float(current["pressure"]),
      wind_speed=float(current["windspeedKmph"]) * 1000 / 3600,
      date=datetime.strptime(current["localObsDateTime"], "%Y-%m-%d %I:%M %p"),
      units="C",
    )

  def get_forecast(self, days: int, location: str) -> Iterable[Weather]:
    # TODO: how to get more days
    data = self._send_request(location)
    for day in data["weather"][:days]:
      hourly = day["hourly"]
      # temp = sum(float(hour["tempC"]) for hour in day["hourly"]) / len(hourly)
      # TODO: same for other ?
      hour = hourly[2]
      yield Weather(
        location=data["nearest_area"][0]["areaName"][0]["value"],
        temperature=hour["tempC"],
        feels_like=float(hour["FeelsLikeC"]),
        humidity=float(hour["humidity"]),
        pressure=float(hour["pressure"]),
        wind_speed=float(hour["windspeedKmph"]),
        date=datetime.strptime(f"{day['date']} {hour['time'].zfill(4)}", "%Y-%m-%d %H%M"),
        units="C",
      )
