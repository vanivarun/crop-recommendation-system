# src/weather.py
"""
Real weather integration for AgriSense.
Uses:
  - Nominatim (OpenStreetMap) for geocoding  — FREE, no API key
  - Open-Meteo for weather data              — FREE, no API key
"""
import requests
import streamlit as st

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "AgriSense/1.0 (crop-recommendation-system)"}


def geocode_city(city_name: str) -> dict | None:
    """
    Convert a city/village/district name to lat/lon using Nominatim.

    Args:
        city_name: Any place name — village, district, or city

    Returns:
        dict with lat, lon, display_name — or None if not found
    """
    try:
        params = {
            "q": city_name,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=HEADERS,
            timeout=6,
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            return None

        top = results[0]
        return {
            "lat": float(top["lat"]),
            "lon": float(top["lon"]),
            "display_name": top.get("display_name", city_name),
            "country": top.get("address", {}).get("country", ""),
        }

    except requests.exceptions.Timeout:
        st.warning("⏱️ Location lookup timed out. Try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("🌐 No internet connection for location lookup.")
        return None
    except Exception as e:
        st.warning(f"Location lookup failed: {e}")
        return None


def get_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch current weather from Open-Meteo for given coordinates.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        dict with temperature, humidity, rainfall, wind_speed — or None
    """
    try:
        params = {
            "latitude":  lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "weather_code",
            ]),
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 7,
        }
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=6,
        )
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        daily   = data.get("daily", {})

        # Estimate annual rainfall from 7-day avg × 52 weeks
        daily_rain_sum = daily.get("precipitation_sum", [0])
        avg_daily_mm   = sum(daily_rain_sum) / max(len(daily_rain_sum), 1)
        est_annual_mm  = round(avg_daily_mm * 365, 1)

        return {
            "temperature":  current.get("temperature_2m", 0),
            "humidity":     current.get("relative_humidity_2m", 0),
            "rainfall":     min(est_annual_mm, 299.0),   # clamp to model range
            "precipitation":current.get("precipitation", 0),
            "wind_speed":   current.get("wind_speed_10m", 0),
            "weather_code": current.get("weather_code", 0),
        }

    except requests.exceptions.Timeout:
        st.warning("⏱️ Weather fetch timed out. Try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("🌐 No internet connection for weather data.")
        return None
    except Exception as e:
        st.warning(f"Weather fetch failed: {e}")
        return None


# Weather code → human readable description
WEATHER_CODES = {
    0:  ("Clear sky", "☀️"),
    1:  ("Mainly clear", "🌤️"),
    2:  ("Partly cloudy", "⛅"),
    3:  ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Icy fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    61: ("Slight rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    80: ("Rain showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
}

def weather_description(code: int) -> tuple[str, str]:
    return WEATHER_CODES.get(code, ("Unknown", "🌡️"))


@st.cache_data(ttl=1800)   # cache for 30 minutes
def fetch_location_weather(city_name: str) -> dict | None:
    """
    Full pipeline: city name → coordinates → weather.
    Cached for 30 minutes to avoid hitting APIs on every rerun.

    Args:
        city_name: Village, district, or city name

    Returns:
        dict with geo + weather data combined — or None on failure
    """
    geo = geocode_city(city_name)
    if geo is None:
        return None

    weather = get_weather(geo["lat"], geo["lon"])
    if weather is None:
        return None

    return {**geo, **weather}
