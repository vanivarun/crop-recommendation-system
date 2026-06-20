# src/weather.py
import requests
import streamlit as st

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Required by Nominatim terms of service
HEADERS = {"User-Agent": "AgriSense-Crop-Recommendation/1.0 (crop-recommendation-system)"}


def geocode_city(city_name: str) -> dict | None:
    """Convert city/village name to lat/lon using Nominatim."""
    try:
        params = {"q": city_name, "format": "json", "limit": 1, "addressdetails": 1}
        response = requests.get(
            NOMINATIM_URL, params=params, headers=HEADERS, timeout=8
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
        }
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            return None  # Rate limited — fail silently
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def get_weather(lat: float, lon: float) -> dict | None:
    """Fetch current weather from Open-Meteo."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 7,
        }
        response = requests.get(OPEN_METEO_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        daily_rain = daily.get("precipitation_sum", [0])
        avg_daily_mm = sum(daily_rain) / max(len(daily_rain), 1)
        est_annual_mm = round(avg_daily_mm * 365, 1)

        return {
            "temperature": float(current.get("temperature_2m", 25)),
            "humidity":    float(current.get("relative_humidity_2m", 65)),
            "rainfall":    float(min(est_annual_mm, 299.0)),
            "wind_speed":  float(current.get("wind_speed_10m", 0)),
            "weather_code":int(current.get("weather_code", 0)),
        }
    except Exception:
        return None


WEATHER_CODES = {
    0:  ("Clear sky", "☀️"),    1: ("Mainly clear", "🌤️"),
    2:  ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),       61: ("Light rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),65: ("Heavy rain", "🌧️"),
    80: ("Rain showers", "⛈️"), 95: ("Thunderstorm", "⛈️"),
}

def weather_description(code: int) -> tuple:
    return WEATHER_CODES.get(code, ("Clear", "🌡️"))


# ── Cache for 24 hours — API called only ONCE per unique city name ──
@st.cache_data(ttl=86400)
def fetch_location_weather(city_name: str) -> dict | None:
    """
    Full pipeline: city name → coordinates → weather.
    Cached 24h so Streamlit reruns never hit the API twice for same city.
    Returns None silently on any failure — caller shows friendly fallback.
    """
    geo = geocode_city(city_name.strip())
    if geo is None:
        return None

    weather = get_weather(geo["lat"], geo["lon"])
    if weather is None:
        return None

    return {**geo, **weather}