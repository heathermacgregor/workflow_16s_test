# ==================================================================================== #

# Standard Library Imports
import requests
from typing import Any, Dict, Optional

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, REQUEST_TIMEOUT, cache_api_call
)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class NWS_API(BaseEnvironmentalAPI):
    """
    Fetches weather forecast from the US National Weather Service API.
    
    Documentation: https://www.weather.gov/documentation/services-web-api
    
    Attributes:
        base_url (str): Base URL for the NWS API.
        verbose (bool): If True, enables verbose logging.
    """
    URL = "https://api.weather.gov"
    def __init__(self, email: str = "contact@example.com", verbose: bool = False):
        self.verbose = verbose
        super().__init__(verbose=self.verbose)
        self.base_url = self.URL
        self.session.headers.update({'User-Agent': f'(Environmental Data Script, {email})'})
    
    @cache_api_call
    def get_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]: # type: ignore
        """Retrieves the NWS weather forecast for a specific lat/lon."""
        try:
            points_url = f"{self.base_url}/points/{lat:.4f},{lon:.4f}"
            if self.verbose: logger.info(f"⏳ Fetching NWS gridpoint data for ({lat}, {lon})...")
            response = self.session.get(points_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            properties = response.json().get("properties", {})
            if not properties or "forecast" not in properties:
                logger.warning("🟨 Could not find forecast URL in NWS response")
                return None
            
            forecast_url = properties["forecast"]
            if self.verbose: logger.info(f"⏳ Fetching NWS forecast from {forecast_url}...")
            response = self.session.get(forecast_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            forecast_periods = response.json().get("properties", {}).get("periods", [])
            
            return {"forecast_periods": forecast_periods[:4]}  # Return next 4 periods
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 NWS API request failed: {e}")
            return None


class SevereWeatherAPI(BaseEnvironmentalAPI):
    """
    Checks for active severe weather alerts from the official NWS API.
    
    Documentation: https://www.weather.gov/documentation/services-web-api
    
    Attributes:
        base_url (str): Base URL for the NWS API.
        verbose (bool): If True, enables verbose logging.
    """
    URL = "https://api.weather.gov"
    def __init__(self, email: str = "contact@example.com", verbose: bool = False):
        super().__init__(verbose=verbose)
        self.base_url = self.URL
        self.verbose = verbose
        self.session.headers.update({'User-Agent': f'(Environmental Data Script, {email})'})
    
    @cache_api_call
    def get_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]: # type: ignore
        """Retrieves active NWS alerts for a location."""
        try:
            points_url = f"{self.base_url}/points/{lat:.4f},{lon:.4f}"
            if self.verbose: logger.info(f"⏳ Fetching NWS zone information for ({lat}, {lon})...")
            response = self.session.get(points_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            properties = response.json().get("properties", {})
            if not properties or "forecastZone" not in properties:
                logger.warning("🟨 Could not determine the forecast zone")
                return {}
            
            zone_id = properties["forecastZone"].split('/')[-1]
            
            alerts_url = f"{self.base_url}/alerts/active/zone/{zone_id}"
            if self.verbose: logger.info(f"⏳ Fetching active alerts for zone {zone_id}...")
            response = self.session.get(alerts_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            alerts_data = response.json().get("features", [])
            
            active_alerts = []
            for alert in alerts_data:
                props = alert.get("properties", {})
                active_alerts.append({
                    "event": props.get("event"), "headline": props.get("headline"),
                    "severity": props.get("severity"),
                    "description": props.get("description")
                })
            
            return {"alerts": active_alerts}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🟥 Severe weather API request failed: {e}")
            return None

# ==================================================================================== #

if __name__ == "__main__":

    print("--- Running NWS API Example Test ---")
    
    # Berkeley, CA coordinates 🌦️
    berkeley_lat = 37.8715
    berkeley_lon = -122.2730
    
    # --- NWS Forecast API Test ---
    print("\n" + "="*50)
    print("## Test 1: Fetching Weather Forecast...")
    print("="*50)
    
    # The NWS API requires a User-Agent header, often an email.
    nws_client = NWS_API(email="your.email@example.com", verbose=True)
    forecast_data = nws_client.get_data(berkeley_lat, berkeley_lon)
    
    if forecast_data and forecast_data.get("forecast_periods"):
        print("\n✅ Success! Found forecast for Berkeley, CA:")
        for period in forecast_data["forecast_periods"]:
            name = period.get('name', 'N/A')
            temp = period.get('temperature', 'N/A')
            unit = period.get('temperatureUnit', '')
            forecast = period.get('shortForecast', 'N/A')
            print(f"  - {name}: {temp}°{unit}, {forecast}")
    else:
        print("\n❌ Failed to retrieve forecast data.")

    # --- Severe Weather Alert API Test ---
    print("\n" + "="*50)
    print("## Test 2: Checking for Severe Weather Alerts...")
    print("="*50)

    severe_weather_client = SevereWeatherAPI(email="your.email@example.com", verbose=True)
    alert_data = severe_weather_client.get_data(berkeley_lat, berkeley_lon)

    if alert_data and alert_data.get("alerts"):
        print(f"\n🚨 Found {len(alert_data['alerts'])} active alert(s):")
        for alert in alert_data["alerts"]:
            print(f"  - Event: {alert.get('event')}")
            print(f"    Headline: {alert.get('headline')}")
    elif alert_data is not None:
        print("\n✅ Success! No active severe weather alerts for this location.")
    else:
        print("\n❌ Failed to retrieve alert data.")