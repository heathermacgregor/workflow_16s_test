# ==================================================================================== #

# Standard Imports
import logging
import requests
from functools import lru_cache

# Third Party Imports
import numpy as np

# Local Imports
from workflow_16s.constants import DEFAULT_USER_AGENT

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")
_session = requests.Session() # Create a single requests session for reuse

# ==================================================================================== #

@lru_cache(maxsize=None)
def _geocode_query(query: str, user_agent: str = DEFAULT_USER_AGENT) -> (float, float):
    """Get coordinates from Nominatim API with caching."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': query, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': user_agent}
    try:
        response = _session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        logger.error(f"Geocoding failed for '{query}': {e}")
    return None, None


def sph2cart(latitudes, longitudes, R=6371):
    """Convert spherical lat/lon to Cartesian coordinates."""
    φ = np.radians(latitudes.astype(float))
    λ = np.radians(longitudes.astype(float))
    x = R * np.cos(φ) * np.cos(λ)
    y = R * np.cos(φ) * np.sin(λ)
    z = R * np.sin(φ)
    return np.column_stack((x, y, z))
