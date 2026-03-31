"""
WoSIS (World Soil Information Service) Environmental Data Handler

Provides access to the WoSIS REST API for soil properties validation and refinement.
Queries nearest soil profiles to provide measured soil characteristics:
- Soil pH
- Clay percentage
- Sand percentage
- Silt percentage
- Organic carbon content
- Profile depth

WoSIS provides direct measurements that can validate/refine SoilGrids predictions.
Free and open access without authentication.

Reference: https://www.isric.org/projects/wosis-rest-services
"""

import logging
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class WoSISAPI(BaseEnvironmentalAPI):
    """
    Query WoSIS (World Soil Information Service) for soil properties.
    
    Features:
    - Global soil profile measurements
    - Spatial search within defined radius (up to 25 km)
    - Multiple soil property measurements
    - No authentication required
    
    Returns:
    Dictionary with soil properties from nearest profiles within search radius:
    - soil_ph: Soil pH (unitless)
    - soil_clay_pct: Clay content (%)
    - soil_sand_pct: Sand content (%)
    - soil_silt_pct: Silt content (%)
    - soil_organic_carbon: Organic carbon (g/kg)
    - profile_depth_cm: Maximum profile depth (cm)
    - nearest_profile_distance_m: Distance to nearest profile (meters)
    - profiles_found: Number of profiles within search radius
    
    Example:
        api = WoSISAPI()
        is_available, msg = api.check_requirements()
        if is_available:
            soil_data = api.get_data(lat=0.0, lon=25.0)
            if soil_data:
                print(f"Soil pH: {soil_data.get('soil_ph')}")
    """
    
    API_NAME = "WoSIS"
    BASE_URL = "https://www.isric.org/projects/wosis-rest-services"
    
    # WoSIS API endpoints
    SEARCH_ENDPOINT = "https://www.isric.org/projects/wosis/wosis.api"
    
    # Soil properties to extract
    SOIL_PROPERTIES = {
        'ph_h2o': 'soil_ph',
        'clay': 'soil_clay_pct',
        'sand': 'soil_sand_pct',
        'silt': 'soil_silt_pct',
        'oc': 'soil_organic_carbon',
        'depth': 'profile_depth_cm'
    }
    
    def __init__(self, verbose: bool = False, search_radius_km: float = 25.0, enable_fallback: bool = False):
        """
        Initialize WoSIS API client.
        
        Args:
            verbose: Enable verbose logging
            search_radius_km: Search radius for spatial queries (default 25 km)
            enable_fallback: If True, retry with doubled radius if no profiles found
        """
        super().__init__(verbose=verbose)
        self.base_url = self.BASE_URL
        self.search_radius_km = search_radius_km
        self.enable_fallback = enable_fallback
        self.timeout = REQUEST_TIMEOUT
        # WoSIS endpoint is spatial-only; fetch_date is accepted for interface
        # compatibility but not used in the request itself.
        self.cache_key_exclude_kwargs.update({"fetch_date", "date", "collection_date"})
        self.logger = get_logger(__name__)

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Check if WoSIS API is accessible.
        
        Returns:
            Tuple of (is_available, error_message)
            is_available: True if API is reachable
            error_message: None if available, error description otherwise
        """
        try:
            # Test API connectivity with a simple request
            response = self.session.head(
                "https://www.isric.org/projects/wosis-rest-services",
                timeout=self.timeout
            )
            if response.status_code in [200, 301, 302]:
                self.logger.info("WoSIS API accessibility check passed")
                return True, None
            else:
                error_msg = f"WoSIS API returned HTTP {response.status_code}"
                self.logger.warning(error_msg)
                return False, error_msg
        except requests.exceptions.Timeout:
            error_msg = "WoSIS API timeout during connectivity check"
            self.logger.warning(error_msg)
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"WoSIS API connectivity error: {str(e)}"
            self.logger.warning(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"WoSIS API check failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    @cache_api_call
    def get_data(
        self,
        lat: float,
        lon: float,
        fetch_date: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve soil properties from nearest WoSIS profiles.
        
        NOTE: WoSIS is a sparse database with ~194,000 global profiles.
        Coverage is incomplete, especially in remote areas and developing regions.
        Use with SoilGrids as primary source for predictive soil data.
        
        Searches for soil profiles within the specified radius and aggregates
        properties from the nearest ones, returning mean values.
        
        Args:
            lat: Latitude of query location (-90 to 90)
            lon: Longitude of query location (-180 to 180)
            fetch_date: Optional date parameter (not used but included for API consistency)
            **kwargs: Additional keyword arguments (e.g., logger from decorator)
            
        Returns:
            Dictionary with soil properties or None if no data found
            Returns empty dict with null values if API unavailable but location valid
            
        Raises:
            No exceptions raised; errors are logged and None returned
        """
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None
            
            logger = kwargs.get('logger', self.logger)
            
            # Query WoSIS REST API for soil profiles near location
            # Using a simple spatial search approach
            result = self._query_wosis_profiles(lat, lon, logger=logger)
            
            if result is None:
                logger.debug(f"No WoSIS data found at ({lat:.4f}, {lon:.4f})")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"WoSIS query failed at ({lat:.4f}, {lon:.4f}): {str(e)}")
            return None

    def _query_wosis_profiles(
        self,
        lat: float,
        lon: float,
        logger: Optional[Any] = None,
        max_profiles: int = 5,
        search_radius_km: Optional[float] = None,
        attempt: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Query WoSIS REST API for soil profiles within search radius.
        
        Args:
            lat: Latitude
            lon: Longitude
            logger: Logger instance
            max_profiles: Maximum number of profiles to retrieve
            search_radius_km: Search radius (uses self.search_radius_km if None)
            attempt: Internal counter for fallback attempts
            
        Returns:
            Aggregated soil property dictionary or None if no data
        """
        if logger is None:
            logger = self.logger
        
        if search_radius_km is None:
            search_radius_km = self.search_radius_km
        
        try:
            # WoSIS API typically requires specific query format
            # Construct a bounding box around the point (rough: 1 deg ≈ 111 km)
            radius_deg = search_radius_km / 111.0
            
            # Build query parameters
            params = {
                'north': lat + radius_deg,
                'south': lat - radius_deg,
                'east': lon + radius_deg,
                'west': lon - radius_deg,
                'limit': max_profiles,
                'IncludeQualityFlags': 'true'
            }
            
            # Query WoSIS API
            logger.debug(f"WoSIS: Querying with radius {search_radius_km}km at ({lat:.4f}, {lon:.4f})")
            
            response = self.session.get(
                self.SEARCH_ENDPOINT,
                params=params,
                timeout=self.timeout
            )
            
            # Better status code handling
            if response.status_code == 404:
                logger.info(
                    f"WoSIS: No profiles found at ({lat:.4f}, {lon:.4f}) "
                    f"within {search_radius_km}km radius (sparse coverage)"
                )
                
                # Optional fallback: try with larger radius
                if self.enable_fallback and attempt == 1 and search_radius_km < 100:
                    logger.debug(f"WoSIS: Retrying with extended radius {search_radius_km * 2}km")
                    return self._query_wosis_profiles(
                        lat, lon, 
                        logger=logger,
                        max_profiles=max_profiles,
                        search_radius_km=search_radius_km * 2,
                        attempt=2
                    )
                
                return None
            elif response.status_code != 200:
                logger.warning(
                    f"WoSIS API error: HTTP {response.status_code} at ({lat:.4f}, {lon:.4f})"
                )
                return None
            
            data = response.json()
            
            # Check if profiles were returned
            if not data or 'profiles' not in data or len(data['profiles']) == 0:
                logger.info(
                    f"WoSIS: No soil profiles in response for ({lat:.4f}, {lon:.4f}) "
                    f"within {search_radius_km}km (sparse coverage)"
                )
                logger.debug(
                    "Note: WoSIS is a sparse supplementary source. "
                    "Use SoilGrids as primary for complete coverage."
                )
                
                # Optional fallback
                if self.enable_fallback and attempt == 1 and search_radius_km < 100:
                    logger.debug(f"WoSIS: Retrying with extended radius {search_radius_km * 2}km")
                    return self._query_wosis_profiles(
                        lat, lon,
                        logger=logger,
                        max_profiles=max_profiles,
                        search_radius_km=search_radius_km * 2,
                        attempt=2
                    )
                
                return None
            
            # Success case
            profiles = data['profiles']
            aggregated = self._aggregate_profile_properties(profiles, logger=logger)
            
            if aggregated:
                aggregated['profiles_found'] = len(profiles)
                aggregated['nearest_profile_distance_m'] = self._calculate_distance(
                    lat, lon, 
                    profiles[0].get('latitude'), 
                    profiles[0].get('longitude')
                )
                aggregated['search_radius_km'] = search_radius_km
                logger.debug(f"WoSIS: Aggregated {len(profiles)} soil profiles")
            
            return aggregated
            
        except requests.exceptions.Timeout:
            logger.warning(f"WoSIS API timeout at ({lat:.4f}, {lon:.4f})")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"WoSIS API request failed: {str(e)}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"WoSIS: Error parsing response: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"WoSIS: Unexpected error: {str(e)}")
            return None

    def _aggregate_profile_properties(
        self,
        profiles: list,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate soil properties from multiple profiles.
        
        Computes mean values across profiles for each property.
        
        Args:
            profiles: List of profile dictionaries from WoSIS
            logger: Logger instance
            
        Returns:
            Dictionary with aggregated properties or None if no valid data
        """
        if logger is None:
            logger = self.logger
        
        try:
            aggregated = {}
            property_counts = {}
            
            # Sum up all property values
            for profile in profiles:
                if not isinstance(profile, dict):
                    continue
                
                # Extract properties from profile
                properties = profile.get('properties', {})
                if not properties:
                    continue
                
                for wosis_key, output_key in self.SOIL_PROPERTIES.items():
                    if wosis_key in properties:
                        value = properties.get(wosis_key)
                        
                        # Validate numeric value
                        try:
                            value = float(value)
                            if output_key not in aggregated:
                                aggregated[output_key] = 0.0
                                property_counts[output_key] = 0
                            aggregated[output_key] += value
                            property_counts[output_key] += 1
                        except (ValueError, TypeError):
                            logger.debug(f"Invalid value for {wosis_key}: {value}")
                            continue
            
            # Calculate means
            for key in aggregated:
                if property_counts.get(key, 0) > 0:
                    aggregated[key] = aggregated[key] / property_counts[key]
            
            return aggregated if aggregated else None
            
        except Exception as e:
            logger.error(f"Error aggregating profile properties: {str(e)}")
            return None

    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate approximate distance between two coordinates in meters.
        
        Uses simplified haversine for efficiency.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Approximate distance in meters
        """
        try:
            import math
            
            if lat2 is None or lon2 is None:
                return float('nan')
            
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            lon1_rad = math.radians(lon1)
            lon2_rad = math.radians(lon2)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            # Earth radius in meters
            earth_radius_m = 6371000
            
            return earth_radius_m * c
        except Exception as e:
            self.logger.debug(f"Distance calculation error: {e}")
            return float('nan')
