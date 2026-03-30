"""
Global Power Plant Database Environmental Data Handler

Provides access to the Global Power Plant Database for industrial proximity analysis.
The Global Power Plant Database is a comprehensive, open-source database of power
plants around the world containing location, capacity, generation, and ownership data.

Data Source: World Resources Institute (WRI) Global Power Plant Database
- Official Site: https://www.wri.org/research/global-power-plant-database
- GitHub Repo: https://github.com/wri/global-power-plant-database
- Version: 1.3.0 (final maintained version, as of June 2020)
- Status: Project no longer maintained by WRI as of early 2022
- Data Format: CSV, GeoJSON
- Update Frequency: Stable/No updates planned
- Coverage: ~35,000 power plants globally
- Includes: Capacity (MW), fuel type, generation data
- Public domain (CC BY 4.0 license)

Industries Covered:
- Coal, natural gas, and oil-fired plants
- Hydroelectric facilities
- Wind and solar installations  
- Nuclear power plants
- Biomass and waste facilities
- Geothermal installations

Use Cases:
- Industrial influence on environmental microbiota
- Proximity to emission sources (air/water quality)
- Grid connectivity patterns
- Resource extraction areas

Variables Extracted:
- Nearest plant distance (km)
- Plant count by distance bands (10, 25, 50, 100 km)
- Dominant fuel type
- Total nearby capacity (MW)
- Plant density metrics

Reference: https://github.com/wri/global-power-plant-database
"""

import logging
import requests
import csv
import io
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta
import json
import math

from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call
from .constants import CACHE_DIR
from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class GlobalPowerPlantsAPI(BaseEnvironmentalAPI):
    """
    Query Global Power Plant Database for industrial proximity analysis.
    
    Features:
    - ~35,000 power plants globally (v1.3.0, stable)
    - Multiple fuel types
    - Plant capacity and generation data
    - Lightweight spatial indexing for fast nearest-neighbor search
    - No authentication required (public data, CC BY 4.0 license)
    - Fallback to cached CSV if download fails
    
    Note: This database is no longer actively maintained by WRI (as of early 2022).
    Version 1.3.0 is the final release. The GitHub repository contains the stable source.
    
    Returns:
    Dictionary with industrial proximity metrics:
    - nearest_plant_distance_km: Distance to closest plant (km)
    - nearby_plant_count_10km: Number of plants within 10 km radius
    - nearby_plant_count_25km: Number of plants within 25 km radius
    - nearby_plant_count_50km: Number of plants within 50 km radius
    - nearby_plant_count_100km: Number of plants within 100 km radius
    - dominant_fuel_type: Most common fuel type in 100 km radius
    - total_nearby_capacity_mw: Sum of MW capacity within 100 km
    - plant_density_per_km2: Plant count per 1000 km² (100 km radius)
    
    Example:
        api = GlobalPowerPlantsAPI()
        is_available, msg = api.check_requirements()
        if is_available:
            proximity_data = api.get_data(lat=0.0, lon=25.0)
            if proximity_data:
                print(f"Nearest plant: {proximity_data.get('nearest_plant_distance_km')} km")
    """
    
    API_NAME = "GlobalPowerPlants"
    # GitHub raw URL for v1.3.0 (WRI Global Power Plant Database, final maintained version)
    # Project no longer maintained as of early 2022, so we use the stable GitHub source
    DATA_URL = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
    # Fallback URLs if primary source fails
    FALLBACK_URLS = [
        "https://datasets.wri.org/dataset/globalpowerplantdatabase",  # WRI datasets portal (may require manual download)
    ]
    FALLBACK_CSV_NAME = "global_power_plants.csv"
    
    # Fuel types mapping
    FUEL_TYPES = {
        'Coal': 'coal',
        'Natural Gas': 'gas',
        'Oil': 'oil',
        'Nuclear': 'nuclear',
        'Hydro': 'hydro',
        'Wind': 'wind',
        'Solar': 'solar',
        'Geothermal': 'geothermal',
        'Biomass': 'biomass',
    }
    
    def __init__(self, verbose: bool = False, cache_dir: Optional[Path] = None):
        """
        Initialize Global Power Plants API client.
        
        Args:
            verbose: Enable verbose logging
            cache_dir: Directory to cache power plant database CSV
        """
        super().__init__(verbose=verbose)
        self.base_url = "https://datasets.wri.org"
        self.timeout = REQUEST_TIMEOUT
        self.logger = get_logger(__name__)
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = CACHE_DIR / "power_plants"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Plant database cache (populated lazily)
        self._plants = None
        self._plants_loaded = False

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Check if power plant database is accessible.
        
        Returns:
            Tuple of (is_available, error_message)
            is_available: True if database accessible (cached or downloadable)
            error_message: None if available, error description otherwise
        """
        try:
            # Check if we can load the database (either cached or from URL)
            cached_file = self.cache_dir / self.FALLBACK_CSV_NAME
            
            if cached_file.exists():
                self.logger.info("Global Power Plants database found in cache")
                return True, None
            
            # Try to download database file
            response = self.session.head(
                self.DATA_URL,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                self.logger.info("Global Power Plants database accessible for download")
                return True, None
            else:
                error_msg = f"Power plant database returned HTTP {response.status_code}"
                self.logger.warning(error_msg)
                return False, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Power plant database timeout during connectivity check"
            self.logger.warning(error_msg)
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Power plant database connectivity error: {str(e)}"
            self.logger.warning(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Power plant check failed: {str(e)}"
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
        Retrieve industrial proximity metrics from Global Power Plant Database.
        
        Performs spatial search to find power plants within defined radius bands
        and aggregates metrics about nearest plants and capacity.
        
        Args:
            lat: Latitude of query location (-90 to 90)
            lon: Longitude of query location (-180 to 180)
            fetch_date: Optional date parameter (not used; included for API consistency)
            **kwargs: Additional keyword arguments (e.g., logger from decorator)
            
        Returns:
            Dictionary with proximity metrics or None if no data found
            
        Raises:
            No exceptions raised; errors are logged and None returned
        """
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None
            
            logger = kwargs.get('logger', self.logger)
            
            # Load plant database
            if not self._plants_loaded:
                self._load_plant_database(logger=logger)
            
            if not self._plants:
                logger.debug("No power plant database available")
                return None
            
            # Query industrial proximity
            result = self._query_proximity(lat, lon, logger=logger)
            
            if result is None:
                logger.debug(f"No power plants found near ({lat:.4f}, {lon:.4f})")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Global Power Plants query failed at ({lat:.4f}, {lon:.4f}): {str(e)}")
            return None

    def _load_plant_database(self, logger: Optional[Any] = None):
        """
        Load power plant database into memory.
        
        Attempts to load from cache first, then downloads if necessary.
        
        Args:
            logger: Logger instance
        """
        if logger is None:
            logger = self.logger
        
        self._plants_loaded = True
        self._plants = []
        
        try:
            cached_file = self.cache_dir / self.FALLBACK_CSV_NAME
            
            # Try cached version first
            if cached_file.exists():
                logger.debug(f"Loading power plants from cache: {cached_file}")
                self._plants = self._read_plant_csv(cached_file, logger)
                if self._plants:
                    logger.info(f"Loaded {len(self._plants)} power plants from cache")
                    return
            
            # Download fresh copy
            logger.debug("Downloading Global Power Plant Database...")
            response = self.session.get(
                self.DATA_URL,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Save to cache
                cached_file.write_bytes(response.content)
                logger.debug(f"Cached power plant database to {cached_file}")
                
                # Parse CSV
                self._plants = self._read_plant_csv(cached_file, logger)
                if self._plants:
                    logger.info(f"Downloaded and loaded {len(self._plants)} power plants")
            else:
                logger.warning(f"Download failed with HTTP {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Error loading power plant database: {e}")

    def _read_plant_csv(
        self,
        file_path: Path,
        logger: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Read power plant CSV file.
        
        Args:
            file_path: Path to CSV file
            logger: Logger instance
            
        Returns:
            List of plant dictionaries with lat/lon/capacity
        """
        if logger is None:
            logger = self.logger
        
        plants = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        plant = {
                            'lat': float(row.get('latitude', 0)),
                            'lon': float(row.get('longitude', 0)),
                            'capacity_mw': float(row.get('capacity_mw', 0)),
                            'fuel1': row.get('primary_fuel', 'Unknown'),
                            'name': row.get('name', 'Unknown'),
                        }
                        if plant['lat'] != 0 or plant['lon'] != 0:
                            plants.append(plant)
                    except (ValueError, KeyError):
                        continue
            
            logger.debug(f"Read {len(plants)} valid plants from CSV")
            return plants
            
        except Exception as e:
            logger.warning(f"CSV read error: {e}")
            return []

    def _query_proximity(
        self,
        lat: float,
        lon: float,
        logger: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate industrial proximity metrics.
        
        Args:
            lat: Latitude
            lon: Longitude
            logger: Logger instance
            
        Returns:
            Dictionary with proximity metrics or None
        """
        if logger is None:
            logger = self.logger
        
        try:
            if not self._plants:
                return None
            
            # Calculate distances to all plants
            distances = []
            for plant in self._plants:
                dist_km = self._haversine_distance(
                    lat, lon,
                    plant['lat'], plant['lon']
                )
                distances.append({
                    'distance_km': dist_km,
                    'capacity_mw': plant.get('capacity_mw', 0),
                    'fuel_type': plant.get('fuel1', 'Unknown'),
                })
            
            # Sort by distance
            distances.sort(key=lambda x: x['distance_km'])
            
            # Calculate metrics
            result = {}
            
            # Nearest plant distance
            if distances:
                result['nearest_plant_distance_km'] = round(distances[0]['distance_km'], 2)
            else:
                return None
            
            # Count plants by radius
            result['nearby_plant_count_10km'] = sum(
                1 for d in distances if d['distance_km'] <= 10
            )
            result['nearby_plant_count_25km'] = sum(
                1 for d in distances if d['distance_km'] <= 25
            )
            result['nearby_plant_count_50km'] = sum(
                1 for d in distances if d['distance_km'] <= 50
            )
            result['nearby_plant_count_100km'] = sum(
                1 for d in distances if d['distance_km'] <= 100
            )
            
            # Total capacity within 100 km
            capacity_100 = sum(
                d['capacity_mw'] for d in distances
                if d['distance_km'] <= 100
            )
            result['total_nearby_capacity_mw'] = round(capacity_100, 2)
            
            # Dominant fuel type in 100 km
            fuel_counts = {}
            for d in distances:
                if d['distance_km'] <= 100:
                    fuel = d['fuel_type']
                    fuel_counts[fuel] = fuel_counts.get(fuel, 0) + 1
            
            if fuel_counts:
                dominant_fuel = max(fuel_counts, key=fuel_counts.get)
                result['dominant_fuel_type'] = dominant_fuel
            else:
                result['dominant_fuel_type'] = 'Unknown'
            
            # Plant density (count per 1000 km²)
            # Approximate area: radius=100 km => area ≈ 31416 km²
            area_km2 = math.pi * (100 ** 2)
            density = (result['nearby_plant_count_100km'] / area_km2) * 1000
            result['plant_density_per_km2'] = round(density, 4)
            
            logger.debug(f"Proximity metrics: {result['nearby_plant_count_100km']} plants within 100 km")
            
            return result
            
        except Exception as e:
            logger.debug(f"Proximity calculation error: {e}")
            return None

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate great-circle distance between coordinates in kilometers.
        
        Args:
            lat1, lon1: First point
            lat2, lon2: Second point
            
        Returns:
            Distance in kilometers
        """
        try:
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            lon1_rad = math.radians(lon1)
            lon2_rad = math.radians(lon2)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
            c = 2 * math.asin(math.sqrt(a))
            
            # Earth radius in km
            earth_radius_km = 6371
            
            return earth_radius_km * c
            
        except Exception:
            return float('inf')
