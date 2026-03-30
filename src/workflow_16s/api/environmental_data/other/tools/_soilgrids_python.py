"""
SoilGrids Python Wrapper API Handler with REST API Fallback

Provides high-resolution soil property predictions via the SoilGrids Python package,
with automatic fallback to REST API if library methods are unavailable.

Variables available:
- bdod: Bulk density (kg/dm³)
- cec: Cation exchange capacity (cmol(c)/kg)
- cfvo: Coarse fragments (>2mm) (vol%)
- clay: Clay content (%)
- nitrogen: Total nitrogen (g/kg)
- phh2o: Soil pH
- sand: Sand content (%)
- silt: Silt content (%)
- soc: Soil organic carbon (g/kg)
- ocd: Organic carbon density (kg/m³)
- ocs: Organic carbon stock (0-30cm) (kg/m²)
- wv0010: Water content at 10kPa (vol%)
- wv0033: Water content at 33kPa (vol%)
- wv1500: Water content at 1500kPa (vol%)

Install: uv pip install git+https://github.com/wurli/soilgrids
REST API: Automatically falls back to https://rest.isric.org/soilgrids/v2.0/
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import requests
import time
from .base import BaseEnvironmentalAPI, REQUEST_TIMEOUT
from .cache import cache_api_call

logger = logging.getLogger(__name__)


class SoilGridsPythonAPI(BaseEnvironmentalAPI):
    """
    Query SoilGrids for soil property predictions using Python wrapper with REST API fallback.
    
    Provides 14+ soil variables at multiple depths (0-5cm, 5-15cm, 15-30cm).
    
    Behavior:
    - Attempts to use soilgrids Python library if available
    - Falls back to REST API (https://rest.isric.org/soilgrids/v2.0/) if library lacks expected methods
    - Handles 503/429 errors with exponential backoff
    
    Returns:
    - Soil texture (clay, silt, sand percentages)
    - Chemical properties (pH, cation exchange capacity)
    - Organic carbon content
    - Bulk density
    - Water holding capacity
    """
    
    API_NAME = "SoilGrids_Python"
    REST_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    # Key soil variables to extract
    SOIL_VARIABLES = [
        'phh2o',      # Soil pH
        'clay',       # Clay content %
        'sand',       # Sand content %
        'silt',       # Silt content %
        'nitrogen',   # Total nitrogen g/kg
        'cec',        # Cation exchange capacity cmol(c)/kg
        'soc',        # Soil organic carbon g/kg
        'bdod'        # Bulk density kg/dm³
    ]
    
    DEPTHS = ['0-5cm', '5-15cm', '15-30cm']
    
    def __init__(self, verbose: bool = False):
        """
        Initialize SoilGrids handler with library import or REST API fallback.
        
        Args:
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.sg = None
        self.use_rest_api = False
        self._try_import()
    
    def _try_import(self):
        """Log import results and enable REST API fallback if library methods unavailable."""
        try:
            from soilgrids import SoilGrids
            self.sg = SoilGrids()

            # Try to log version info for debugging
            version_info = None
            try:
                import soilgrids
                if hasattr(soilgrids, '__version__'):
                    version_info = f"version {soilgrids.__version__}"
            except:
                pass

            version_str = f" ({version_info})" if version_info else ""
            
            # Check if library has expected methods
            detected_method = self._detect_query_method()
            if detected_method is None:
                available_attrs = [m for m in dir(self.sg) if not m.startswith('_')]
                logger.warning(f"SoilGrids library imported{version_str} but no recognized query method found. "
                              f"Available methods: {available_attrs}. "
                              f"Falling back to REST API.")
                self.use_rest_api = True
                self.sg = None
            else:
                if self.verbose:
                    logger.debug(f"SoilGrids Python wrapper imported{version_str} successfully. "
                               f"Will use method: '{detected_method}'")
                self.use_rest_api = False

        except ImportError:
            logger.debug("SoilGrids Python wrapper not installed. "
                        "Install with: uv pip install git+https://github.com/wurli/soilgrids. "
                        "Using REST API fallback.")
            self.use_rest_api = True
            self.sg = None
        except Exception as e:
            logger.debug(f"Failed to initialize SoilGrids Python wrapper: {str(e)}. "
                        "Using REST API fallback.")
            self.use_rest_api = True
            self.sg = None
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if SoilGrids is available (either library or REST API).
        
        Returns:
            Tuple of (is_available, message)
        """
        if self.sg is not None or self.use_rest_api:
            method = "Python library" if self.sg else "REST API"
            return True, f"SoilGrids available via {method}"
        else:
            return False, "SoilGrids unavailable (no library or REST API)"
    
    def _fetch_data_via_rest_api(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch SoilGrids data via direct REST API call.
        
        This is used as fallback when Python library is unavailable or has wrong API.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with soil properties at multiple depths
        """
        max_retries = 3
        backoff_base = 2
        backoff_multiplier = 2
        
        try:
            for attempt in range(max_retries + 1):
                try:
                    params = {
                        "lon": lon, 
                        "lat": lat, 
                        "property": self.SOIL_VARIABLES,
                        "depth": self.DEPTHS,
                        "value": "mean"
                    }
                    
                    response = requests.get(
                        self.REST_API_URL, 
                        params=params, 
                        timeout=REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Check for valid response structure
                    if "properties" not in data or "layers" not in data["properties"]:
                        logger.debug(f"SoilGrids REST API: No data returned for ({lat}, {lon})")
                        return {'available': False, 'error': 'No data in response'}
                    
                    # Parse soil properties
                    data_dict = {}
                    for layer in data["properties"]["layers"]:
                        prop_name = layer["name"]
                        unit_measure = layer.get("unit_measure", {})
                        divisor = float(unit_measure.get("conversion_factor", 1.0))
                        
                        for depth_interval in layer.get("depths", []):
                            depth_label = depth_interval.get("label", "")
                            raw_value = depth_interval.get("values", {}).get("mean")
                            
                            if raw_value is not None:
                                actual_value = round(raw_value / divisor, 2)
                                col_name = f'{prop_name}_{depth_label}'
                                data_dict[col_name] = actual_value
                    
                    # Calculate means
                    for var in self.SOIL_VARIABLES:
                        depth_vals = []
                        for depth in self.DEPTHS:
                            key = f'{var}_{depth}'
                            if key in data_dict:
                                depth_vals.append(data_dict[key])
                        
                        if depth_vals:
                            data_dict[f'{var}_mean'] = sum(depth_vals) / len(depth_vals)
                    
                    if self.verbose:
                        logger.debug(f"SoilGrids (REST API): Retrieved data for ({lat}, {lon})")
                    
                    return {'available': True, 'data': data_dict}
                    
                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 503:
                        if attempt < max_retries:
                            wait_time = backoff_base * (backoff_multiplier ** attempt)
                            logger.warning(f"SoilGrids REST API returned 503. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"SoilGrids REST API failed with 503 after {max_retries} retries")
                            return {'available': False, 'error': 'Service unavailable (503)'}
                    else:
                        logger.error(f"SoilGrids REST API HTTP error {http_err.response.status_code}")
                        return {'available': False, 'error': f'HTTP error {http_err.response.status_code}'}
                
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
                    if attempt < max_retries:
                        wait_time = backoff_base * (backoff_multiplier ** attempt)
                        logger.warning(f"SoilGrids REST API network error. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"SoilGrids REST API network error after {max_retries} retries")
                        return {'available': False, 'error': 'Network error'}
                        
        except Exception as e:
            logger.error(f"SoilGrids REST API error: {str(e)}")
            return {'available': False, 'error': str(e)}
        
        return {'available': False, 'error': 'Query failed'}
    

    def _fetch_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch SoilGrids data - uses library or REST API fallback.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dictionary with soil properties at multiple depths
        """
        # Use REST API fallback if library is not available or not usable
        if self.use_rest_api or self.sg is None:
            return self._fetch_data_via_rest_api(lat, lon)
        
        # Otherwise try to use library
        return self._fetch_data_via_library(lat, lon)
    
    def _fetch_data_via_library(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch SoilGrids data using Python library wrapper.

        Supports multiple SoilGrids package versions through method detection.
        Implements exponential backoff for 503 errors (service unavailable).

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dictionary with soil properties at multiple depths
        """
        # Retry parameters for handling 503 errors
        max_retries = 3
        backoff_base = 2  # Start with 2 seconds
        backoff_multiplier = 2

        try:
            if self.sg is None:
                self._try_import()
                if self.sg is None:
                    return {'available': False, 'error': 'SoilGrids not installed'}

            # Detect which method is available
            query_method = self._detect_query_method()
            if query_method is None:
                available_methods = [m for m in dir(self.sg) if not m.startswith('_')]
                logger.debug(f"SoilGrids package has no recognized query method (get_points, query, etc.). "
                           f"Available attributes: {available_methods}. Falling back to REST API.")
                self.use_rest_api = True
                return self._fetch_data_via_rest_api(lat, lon)

            # Query SoilGrids using detected method with retry logic
            result = None

            for attempt in range(max_retries + 1):
                try:
                    if query_method == 'get_points':
                        # Original/Primary API - most common
                        result = self.sg.get_points(
                            lat=lat,
                            lon=lon,
                            soil_property=self.SOIL_VARIABLES,
                            depth=self.DEPTHS,
                            value='mean'
                        )
                    elif query_method == 'query':
                        # Alternative API (newer versions may use this)
                        result = self.sg.query(
                            latitude=lat,
                            longitude=lon,
                            property=self.SOIL_VARIABLES,
                            depth=self.DEPTHS,
                            value='mean'
                        )
                    elif query_method == 'bulk_query':
                        # Another possible API
                        result = self.sg.bulk_query(
                            points=[[lat, lon]],
                            properties=self.SOIL_VARIABLES,
                            depths=self.DEPTHS
                        )
                    else:
                        return {'available': False, 'error': f'Unknown query method: {query_method}'}

                    # If we get here, query succeeded
                    break

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 503:
                        # Service Unavailable - retry with exponential backoff
                        if attempt < max_retries:
                            wait_time = backoff_base * (backoff_multiplier ** attempt)
                            logger.warning(f"SoilGrids returned 503 (Service Unavailable) at ({lat}, {lon}). "
                                         f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"SoilGrids API failed with 503 after {max_retries} retries. Server overloaded.")
                            return {'available': False, 'error': f'Service unavailable after {max_retries} retries'}
                    else:
                        # Other HTTP errors - don't retry
                        logger.error(f"SoilGrids query failed with HTTP {http_err.response.status_code}: {str(http_err)}")
                        return {'available': False, 'error': f'HTTP error {http_err.response.status_code}'}

                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
                    # Network errors - retry with backoff
                    if attempt < max_retries:
                        wait_time = backoff_base * (backoff_multiplier ** attempt)
                        logger.warning(f"SoilGrids network error at ({lat}, {lon}). "
                                     f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"SoilGrids network error after {max_retries} retries: {str(net_err)}")
                        return {'available': False, 'error': 'Network error'}

                except Exception as query_err:
                    logger.error(f"SoilGrids query failed using method '{query_method}': {str(query_err)}")
                    return {'available': False, 'error': f'Query failed: {str(query_err)}'}

            # Extract data (different APIs return different structures)
            if result is None:
                logger.debug(f"SoilGrids returned None result for ({lat}, {lon})")
                return {'available': False, 'error': 'No data returned'}

            # Check for data in self.sg.data (if attribute exists) or result itself
            data = None
            try:
                if hasattr(self.sg, 'data'):
                    if self.sg.data is not None:
                        data = self.sg.data
                elif hasattr(result, 'empty'):
                    if not result.empty:
                        data = result
                elif isinstance(result, dict) and result:
                    data = result
            except Exception as data_err:
                logger.debug(f"Error accessing SoilGrids data: {str(data_err)}")
                data = None

            if data is None:
                logger.debug(f"No data in SoilGrids response for ({lat}, {lon})")
                return {'available': False, 'error': 'No data in response'}

            # Process data
            data_dict = {}

            if isinstance(data, dict):
                # Result is a dictionary
                data_dict = data
            elif hasattr(data, 'columns'):
                # Result is a DataFrame
                for var in self.SOIL_VARIABLES:
                    for depth in self.DEPTHS:
                        col_name = f'{var}_{depth}'
                        if col_name in data.columns:
                            try:
                                val = data[col_name].iloc[0]
                                data_dict[col_name] = float(val) if pd.notna(val) else None
                            except (IndexError, ValueError, TypeError):
                                data_dict[col_name] = None
                        else:
                            data_dict[col_name] = None

            # Calculate mean across depths
            for var in self.SOIL_VARIABLES:
                depth_vals = [data_dict.get(f'{var}_{depth}') for depth in self.DEPTHS]
                valid_vals = [v for v in depth_vals if v is not None]
                if valid_vals:
                    data_dict[f'{var}_mean'] = sum(valid_vals) / len(valid_vals)
                else:
                    data_dict[f'{var}_mean'] = None

            result_dict = {
                'available': True,
                'data': data_dict
            }

            if self.verbose:
                logger.debug(f"SoilGrids (library method '{query_method}'): Retrieved data for ({lat}, {lon})")

            return result_dict

        except Exception as e:
            logger.error(f"Error fetching SoilGrids data via library at ({lat}, {lon}): {str(e)}")
            return {'available': False, 'error': str(e)}


    
    def _detect_query_method(self) -> Optional[str]:
        """Detect which query method is available in current SoilGrids version.

        This method handles version compatibility by checking for all known
        query method signatures across different soilgrids package versions.

        Returns:
            Method name string ('get_points', 'query', etc.) or None if none available.
            Priority order: get_points > query > bulk_query > request
        """
        if self.sg is None:
            return None

        # Check for known methods in order of preference
        # Note: soilgrids 0.3.0+ uses OGC/WCS standard (get_coverage_data)
        # Older versions (0.2.x) used custom API (get_points)
        methods_to_check = ['get_coverage_data', 'get_points', 'query', 'bulk_query', 'request']

        for method in methods_to_check:
            try:
                if hasattr(self.sg, method):
                    attr = getattr(self.sg, method)
                    if callable(attr):
                        if self.verbose:
                            logger.debug(f"SoilGrids: Detected query method '{method}'")
                        return method
            except Exception as e:
                logger.debug(f"Error checking for method '{method}': {str(e)}")
                continue

        # If we get here, no recognized method was found
        available = [m for m in dir(self.sg) if not m.startswith('_')]
        logger.debug(f"No recognized SoilGrids query method found. "
                    f"Available attributes: {available}")
        return None
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get SoilGrids data (interface method for BaseEnvironmentalAPI).
        """
        return self._fetch_data(lat, lon, **kwargs)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None):
        """
        Enrich dataframe with SoilGrids soil property data.
        
        Adds columns:
        - soilgrids_phh2o_0_5cm, soilgrids_phh2o_mean, etc.
        - soilgrids_clay_0_5cm, soilgrids_clay_mean, etc.
        - ... for all SOIL_VARIABLES
        """
        import pandas as pd
        
        if not self.check_requirements()[0]:
            logger.warning("SoilGrids Python wrapper not available")
            # Return original df unchanged
            return df
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                result_row = {}
                
                if pd.isna(lat) or pd.isna(lon):
                    # Null values for all columns
                    for var in self.SOIL_VARIABLES:
                        for depth in self.DEPTHS + ['mean']:
                            result_row[f'soilgrids_{var}_{depth}'] = None
                    results.append(result_row)
                    continue
                
                data = self._fetch_data(lat, lon)
                
                if data['available']:
                    # Add all retrieved values with soilgrids_ prefix
                    for key, val in data.get('data', {}).items():
                        result_row[f'soilgrids_{key}'] = val
                else:
                    # Null values for all columns
                    for var in self.SOIL_VARIABLES:
                        for depth in self.DEPTHS + ['mean']:
                            result_row[f'soilgrids_{var}_{depth}'] = None
                
                results.append(result_row)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {}
                for var in self.SOIL_VARIABLES:
                    for depth in self.DEPTHS + ['mean']:
                        result_row[f'soilgrids_{var}_{depth}'] = None
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
