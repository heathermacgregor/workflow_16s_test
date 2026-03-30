"""
SoilGrids Python Wrapper API Handler

Provides high-resolution soil property predictions via the SoilGrids Python package.

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
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class SoilGridsPythonAPI(BaseEnvironmentalAPI):
    """
    Query SoilGrids for soil property predictions using Python wrapper.
    
    Provides 14+ soil variables at multiple depths (0-5cm, 5-15cm, 15-30cm).
    
    Returns:
    - Soil texture (clay, silt, sand percentages)
    - Chemical properties (pH, cation exchange capacity)
    - Organic carbon content
    - Bulk density
    - Water holding capacity
    """
    
    API_NAME = "SoilGrids_Python"
    
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
        Initialize SoilGrids Python wrapper handler.
        
        Args:
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.sg = None
        self._try_import()
    
    def _try_import(self):
        """Try to import soilgrids library and log version info."""
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
            if self.verbose:
                logger.debug(f"SoilGrids Python wrapper imported successfully{version_str}")
        except ImportError:
            logger.warning("SoilGrids Python wrapper not installed. "
                          "Install with: uv pip install git+https://github.com/wurli/soilgrids")
            self.sg = None
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if SoilGrids Python wrapper is available.
        
        Returns:
            Tuple of (is_available, message)
        """
        if self.sg is not None:
            return True, "SoilGrids Python wrapper available"
        else:
            return False, "SoilGrids Python wrapper not installed"
    
    def _fetch_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch SoilGrids data for a point location.
        
        Supports multiple SoilGrids package versions through method detection.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with soil properties at multiple depths
        """
        try:
            if self.sg is None:
                self._try_import()
                if self.sg is None:
                    return {'available': False, 'error': 'SoilGrids not installed'}
            
            # Detect which method is available
            query_method = self._detect_query_method()
            if query_method is None:
                logger.error("SoilGrids package has no recognized query method (get_points, query, etc.)")
                return {'available': False, 'error': 'No query method available'}
            
            # Query SoilGrids using detected method
            try:
                if query_method == 'get_points':
                    # Original API
                    result = self.sg.get_points(
                        [lat],
                        [lon],
                        soil_property=self.SOIL_VARIABLES,
                        depth=self.DEPTHS,
                        value='mean'
                    )
                elif query_method == 'query':
                    # Alternative API (newer versions may use this)
                    result = self.sg.query(
                        latitude=[lat],
                        longitude=[lon],
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
            
            except Exception as ae:
                logger.error(f"SoilGrids query failed using method '{query_method}': {str(ae)}")
                return {'available': False, 'error': f'Query failed: {str(ae)}'}
            
            # Extract data (different APIs return different structures)
            if result is None:
                logger.debug("SoilGrids returned None result")
                return {'available': False, 'error': 'No data returned'}
            
            # Check for data in self.sg.data (if attribute exists)
            data = None
            if hasattr(self.sg, 'data') and self.sg.data is not None:
                data = self.sg.data
            elif hasattr(result, 'empty') and not result.empty:
                data = result
            elif isinstance(result, dict) and result:
                data = result
            else:
                logger.debug("No data in SoilGrids response")
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
                            val = data[col_name].iloc[0]
                            data_dict[col_name] = float(val) if pd.notna(val) else None
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
                logger.debug(f"SoilGrids ({query_method}): Retrieved data for ({lat}, {lon})")
            
            return result_dict
        
        except Exception as e:
            logger.error(f"Error fetching SoilGrids data: {str(e)}")
            return {'available': False, 'error': str(e)}
    
    def _detect_query_method(self) -> Optional[str]:
        """Detect which query method is available in current SoilGrids version.
        
        Returns:
            Method name string ('get_points', 'query', etc.) or None if none available
        """
        # Check for known methods in order of preference
        methods_to_check = ['get_points', 'query', 'bulk_query', 'request']
        
        for method in methods_to_check:
            if hasattr(self.sg, method) and callable(getattr(self.sg, method)):
                if self.verbose:
                    logger.debug(f"SoilGrids: Detected query method '{method}'")
                return method
        
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
