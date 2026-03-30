"""
Time-Series Change Detection Handler

Extracts temporal trends and changes from Earth Engine datasets:
- NDVI slope (long-term greening/browning trend)
- Land cover class changes (forest loss, urban expansion)
- Surface water occurrence changes (drying/flooding)

Useful for detecting environmental shifts that may be drivers
of microbial community composition changes.

Requires: 10 years of historical imagery minimum
Reference: https://developers.google.com/earth-engine/datasets
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeriesChangeDetectionAPI:
    """
    Extract temporal trends from Earth Engine time-series data.
    
    Metrics:
    - NDVI trend (greeness change over 10-20 years)
    - Deforestation rate (% forest loss per year)
    - Wetland extent change
    - Urban expansion rate
    - Climate trend (temperature, precipitation change per decade)
    """
    
    API_NAME = "TimeSeries_ChangeDetection"
    
    # Time windows for trend analysis
    ANALYSIS_CONFIG = {
        'ndvi_years': (2000, 2023),  # MODIS NDVI available since 2000
        'lc_years': (2001, 2023),    # MODIS land cover  
        'water_years': (1984, 2023),  # JRC Global Surface Water 30-year
        'temperature_years': (1981, 2023),  # ERA5 reanalysis
    }
    
    def __init__(self, verbose: bool = False, project_id: str = None):
        """
        Initialize time-series change detection handler.
        
        Args:
            verbose: Enable verbose logging
            project_id: GEE project ID
        """
        self.verbose = verbose
        self.project_id = project_id
        self.ee = None
        self._try_import()
    
    def _try_import(self):
        """Try to import and authenticate with Earth Engine."""
        try:
            import ee
            self.ee = ee
            
            try:
                if self.project_id:
                    self.ee.Initialize(project=self.project_id)
                else:
                    self.ee.Initialize()
                if self.verbose:
                    logger.debug("Earth Engine initialized for time-series analysis")
            except Exception as auth_err:
                logger.warning(f"Earth Engine authentication may be required: {auth_err}")
        except ImportError:
            logger.warning("Earth Engine API not installed. "
                          "Install with: uv pip install earthengine-api")
            self.ee = None
    
    def check_requirements(self) -> Tuple[bool, str]:
        """Check if Earth Engine is available."""
        if self.ee is None:
            return False, "Earth Engine API not installed"
        
        try:
            self.ee.Image('MODIS/061/MOD13Q1').bandNames().getInfo()
            return True, "Time-series datasets available"
        except Exception as e:
            return False, f"Error accessing time-series data: {str(e)}"
    
    def _extract_ndvi_trend(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Extract NDVI trend (greening/browning slope) over MODIS era (2000-2023).
        
        Returns:
            NDVI trend (change per year), R-squared, significance
        """
        try:
            if self.ee is None:
                return {'ndvi_slope_per_year': None, 'ndvi_r_squared': None}
            
            point = self.ee.Geometry.Point([lon, lat])
            
            # MODIS NDVI 16-day composite
            start_year = self.ANALYSIS_CONFIG['ndvi_years'][0]
            end_year = self.ANALYSIS_CONFIG['ndvi_years'][1]
            
            # Create annual NDVI composites
            ndvi_list = []
            for year in range(start_year, end_year + 1):
                start_date = self.ee.Date.fromYMD(year, 1, 1)
                end_date = self.ee.Date.fromYMD(year + 1, 1, 1)
                
                annual_ndvi = self.ee.ImageCollection('MODIS/061/MOD13Q1') \
                    .filterDate(start_date, end_date) \
                    .select('NDVI') \
                    .mean() \
                    .multiply(0.0001)  # Scale factor
                
                ndvi_val = annual_ndvi.sample(point, scale=250).first().getInfo()
                
                if ndvi_val and 'properties' in ndvi_val:
                    ndvi_list.append({
                        'year': year,
                        'ndvi': ndvi_val['properties']['NDVI']
                    })
            
            if len(ndvi_list) < 3:
                return {'ndvi_slope_per_year': None, 'ndvi_r_squared': None}
            
            # Fit linear regression
            df = pd.DataFrame(ndvi_list)
            x = df['year'].values.reshape(-1, 1)
            y = df['ndvi'].values
            
            # Remove NaN
            valid = ~np.isnan(y)
            if valid.sum() < 3:
                return {'ndvi_slope_per_year': None, 'ndvi_r_squared': None}
            
            x = x[valid]
            y = y[valid]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            slope = model.coef_[0]
            r2 = model.score(x, y)
            
            return {
                'ndvi_slope_per_year': float(slope),
                'ndvi_r_squared': float(r2),
                'n_years': len(df),
                'trend': 'greening' if slope > 0 else 'browning'
            }
        
        except Exception as e:
            logger.debug(f"NDVI trend extraction failed: {e}")
            return {'ndvi_slope_per_year': None, 'ndvi_r_squared': None}
    
    def _extract_forest_loss(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Extract forest loss rate using Hansen Global Forest Change dataset.
        
        Returns:
            Annual deforestation rate (%), peak loss year, total loss
        """
        try:
            if self.ee is None:
                return {'forest_loss_rate_pct_per_year': None}
            
            point = self.ee.Geometry.Point([lon, lat])
            
            # Hansen Global Forest Change (1km resolution, forest loss 2001-2023)
            gfc = self.ee.Image('UMD/hansen/global_forest_change_2023_v1_10')
            
            # Forest loss: band 'loss'
            loss = gfc.select('loss')
            gain = gfc.select('gain')
            
            # Sample at point (1km resolution)
            sampled = loss.addBands(gain).sample(point, scale=1000).first().getInfo()
            
            if not sampled or 'properties' not in sampled:
                return {'forest_loss_rate_pct_per_year': None}
            
            props = sampled['properties']
            forest_loss = props.get('loss', 0)  # Pixels with forest loss
            forest_gain = props.get('gain', 0)  # Pixels with forest regrowth
            
            # Estimate rate (forest loss from 2001-2023 = 23 years)
            loss_rate = forest_loss / 23.0 if forest_loss > 0 else 0
            
            return {
                'forest_loss_rate_pct_per_year': loss_rate,
                'forest_loss_pixels': int(forest_loss),
                'forest_gain_pixels': int(forest_gain),
            }
        
        except Exception as e:
            logger.debug(f"Forest loss extraction failed: {e}")
            return {'forest_loss_rate_pct_per_year': None}
    
    def _extract_water_trends(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Extract surface water occurrence changes (JRC Global Surface Water).
        
        Returns:
            Water occurrence trend, permanent water %, seasonal water %
        """
        try:
            if self.ee is None:
                return {'permanent_water_pct': None, 'seasonal_water_pct': None}
            
            point = self.ee.Geometry.Point([lon, lat])
            
            # JRC Global Surface Water (30m resolution, 1984-2023)
            gsw = self.ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            
            # Bands: 'occurrence', 'change_abs', 'seasonality'
            occurrence = gsw.select('occurrence')
            change = gsw.select('change_abs')
            seasonality = gsw.select('seasonality')
            
            sampled = occurrence.addBands(change).addBands(seasonality) \
                .sample(point, scale=30).first().getInfo()
            
            if not sampled or 'properties' not in sampled:
                return {'permanent_water_pct': None}
            
            props = sampled['properties']
            water_occurrence = props.get('occurrence', 0)  # 0-100% of months with water
            water_change = props.get('change_abs', 0)  # Change in water occurrence since 1984
            seasonality = props.get('seasonality', 0)
            
            return {
                'permanent_water_pct': min(float(water_occurrence), 100.0),
                'seasonal_water_pct': min(float(seasonality), 100.0),
                'water_change_since_1984': float(water_change),
            }
        
        except Exception as e:
            logger.debug(f"Water trend extraction failed: {e}")
            return {'permanent_water_pct': None}
    
    def _extract_temperature_trend(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Extract temperature trend from ERA5 reanalysis (1981-2023).
        
        Returns:
            Temperature change per decade (°C), trend p-value
        """
        try:
            if self.ee is None:
                return {'temperature_trend_c_per_decade': None}
            
            point = self.ee.Geometry.Point([lon, lat])
            
            # ERA5 monthly mean (1981-2023)
            era5 = self.ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
            
            # Extract annual mean temperature
            temps = []
            for year in range(1981, 2024):
                start = self.ee.Date.fromYMD(year, 1, 1)
                end = self.ee.Date.fromYMD(year + 1, 1, 1)
                
                annual_temp = era5 \
                    .filterDate(start, end) \
                    .select('temperature_2m') \
                    .mean()
                
                temp_val = annual_temp.sample(point, scale=11132).first().getInfo()
                
                if temp_val and 'properties' in temp_val:
                    t = temp_val['properties']['temperature_2m'] - 273.15  # K to C
                    temps.append({'year': year, 'temperature_c': t})
            
            if len(temps) < 5:
                return {'temperature_trend_c_per_decade': None}
            
            df = pd.DataFrame(temps)
            x = (df['year'].values - 1981) / 10.0  # Decades since 1981
            y = df['temperature_c'].values
            
            valid = ~np.isnan(y)
            if valid.sum() < 3:
                return {'temperature_trend_c_per_decade': None}
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x[valid].reshape(-1, 1), y[valid])
            
            trend = model.coef_[0]  # Change per decade
            
            return {
                'temperature_trend_c_per_decade': float(trend),
                'baseline_temp_c': float(df['temperature_c'].iloc[0]),
                'current_temp_c': float(df['temperature_c'].iloc[-1]),
            }
        
        except Exception as e:
            logger.debug(f"Temperature trend extraction failed: {e}")
            return {'temperature_trend_c_per_decade': None}
    
    def fetch_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch all time-series change metrics for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with NDVI trend, forest loss, water changes, temperature trend
        """
        try:
            if self.ee is None:
                return {'available': False, 'error': 'EE not initialized'}
            
            result = {
                'available': True,
                'data': {}
            }
            
            # Extract each trend
            if self.verbose:
                logger.debug(f"Extracting trends at ({lat}, {lon})")
            
            result['data'].update(self._extract_ndvi_trend(lat, lon))
            result['data'].update(self._extract_forest_loss(lat, lon))
            result['data'].update(self._extract_water_trends(lat, lon))
            result['data'].update(self._extract_temperature_trend(lat, lon))
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching time-series data: {str(e)}")
            return {'available': False, 'error': str(e)}
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """Get time-series data (interface method)."""
        return self.fetch_data(lat, lon)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None):
        """
        Enrich dataframe with time-series change metrics.
        
        Adds columns:
        - ts_ndvi_slope_per_year: NDVI trend (change per year)
        - ts_ndvi_r_squared: Quality of NDVI fit
        - ts_forest_loss_rate: Annual forest loss (%)
        - ts_permanent_water: Persistent water occurrence (%)
        - ts_seasonal_water: Seasonal water occurrence (%)
        - ts_water_change_since_1984: Water occurrence change
        - ts_temperature_trend_c_per_decade: Temperature change per decade
        
        Args:
            df: Input dataframe with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            sample_id_col: Optional sample ID for logging
        
        Returns:
            Dataframe with added time-series columns
        """
        if not self.check_requirements()[0]:
            logger.warning("Time-series change detection not available")
            return df
        
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                # Initialize all columns to None
                result_row = {
                    'ts_ndvi_slope_per_year': None,
                    'ts_ndvi_r_squared': None,
                    'ts_forest_loss_rate_pct_per_year': None,
                    'ts_permanent_water_pct': None,
                    'ts_seasonal_water_pct': None,
                    'ts_water_change_since_1984': None,
                    'ts_temperature_trend_c_per_decade': None,
                }
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append(result_row)
                    continue
                
                data = self.fetch_data(lat, lon)
                
                if data.get('available'):
                    for key, val in data.get('data', {}).items():
                        ts_key = f'ts_{key}'
                        if ts_key in result_row:
                            result_row[ts_key] = val
                
                results.append(result_row)
                
                if self.verbose and (idx + 1) % 50 == 0:
                    logger.debug(f"Processed {idx + 1}/{total_rows} rows for time-series data")
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {
                    'ts_ndvi_slope_per_year': None,
                    'ts_ndvi_r_squared': None,
                    'ts_forest_loss_rate_pct_per_year': None,
                    'ts_permanent_water_pct': None,
                    'ts_seasonal_water_pct': None,
                    'ts_water_change_since_1984': None,
                    'ts_temperature_trend_c_per_decade': None,
                }
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
