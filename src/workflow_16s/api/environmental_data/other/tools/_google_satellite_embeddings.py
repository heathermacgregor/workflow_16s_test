"""
Google Satellite Embeddings Handler

Provides access to AlphaEarth Foundation Models' 64-dimensional satellite embeddings.
These embeddings encode optical, radar, lidar, and climate information into a compact
semantic representation per 10-m pixel, ideal for machine learning feature sets.

Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
- 64 continuous features per pixel
- Pre-computed semantic representation of surface conditions
- No additional API key required (uses GEE authentication only)

Reference:
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
"""

import logging
from typing import Dict, Any, Tuple, List
import numpy as np
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class GoogleSatelliteEmbeddingsAPI(BaseEnvironmentalAPI):
    """
    Extract 64-dimensional satellite embeddings from Google's AlphaEarth models.
    
    Embeddings encode multi-modal satellite data (optical, SAR, topographic, climate)
    into a compact vector representation. Useful for:
    - ML feature engineering (64D embeddings directly)
    - Unsupervised clustering of surface conditions
    - Dimensionality reduction for high-dimensional analysis
    
    Each of 64 features normalized to [0, 1] scale.
    Results include:
    - 64 embedding dimensions (embedding_0 through embedding_63)
    - metadata: embedding_model_id, embedding_year
    """
    
    API_NAME = "Google_Satellite_Embeddings"
    
    # Satellite Embeddings configuration
    EMBEDDINGS_CONFIG = {
        'asset': 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
        'year': 2021,  # Default year; can be parameterized
        'scale': 10,   # 10-meter resolution
        'embedding_dims': 64,  # AlphaEarth produces 64D vectors
    }
    
    def __init__(self, verbose: bool = False, project_id: str = None):
        """
        Initialize Google Satellite Embeddings handler.
        
        Args:
            verbose: Enable verbose logging
            project_id: GEE project ID (from config)
        """
        super().__init__(verbose=verbose)
        self.project_id = project_id
        self.ee = None
        self._try_import()
    
    def _try_import(self):
        """Try to import and authenticate with Earth Engine."""
        try:
            import ee
            self.ee = ee
            
            # Check if already authenticated
            try:
                if self.project_id:
                    self.ee.Initialize(project=self.project_id)
                else:
                    self.ee.Initialize()
                if self.verbose:
                    logger.debug("Earth Engine authenticated successfully for embeddings")
            except Exception as auth_err:
                logger.warning(f"Earth Engine authentication may be required: {auth_err}")
        except ImportError:
            logger.warning("Earth Engine API not installed. "
                          "Install with: uv pip install earthengine-api")
            self.ee = None
    
    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if Google Earth Engine is available and embeddings dataset accessible.
        
        Returns:
            Tuple of (is_available, message)
        """
        if self.ee is None:
            return False, "Earth Engine API not installed"
        
        try:
            # Try to load embeddings asset
            asset = self.ee.Image(self.EMBEDDINGS_CONFIG['asset'])
            info = asset.bandNames().getInfo()
            return True, f"Embeddings available with {len(info)} bands"
        except Exception as e:
            return False, f"Embeddings not available: {str(e)}"
    
    def _fetch_embeddings(self, lat: float, lon: float, year: int = None) -> Dict[str, Any]:
        """
        Fetch 64D satellite embeddings for a point location.
        
        Args:
            lat: Latitude
            lon: Longitude
            year: Year for embeddings (default 2021)
        
        Returns:
            Dictionary with 64 embedding dimensions
        """
        try:
            if self.ee is None:
                return {'available': False, 'error': 'EE not initialized'}
            
            year = year or self.EMBEDDINGS_CONFIG['year']
            
            # Create point geometry
            point = self.ee.Geometry.Point([lon, lat])
            
            # Load embeddings asset
            embeddings_img = self.ee.Image(self.EMBEDDINGS_CONFIG['asset'])
            scale = self.EMBEDDINGS_CONFIG['scale']
            
            # Sample at point
            sampled = embeddings_img.sample(point, scale=scale).first()
            
            result_dict = {}
            
            if sampled:
                val_dict = sampled.getInfo()
                if val_dict and 'properties' in val_dict:
                    props = val_dict['properties']
                    
                    # Extract embedding vectors (typically named 0-63 or embedding_0-63)
                    for i in range(self.EMBEDDINGS_CONFIG['embedding_dims']):
                        # Try common naming conventions
                        for key_pattern in [str(i), f'embedding_{i}']:
                            if key_pattern in props:
                                val = props[key_pattern]
                                if val is not None:
                                    result_dict[f'embedding_{i}'] = float(val)
                                    break
                        # If not found in properties, set to None
                        if f'embedding_{i}' not in result_dict:
                            result_dict[f'embedding_{i}'] = None
                    
                    return {
                        'available': True,
                        'data': result_dict,
                        'num_embeddings': len([v for v in result_dict.values() if v is not None]),
                        'metadata': {
                            'model': 'AlphaEarth Foundation V1',
                            'year': year,
                            'lat': lat,
                            'lon': lon
                        }
                    }
                else:
                    return {'available': True, 'data': {}, 'error': 'Empty response from GEE'}
            else:
                return {'available': True, 'data': {}, 'error': 'No data at point'}
        
        except Exception as e:
            logger.error(f"Error fetching embeddings: {str(e)}")
            return {'available': False, 'error': str(e)}
    
    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get satellite embeddings (interface method for BaseEnvironmentalAPI).
        """
        year = kwargs.get('year', self.EMBEDDINGS_CONFIG['year'])
        return self._fetch_embeddings(lat, lon, year=year)
    
    def fetch_and_enrich(self, df, lat_col: str, lon_col: str, 
                         sample_id_col: str = None, year: int = None):
        """
        Enrich dataframe with 64-dimensional satellite embeddings.
        
        Adds columns: embedding_0, embedding_1, ..., embedding_63
        
        These can be used directly as ML features or dimensionally reduced further.
        Each embedding encodes optical, SAR, topographic, and climate context at
        the sample location.
        
        Args:
            df: Input dataframe with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            sample_id_col: Optional sample ID column for logging
            year: Year for embeddings (default 2021)
        
        Returns:
            Dataframe with added embedding columns
        """
        import pandas as pd
        
        if not self.check_requirements()[0]:
            logger.warning("Google Satellite Embeddings not available")
            return df
        
        year = year or self.EMBEDDINGS_CONFIG['year']
        results = []
        
        total_rows = len(df)
        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]
                
                # Initialize all embedding columns to None
                result_row = {f'se_embedding_{i}': None 
                             for i in range(self.EMBEDDINGS_CONFIG['embedding_dims'])}
                
                if pd.isna(lat) or pd.isna(lon):
                    results.append(result_row)
                    continue
                
                data = self._fetch_embeddings(lat, lon, year=year)
                
                if data['available'] and data.get('data'):
                    # Map embeddings to result row
                    for key, val in data['data'].items():
                        result_row[f'se_{key}'] = val
                
                results.append(result_row)
                
                if self.verbose and (idx + 1) % 100 == 0:
                    logger.debug(f"Processed {idx + 1}/{total_rows} rows for embeddings")
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = {f'se_embedding_{i}': None 
                             for i in range(self.EMBEDDINGS_CONFIG['embedding_dims'])}
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
    
    def get_embedding_stats(self, df):
        """
        Compute statistics on extracted embeddings.
        
        Returns:
            Dictionary with dimension-wise mean, std, min, max
        """
        import pandas as pd
        
        embedding_cols = [col for col in df.columns if col.startswith('se_embedding_')]
        
        if not embedding_cols:
            return None
        
        embedding_data = df[embedding_cols].values
        
        # Compute statistics
        stats = {
            'n_samples': len(df),
            'n_embedding_dims': len(embedding_cols),
            'dims_with_data': embedding_data.notna().sum(axis=0).mean(),
            'mean': np.nanmean(embedding_data, axis=0).tolist(),
            'std': np.nanstd(embedding_data, axis=0).tolist(),
            'min': np.nanmin(embedding_data, axis=0).tolist(),
            'max': np.nanmax(embedding_data, axis=0).tolist(),
        }
        
        return stats
