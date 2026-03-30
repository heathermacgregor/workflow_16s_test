"""
Google Earth Engine ISDASOIL (Integrated Soil Database for the Analysis of Sub-Saharan Africa)

Provides access to 21 high-resolution soil geochemistry datasets covering Africa:
- Heavy metals: Al, Fe, Zn (extractable)
- Macro nutrients: Ca, Mg, K (extractable)
- Micro nutrients: P, S (extractable), N (total)
- Soil texture: clay, sand, silt content
- Soil properties: pH, CEC, organic carbon, bulk density, bedrock depth

DIRECTLY RELEVANT to metal_selection_pressure analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger(__name__)


class ISDASoilGeochemistryAPI:
    """
    Access ISDASOIL v1 datasets for African soil geochemistry.
    
    These datasets are CRUCIAL for metal enrichment proxy calculation:
    - Metal concentrations indicate cumulative weathering/enrichment
    - Soil texture predicts metal mobility and bioavailability
    - pH controls metal speciation and bioavailability
    - CEC indicates metal retention capacity
    """
    
    # All available ISDASOIL v1 assets
    ISDASOIL_ASSETS = {
        # Heavy metals (directly relevant to metal selection pressure)
        'aluminium_extractable': 'ISDASOIL/Africa/v1/aluminium_extractable',
        'iron_extractable': 'ISDASOIL/Africa/v1/iron_extractable',
        'zinc_extractable': 'ISDASOIL/Africa/v1/zinc_extractable',
        
        # Macro cations (metal competitors)
        'calcium_extractable': 'ISDASOIL/Africa/v1/calcium_extractable',
        'magnesium_extractable': 'ISDASOIL/Africa/v1/magnesium_extractable',
        'potassium_extractable': 'ISDASOIL/Africa/v1/potassium_extractable',
        
        # Micro nutrients
        'phosphorus_extractable': 'ISDASOIL/Africa/v1/phosphorus_extractable',
        'sulphur_extractable': 'ISDASOIL/Africa/v1/sulphur_extractable',
        'nitrogen_total': 'ISDASOIL/Africa/v1/nitrogen_total',
        
        # Soil texture (weathering proxy)
        'clay_content': 'ISDASOIL/Africa/v1/clay_content',
        'sand_content': 'ISDASOIL/Africa/v1/sand_content',
        'silt_content': 'ISDASOIL/Africa/v1/silt_content',
        'stone_content': 'ISDASOIL/Africa/v1/stone_content',
        
        # Soil chemistry (metal mobility control)
        'ph': 'ISDASOIL/Africa/v1/ph',
        'carbon_organic': 'ISDASOIL/Africa/v1/carbon_organic',
        'carbon_total': 'ISDASOIL/Africa/v1/carbon_total',
        'cation_exchange_capacity': 'ISDASOIL/Africa/v1/cation_exchange_capacity',
        
        # Bedrock & architecture
        'bedrock_depth': 'ISDASOIL/Africa/v1/bedrock_depth',
        'bulk_density': 'ISDASOIL/Africa/v1/bulk_density',
        'texture_class': 'ISDASOIL/Africa/v1/texture_class',
        
        # Multi-property composite
        'fcc': 'ISDASOIL/Africa/v1/fcc',  # Fraction of classes
    }
    
    # Depth layers (ISDASOIL typically has shallow, mid, deep predictions)
    DEPTH_LAYERS = ['mean_0_20', 'mean_20_50', 'mean_50_100']
    
    def __init__(self, authenticated: bool = False):
        """
        Initialize ISDASOIL API client.
        
        Args:
            authenticated: If True, requires GEE authentication (ee.Initialize())
        """
        self._authenticated = authenticated and ee is not None
        if self._authenticated:
            try:
                ee.Initialize()
            except Exception as e:
                logger.warning(f"GEE initialization failed: {e}")
                self._authenticated = False
    
    def query_by_point(
        self,
        latitude: float,
        longitude: float,
        properties: Optional[List[str]] = None,
        scale_m: int = 250
    ) -> Optional[Dict[str, float]]:
        """
        Query ISDASOIL data at a single point.
        
        Args:
            latitude: Sample latitude (-90 to 90)
            longitude: Sample longitude (-180 to 180)
            properties: List of ISDASOIL properties to retrieve
                       (default: all major metals + texture)
            scale_m: Pixel resolution in meters (250 is native)
            
        Returns:
            Dict mapping property names to values, or None if point outside Africa
        """
        if not self._authenticated or ee is None:
            return None
        
        if properties is None:
            properties = [
                'aluminium_extractable', 'iron_extractable', 'zinc_extractable',
                'clay_content', 'sand_content', 'ph',
                'cation_exchange_capacity', 'carbon_organic', 'bedrock_depth'
            ]
        
        try:
            # Check if point is in Africa (rough bounds: 37°W to 55°E, 35°S to 37°N)
            if longitude < -37 or longitude > 55 or latitude < -35 or latitude > 37:
                logger.debug(f"Point ({latitude}, {longitude}) outside ISDASOIL coverage (Africa)")
                return None
            
            point = ee.Geometry.Point([longitude, latitude])
            result = {}
            
            for prop in properties:
                if prop not in self.ISDASOIL_ASSETS:
                    logger.warning(f"Unknown ISDASOIL property: {prop}")
                    continue
                
                try:
                    asset_id = self.ISDASOIL_ASSETS[prop]
                    image = ee.Image(asset_id)
                    
                    # Sample at point
                    sample = image.sample(point, scale_m)
                    sample_data = sample.first().getInfo()
                    
                    # Extract mean values (ISDASOIL stores uncertainty + mean)
                    if sample_data and 'properties' in sample_data:
                        props = sample_data['properties']
                        # Usually has mean_0_20, mean_20_50, stdev_0_20, etc.
                        if 'mean_0_20' in props:
                            result[f'{prop}_0_20'] = props['mean_0_20']
                        if 'mean_20_50' in props:
                            result[f'{prop}_20_50'] = props['mean_20_50']
                        if 'mean' in props:
                            result[prop] = props['mean']
                except Exception as e:
                    logger.debug(f"Error sampling {prop}: {e}")
                    continue
            
            return result if result else None
            
        except Exception as e:
            logger.warning(f"GEE query failed for ({latitude}, {longitude}): {e}")
            return None
    
    def query_batch_points(
        self,
        coordinates: List[Tuple[float, float]],
        properties: Optional[List[str]] = None,
        scale_m: int = 250
    ) -> pd.DataFrame:
        """
        Query ISDASOIL for multiple points.
        
        Args:
            coordinates: List of (lat, lon) tuples
            properties: Properties to retrieve
            scale_m: Pixel resolution in meters
            
        Returns:
            DataFrame with coordinates and ISDASOIL properties
        """
        results = []
        
        for i, (lat, lon) in enumerate(coordinates):
            if i % 100 == 0:
                logger.debug(f"ISDASOIL: querying point {i}/{len(coordinates)}")
            
            try:
                data = self.query_by_point(lat, lon, properties, scale_m)
                if data:
                    results.append({
                        'lat': lat,
                        'lon': lon,
                        **data
                    })
            except Exception as e:
                logger.debug(f"Error at point {i}: {e}")
                continue
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def get_dataset_metadata() -> Dict[str, Dict]:
        """
        Return metadata for all ISDASOIL datasets.
        
        Returns:
            Dict mapping property names to metadata dicts
        """
        return {
            'aluminium_extractable': {
                'title': 'Al (extractable)',
                'unit': 'mg/kg',
                'relevance': 'Primary metal; weathering indicator',
                'depths': ['0-20cm', '20-50cm']
            },
            'iron_extractable': {
                'title': 'Fe (extractable)',
                'unit': 'mg/kg',
                'relevance': 'Primary metal; oxidation-reduction indicator',
                'depths': ['0-20cm', '20-50cm']
            },
            'zinc_extractable': {
                'title': 'Zn (extractable)',
                'unit': 'mg/kg',
                'relevance': 'Heavy metal; bioavailability control',
                'depths': ['0-20cm', '20-50cm']
            },
            'clay_content': {
                'title': 'Clay content',
                'unit': '%',
                'relevance': 'Weathering proxy; metal retention',
                'depths': ['0-20cm', '20-50cm', '50-100cm']
            },
            'ph': {
                'title': 'Soil pH',
                'unit': 'pH',
                'relevance': 'Solubility buffer; metal bioavailability',
                'depths': ['0-20cm', '20-50cm']
            },
            'cation_exchange_capacity': {
                'title': 'CEC',
                'unit': 'cmol+/kg',
                'relevance': 'Metal retention capacity',
                'depths': ['0-20cm', '20-50cm']
            },
            'carbon_organic': {
                'title': 'Organic carbon',
                'unit': '%',
                'relevance': 'Metal chelation; bioavailability',
                'depths': ['0-20cm', '20-50cm']
            },
            'bedrock_depth': {
                'title': 'Bedrock depth',
                'unit': 'cm',
                'relevance': 'Weathering depth; metal source indicator',
                'depths': ['0-200cm']
            },
        }


def get_isdasoil_client() -> Optional[ISDASoilGeochemistryAPI]:
    """Factory function to get ISDASOIL client with GEE authentication."""
    try:
        client = ISDASoilGeochemistryAPI(authenticated=True)
        return client if client._authenticated else None
    except Exception as e:
        logger.warning(f"Could not initialize ISDASOIL client: {e}")
        return None
