"""
Band-Stacking (Mega-Image) Builder for Google Earth Engine

Dramatically reduces API calls by:
1. Stacking all enabled GEE datasets into a single composite image
2. Sampling ALL bands at once (1 API call per point vs N calls)
3. Organizing band metadata for data dictionary generation

Expected improvement: 7 datasets × 5 bands = 35 calls → 1 call per point

Architecture:
- MegaImageBuilder: Orchestrates mega-image construction and sampling
- Band catalog: Maps GEE asset names to actual image collections
- Metadata tracking: Stores provenance and data types for each band
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger(__name__)


# ============================================================================
# BAND CATALOG: Maps dataset names to GEE asset IDs and band information
# ============================================================================

GEE_ASSET_CATALOG = {
    'jrc_global_water': {
        'asset_id': 'JRC/GSW1_4/GlobalSurfaceWater',
        'image_type': 'single_image',
        'bands': {
            'occurrence': {'description': 'Water occurrence (0-100%)', 'dtype': 'uint8'},
            'seasonality': {'description': 'Seasonality month (1-12)', 'dtype': 'uint8'},
            'recurrence': {'description': 'Recurrence (0-100%)', 'dtype': 'uint8'},
            'transition': {'description': 'Water transition (0-100%)', 'dtype': 'uint8'},
            'max_extent': {'description': 'Maximum extent (0-100%)', 'dtype': 'uint8'},
        },
        'scale_m': 30,
        'description': 'JRC Global Surface Water dataset'
    },

    'viirs_nighttime_lights': {
        'asset_id': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG',
        'image_type': 'image_collection_median',
        'date_range_days': 365,
        'bands': {
            'avg_rad': {'description': 'Average radiance (nanoW/cm²/sr)', 'dtype': 'float32'},
            'cf_cvg': {'description': 'Cloud-free coverage (%)', 'dtype': 'uint8'},
        },
        'scale_m': 463,
        'description': 'VIIRS Nighttime Lights (monthly composite)'
    },

    'hansen_global_forest_change': {
        'asset_id': 'UMD/hansen/global_forest_change_2023_v1_10',
        'image_type': 'single_image',
        'bands': {
            'treecover2000': {'description': 'Tree cover 2000 (%)', 'dtype': 'uint8'},
            'loss': {'description': 'Forest loss binary', 'dtype': 'uint8'},
            'gain': {'description': 'Forest gain 2000-2012 (%)', 'dtype': 'uint8'},
            'lossyear': {'description': 'Loss year (0-23, years since 2000)', 'dtype': 'uint8'},
            'datamask': {'description': 'Data mask (0=no data, 1=data)', 'dtype': 'uint8'},
        },
        'scale_m': 30,
        'description': 'Hansen Global Forest Change 2023 v1.10'
    },

    'copernicus_dem': {
        'asset_id': 'COPERNICUS/DEM/GLO30',
        'image_type': 'single_image',
        'bands': {
            'DEM': {'description': 'Elevation (meters)', 'dtype': 'int16'},
            'EDM': {'description': 'Error estimate (meters)', 'dtype': 'uint8'},
            'FLM': {'description': 'Flag: void-filled (0/1)', 'dtype': 'uint8'},
        },
        'scale_m': 30,
        'description': 'Copernicus 30m Digital Elevation Model',
        'derived_bands': {
            'slope': {'description': 'Slope (degrees)', 'dtype': 'float32'},
            'aspect': {'description': 'Aspect (degrees)', 'dtype': 'float32'},
        }
    },

    'era5_climate': {
        'asset_id': 'ECMWF/ERA5/MONTHLY',
        'image_type': 'image_collection_mean',
        'date_range_days': 365,
        'bands': {
            'mean_2m_air_temperature': {'description': 'Mean 2m air temperature (K)', 'dtype': 'float32'},
            'maximum_2m_air_temperature': {'description': 'Max 2m air temperature (K)', 'dtype': 'float32'},
            'minimum_2m_air_temperature': {'description': 'Min 2m air temperature (K)', 'dtype': 'float32'},
            'total_precipitation': {'description': 'Total precipitation (mm)', 'dtype': 'float32'},
            'mean_total_column_water_vapour': {'description': 'Column water vapor (kg/m²)', 'dtype': 'float32'},
            'mean_sea_level_pressure': {'description': 'Sea level pressure (Pa)', 'dtype': 'float32'},
        },
        'scale_m': 31000,  # ~31km resolution
        'description': 'ERA5 Climate Reanalysis (monthly)'
    },

    'worldcover_landuse': {
        'asset_id': 'ESA/WorldCover/v200',
        'image_type': 'single_image',
        'bands': {
            'Map': {'description': 'Land cover class (0-100)', 'dtype': 'uint8'},
        },
        'scale_m': 10,
        'description': 'ESA WorldCover 10m Land Cover Classification'
    },

    'modis_vegetation': {
        'asset_id': 'MODIS/061/MOD09GA',
        'image_type': 'image_collection_median',
        'date_range_days': 365,
        'bands': {
            'sur_refl_b01': {'description': 'Surface reflectance band 1 (620-670nm, red)', 'dtype': 'int16'},
            'sur_refl_b02': {'description': 'Surface reflectance band 2 (841-876nm, NIR)', 'dtype': 'int16'},
            'sur_refl_b03': {'description': 'Surface reflectance band 3 (459-479nm, blue)', 'dtype': 'int16'},
        },
        'scale_m': 500,
        'description': 'MODIS Surface Reflectance (for NDVI/EVI)',
        'derived_bands': {
            'NDVI': {'description': 'Normalized Difference Vegetation Index', 'dtype': 'float32'},
            'EVI': {'description': 'Enhanced Vegetation Index', 'dtype': 'float32'},
        }
    },

    'gldas_soil_moisture': {
        'asset_id': 'NASA/GLDAS/V021/NOAH/G025/T3H',
        'image_type': 'image_collection_mean',
        'date_range_days': 365,
        'bands': {
            'SoilMoist_s_sfc_Percentile': {'description': 'Surface soil moisture (0-100 percentile)', 'dtype': 'uint8'},
            'SoilMoist_s_10cm_Percentile': {'description': '10cm soil moisture (0-100 percentile)', 'dtype': 'uint8'},
        },
        'scale_m': 27000,
        'description': 'GLDAS Soil Moisture Profiles'
    },

    'modis_gpp': {
        'asset_id': 'MODIS/061/MOD17A2H',
        'image_type': 'image_collection_mean',
        'date_range_days': 365,
        'bands': {
            'Gpp': {'description': 'Gross Primary Productivity (kg C/m²/8-day)', 'dtype': 'int16'},
            'GppQC': {'description': 'Quality control flags', 'dtype': 'uint8'},
        },
        'scale_m': 500,
        'description': 'MODIS Gross Primary Productivity'
    },

    'worldpop_density': {
        'asset_id': 'WorldPop/GP/100m/pop',
        'image_type': 'single_image_mosaic',
        'bands': {
            'population_density': {'description': 'Population density (persons/km²)', 'dtype': 'float32'},
        },
        'scale_m': 100,
        'description': 'WorldPop Global Population Density Estimates'
    },

    'modis_snow_cover': {
        'asset_id': 'MODIS/061/MOD10A1',
        'image_type': 'image_collection_mean',
        'date_range_days': 365,
        'bands': {
            'NDSI_Snow_Cover': {'description': 'Snow cover (0-100%)', 'dtype': 'uint8'},
            'NDSI_Data_Quality': {'description': 'Data quality flags', 'dtype': 'uint8'},
        },
        'scale_m': 500,
        'description': 'MODIS Snow Cover Extent'
    },

    'chirps_precipitation': {
        'asset_id': 'UCSB-CHG/CHIRPS/DAILY',
        'image_type': 'image_collection_sum',
        'date_range_days': 365,
        'bands': {
            'precipitation': {'description': 'Daily precipitation (mm)', 'dtype': 'float32'},
        },
        'scale_m': 5000,
        'description': 'CHIRPS Daily Precipitation'
    },

    'modis_lai': {
        'asset_id': 'MODIS/061/MYD15A2H',
        'image_type': 'image_collection_mean',
        'date_range_days': 365,
        'bands': {
            'Lai': {'description': 'Leaf Area Index (m²/m²)', 'dtype': 'int16'},
            'LaiQC': {'description': 'Quality control flags', 'dtype': 'uint8'},
        },
        'scale_m': 500,
        'description': 'MODIS Leaf Area Index'
    },
}


# ============================================================================
# MEGA IMAGE BUILDER
# ============================================================================

class MegaImageBuilder:
    """
    Constructs a mega-image by stacking bands from multiple GEE datasets.

    Benefits:
    - Reduces API calls: all bands sampled in one call per point
    - Improves cache locality: coordinate sorting before sampling
    - Tracks band provenance: knows which band came from which dataset

    Usage:
        builder = MegaImageBuilder(enabled_datasets=['jrc_global_water', 'viirs_nighttime_lights'])
        mega_image = builder.build_mega_image()
        results = builder.sample_mega_image(lats, lons, scale=30)
        metadata = builder.get_band_metadata()
    """

    def __init__(self, enabled_datasets: Optional[List[str]] = None, authenticated: bool = False):
        """
        Initialize builder.

        Args:
            enabled_datasets: List of dataset keys to include (from GEE_ASSET_CATALOG)
            authenticated: Whether GEE is authenticated for API calls
        """
        self._authenticated = authenticated and ee is not None
        self.enabled_datasets = enabled_datasets or []
        self.mega_image = None
        self.band_metadata = {}
        self._band_name_mapping = {}  # Maps internal band names to output names

        logger.info(f"MegaImageBuilder initialized with {len(self.enabled_datasets)} datasets")

    def build_mega_image(self, apply_terrain_products: bool = True) -> Optional[Any]:
        """
        Build mega-image by stacking all enabled datasets.

        Args:
            apply_terrain_products: Whether to compute slope/aspect from DEM

        Returns:
            ee.Image with all bands stacked, or None if error
        """
        if not self._authenticated:
            logger.warning("GEE not authenticated - cannot build mega-image")
            return None

        if not self.enabled_datasets:
            logger.warning("No enabled datasets specified")
            return None

        try:
            logger.info(f"Building mega-image from {len(self.enabled_datasets)} datasets...")

            # Start with None, will accumulate bands
            mega_image = None
            band_index = 0

            for dataset_name in self.enabled_datasets:
                if dataset_name not in GEE_ASSET_CATALOG:
                    logger.warning(f"  ⊘ Dataset not in catalog: {dataset_name}")
                    continue

                catalog_entry = GEE_ASSET_CATALOG[dataset_name]

                try:
                    # Load the image or collection
                    asset_id = catalog_entry['asset_id']
                    image_type = catalog_entry['image_type']

                    logger.info(f"  Loading {dataset_name} ({asset_id})...")

                    if image_type == 'single_image':
                        image = ee.Image(asset_id)

                    elif image_type == 'single_image_mosaic':
                        # Mosaic of single images (WorldPop case)
                        collection = ee.ImageCollection(asset_id)
                        image = collection.mosaic()

                    elif image_type == 'image_collection_median':
                        collection = ee.ImageCollection(asset_id)
                        date_range = catalog_entry.get('date_range_days', 365)
                        image = self._filter_and_aggregate(
                            collection, date_range, 'median'
                        )

                    elif image_type == 'image_collection_mean':
                        collection = ee.ImageCollection(asset_id)
                        date_range = catalog_entry.get('date_range_days', 365)
                        image = self._filter_and_aggregate(
                            collection, date_range, 'mean'
                        )

                    elif image_type == 'image_collection_sum':
                        collection = ee.ImageCollection(asset_id)
                        date_range = catalog_entry.get('date_range_days', 365)
                        image = self._filter_and_aggregate(
                            collection, date_range, 'sum'
                        )

                    else:
                        logger.warning(f"  Unknown image type: {image_type}")
                        continue

                    # Select and rename bands
                    band_names = list(catalog_entry['bands'].keys())
                    image_selected = image.select(band_names)

                    # Create output band names with dataset prefix
                    output_names = [f"{dataset_name}_{band}" for band in band_names]
                    image_renamed = image_selected.rename(output_names)

                    # Store band metadata
                    for original_name, output_name in zip(band_names, output_names):
                        self.band_metadata[output_name] = {
                            'source_dataset': dataset_name,
                            'original_band_name': original_name,
                            'description': catalog_entry['bands'][original_name]['description'],
                            'dtype': catalog_entry['bands'][original_name]['dtype'],
                            'scale_m': catalog_entry['scale_m'],
                        }

                    # Handle derived bands (slope, aspect from DEM)
                    if apply_terrain_products and 'derived_bands' in catalog_entry:
                        derived = self._compute_derived_bands(image, dataset_name, catalog_entry)
                        image_renamed = image_renamed.addBands(derived)

                    # Stack into mega-image
                    if mega_image is None:
                        mega_image = image_renamed
                    else:
                        mega_image = mega_image.addBands(image_renamed)

                    band_index += len(output_names)
                    logger.info(f"    ✓ {dataset_name}: {len(output_names)} bands ({band_index} total)")

                except Exception as e:
                    logger.error(f"  ✗ Failed to load {dataset_name}: {e}")
                    continue

            if mega_image is None:
                logger.error("Failed to build mega-image - no valid datasets")
                return None

            self.mega_image = mega_image
            logger.info(f"✅ Mega-image built successfully with {len(self.band_metadata)} bands")

            return mega_image

        except Exception as e:
            logger.error(f"Mega-image building failed: {e}", exc_info=True)
            return None

    def _filter_and_aggregate(self, collection: Any, days_back: int, aggregation: str) -> Any:
        """
        Filter image collection by date and aggregate.

        Args:
            collection: ee.ImageCollection
            days_back: Number of days back from today
            aggregation: 'median', 'mean', or 'sum'

        Returns:
            ee.Image with aggregation applied
        """
        try:
            end_date = ee.Date.now()
            start_date = end_date.advance(-days_back, 'day')

            filtered = collection.filterDate(start_date, end_date)

            if aggregation == 'median':
                return filtered.median()
            elif aggregation == 'mean':
                return filtered.mean()
            elif aggregation == 'sum':
                return filtered.sum()
            else:
                return filtered.first()

        except Exception as e:
            logger.error(f"Collection filtering failed: {e}")
            return collection.first()

    def _compute_derived_bands(self, dem_image: Any, dataset_name: str, catalog_entry: Dict) -> Any:
        """
        Compute derived bands (e.g., slope, aspect from DEM).

        Args:
            dem_image: ee.Image with DEM
            dataset_name: Name of dataset
            catalog_entry: Catalog entry with derived_bands info

        Returns:
            ee.Image with derived bands
        """
        try:
            # For Copernicus DEM
            if dataset_name == 'copernicus_dem':
                dem = dem_image.select('DEM')
                terrain = ee.Terrain.products(dem)

                slope = terrain.select('slope').rename(f'{dataset_name}_slope')
                aspect = terrain.select('aspect').rename(f'{dataset_name}_aspect')

                # Store metadata for derived bands
                for derived_band_name, derived_info in catalog_entry['derived_bands'].items():
                    output_name = f"{dataset_name}_{derived_band_name}"
                    self.band_metadata[output_name] = {
                        'source_dataset': dataset_name,
                        'original_band_name': derived_band_name,
                        'description': derived_info['description'],
                        'dtype': derived_info['dtype'],
                        'is_derived': True,
                        'scale_m': catalog_entry['scale_m'],
                    }

                return slope.addBands(aspect)

            # For MODIS vegetation (NDVI, EVI)
            elif dataset_name == 'modis_vegetation':
                red = dem_image.select('sur_refl_b01').multiply(0.0001)
                nir = dem_image.select('sur_refl_b02').multiply(0.0001)
                blue = dem_image.select('sur_refl_b03').multiply(0.0001)

                ndvi = nir.subtract(red).divide(nir.add(red)).rename(f'{dataset_name}_NDVI')
                evi = nir.subtract(red).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).multiply(2.5).rename(f'{dataset_name}_EVI')

                # Store metadata
                for derived_band_name, derived_info in catalog_entry['derived_bands'].items():
                    output_name = f"{dataset_name}_{derived_band_name}"
                    self.band_metadata[output_name] = {
                        'source_dataset': dataset_name,
                        'original_band_name': derived_band_name,
                        'description': derived_info['description'],
                        'dtype': derived_info['dtype'],
                        'is_derived': True,
                        'scale_m': catalog_entry['scale_m'],
                    }

                return ndvi.addBands(evi)

        except Exception as e:
            logger.warning(f"Failed to compute derived bands for {dataset_name}: {e}")

        return None

    def sample_mega_image(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        scale: int = 30,
        sort_coords: bool = True,
        sort_method: str = 'lat'
    ) -> Tuple[Dict[int, Dict[str, float]], np.ndarray, np.ndarray]:
        """
        Sample mega-image at coordinates (1 API call per point).

        Args:
            lats: Latitude array
            lons: Longitude array
            scale: Sampling resolution in meters
            sort_coords: Whether to sort coords for cache locality
            sort_method: 'lat', 'lon', or 'hilbert' for sorting

        Returns:
            Tuple of:
            - results: Dict mapping {0, 1, 2, ...} → {band_name: value, ...}
            - sorted_lats: Sorted latitude array (or original if not sorted)
            - sorted_lons: Sorted longitude array (or original if not sorted)
        """
        if self.mega_image is None:
            logger.error("Mega-image not built - call build_mega_image() first")
            return {}, lats, lons

        if not self._authenticated:
            logger.warning("GEE not authenticated - cannot sample")
            return {}, lats, lons

        try:
            n_points = len(lats)
            logger.info(f"Sampling mega-image at {n_points} points (scale={scale}m, sort={sort_coords})")

            # Sort coordinates if requested
            if sort_coords:
                sorted_lats, sorted_lons, sort_indices = sort_coordinates_for_locality(
                    lats, lons, sort_by=sort_method
                )
                logger.info(f"  Coordinates sorted by {sort_method} for cache locality")
            else:
                sorted_lats = lats
                sorted_lons = lons
                sort_indices = np.arange(n_points)

            # Sample all points
            results = {}
            batch_size = 100
            n_batches = (n_points + batch_size - 1) // batch_size

            logger.info(f"  Sampling in {n_batches} batches (batch_size={batch_size})")

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_points)

                batch_lats = sorted_lats[start_idx:end_idx]
                batch_lons = sorted_lons[start_idx:end_idx]
                batch_indices = sort_indices[start_idx:end_idx]

                for sample_idx, lat, lon, original_idx in zip(
                    range(start_idx, end_idx),
                    batch_lats,
                    batch_lons,
                    batch_indices
                ):
                    try:
                        point = ee.Geometry.Point([lon, lat])
                        sample_collection = self.mega_image.sample(point, scale)
                        sample_data = sample_collection.first().getInfo()

                        if sample_data and 'properties' in sample_data:
                            results[int(original_idx)] = sample_data['properties']

                    except Exception as e:
                        logger.debug(f"    Failed to sample point ({lat:.4f}, {lon:.4f}): {e}")

                if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    coverage = len(results) / n_points * 100
                    logger.info(f"    Batch {batch_idx + 1}/{n_batches} ({coverage:.1f}% complete)")

            logger.info(f"✅ Sampling complete: {len(results)}/{n_points} points sampled")

            return results, sorted_lats, sorted_lons

        except Exception as e:
            logger.error(f"Sampling failed: {e}", exc_info=True)
            return {}, lats, lons

    def get_band_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all bands in mega-image.

        Returns:
            Dict mapping {band_name: {description, source_dataset, dtype, scale_m, ...}}
        """
        return self.band_metadata.copy()

    def export_band_metadata_to_json(self, output_file: str):
        """
        Export band metadata to JSON file for data dictionary.

        Args:
            output_file: Path to output JSON file
        """
        try:
            metadata_export = {
                'generated_at': datetime.now().isoformat(),
                'total_bands': len(self.band_metadata),
                'bands': self.band_metadata,
                'enabled_datasets': self.enabled_datasets,
            }

            with open(output_file, 'w') as f:
                json.dump(metadata_export, f, indent=2)

            logger.info(f"Band metadata exported to {output_file}")

        except Exception as e:
            logger.error(f"Failed to export band metadata: {e}")


# ============================================================================
# COORDINATE SORTING FOR CACHE LOCALITY
# ============================================================================

def sort_coordinates_for_locality(
    lats: np.ndarray,
    lons: np.ndarray,
    sort_by: str = 'lat'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort coordinates to improve GEE server cache locality.

    Consecutive queries to nearby coordinates are more likely to hit
    cached data on GEE servers, improving latency and reducing load.

    Args:
        lats: Latitude array
        lons: Longitude array
        sort_by: Sorting method:
            - 'lat': Sort by latitude (fastest, groups by latitude bands)
            - 'lon': Sort by longitude (groups by longitude meridians)
            - 'hilbert': Space-filling Hilbert curve (best locality)

    Returns:
        Tuple of:
        - sorted_lats: Sorted latitude array
        - sorted_lons: Sorted longitude array
        - sort_indices: Indices to map results back to original order
    """
    n_points = len(lats)
    original_indices = np.arange(n_points)

    logger.info(f"Sorting {n_points} coordinates by '{sort_by}' for cache locality...")

    if sort_by == 'lat':
        # Sort by latitude (simple, fast)
        sort_order = np.argsort(lats)

    elif sort_by == 'lon':
        # Sort by longitude
        sort_order = np.argsort(lons)

    elif sort_by == 'hilbert':
        # Space-filling Hilbert curve (best spatial locality)
        sort_order = _sort_by_hilbert_curve(lats, lons)

    else:
        logger.warning(f"Unknown sort method '{sort_by}', falling back to 'lat'")
        sort_order = np.argsort(lats)

    sorted_lats = lats[sort_order]
    sorted_lons = lons[sort_order]
    sort_indices = original_indices[sort_order]

    logger.info(f"  Coordinates sorted. Max distance between consecutive points: {_max_consecutive_distance(sorted_lats, sorted_lons):.2f} degrees")

    return sorted_lats, sorted_lons, sort_indices


def _max_consecutive_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Calculate maximum distance between consecutive points.
    Uses approximate great-circle distance in degrees.
    """
    if len(lats) < 2:
        return 0.0

    diffs = np.sqrt(np.diff(lats)**2 + np.diff(lons)**2)
    return float(np.max(diffs))


def _sort_by_hilbert_curve(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Sort coordinates along a Hilbert space-filling curve.

    This provides excellent spatial locality - nearby points in the
    original coordinate space remain nearby along the curve.

    Args:
        lats: Latitude array
        lons: Longitude array

    Returns:
        Array of indices sorted by Hilbert curve order
    """
    try:
        # Normalize coordinates to 0-1 range
        lat_min, lat_max = -90, 90
        lon_min, lon_max = -180, 180

        norm_lats = (lats - lat_min) / (lat_max - lat_min)
        norm_lons = (lons - lon_min) / (lon_max - lon_min)

        # Clamp to [0, 1]
        norm_lats = np.clip(norm_lats, 0, 1)
        norm_lons = np.clip(norm_lons, 0, 1)

        # Convert to Hilbert curve indices
        # Use order 16 for good precision (2^16 = 65536 cells)
        hilbert_order = 16
        hilbert_indices = _xy_to_hilbert(norm_lons, norm_lats, hilbert_order)

        # Sort by Hilbert index
        sort_order = np.argsort(hilbert_indices)

        return sort_order

    except Exception as e:
        logger.warning(f"Hilbert curve sorting failed: {e}, falling back to latitude sort")
        return np.argsort(lats)


def _xy_to_hilbert(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """
    Convert (x, y) coordinates to Hilbert curve indices.

    Args:
        x: X coordinates (0-1)
        y: Y coordinates (0-1)
        order: Hilbert curve order (higher = finer granularity)

    Returns:
        Hilbert curve indices for each point
    """
    n = 2 ** order  # Number of cells per dimension

    # Convert to integer grid coordinates
    x_int = (x * (n - 1)).astype(int)
    y_int = (y * (n - 1)).astype(int)

    # Compute Hilbert index for each point
    hilbert_indices = np.zeros(len(x), dtype=int)

    for i in range(len(x)):
        hilbert_indices[i] = _xy_to_hilbert_single(x_int[i], y_int[i], order)

    return hilbert_indices


def _xy_to_hilbert_single(x: int, y: int, order: int) -> int:
    """
    Convert a single (x, y) point to Hilbert curve index.

    Uses the standard algorithm for converting Cartesian coordinates
    to Hilbert curve position.

    Args:
        x: X coordinate (0 to 2^order-1)
        y: Y coordinate (0 to 2^order-1)
        order: Hilbert curve order

    Returns:
        Index along Hilbert curve
    """
    d = 0
    s = 1 << (order - 1)

    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)

        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        s >>= 1

    return d
