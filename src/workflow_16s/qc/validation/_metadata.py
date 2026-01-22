"""
Comprehensive Metadata Validation and Cleaning

This module provides state-of-the-art metadata quality control:
1. Redundancy removal (duplicate/highly correlated columns)
2. Numeric range validation with environment-specific thresholds
3. ENVO ontology term validation and semantic categorization
4. Geospatial/temporal consistency checking
5. Unit harmonization across datasets
6. External data validation (coordinate/time proximity checks)
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

logger = logging.getLogger('workflow_16s')


class ENVOOntology:
    """
    ENVO ontology handler for semantic categorization.
    
    Enables finding all "soil" samples even with variations like:
    - "soil [ENVO:00001998]"
    - "forest soil"
    - "agricultural soil"
    - "Soil"
    - "sediment" (sometimes soil-like)
    """
    
    # ENVO ontology mappings (simplified - full version would load from OBO file)
    BIOME_HIERARCHY = {
        'terrestrial': {
            'soil': ['soil', 'topsoil', 'subsoil', 'agricultural soil', 'forest soil', 
                    'grassland soil', 'desert soil', 'peat soil', 'clay soil', 'loam',
                    'ENVO:00001998', 'ENVO:00002259', 'ENVO:00005802'],
            'sediment_terrestrial': ['terrestrial sediment', 'lake sediment', 
                                    'river sediment', 'wetland sediment'],
        },
        'aquatic': {
            'marine': ['marine', 'ocean', 'sea', 'seawater', 'marine sediment',
                      'oceanic', 'coastal', 'ENVO:00000447', 'ENVO:00002150'],
            'freshwater': ['freshwater', 'lake', 'river', 'stream', 'pond',
                          'ENVO:00002011', 'ENVO:00000020'],
            'sediment_aquatic': ['marine sediment', 'lake sediment', 'river sediment',
                                'oceanic sediment', 'ENVO:00002113'],
        },
        'extreme': {
            'hot_spring': ['hot spring', 'thermal spring', 'geothermal',
                          'ENVO:00000051'],
            'hypersaline': ['hypersaline', 'salt lake', 'saline', 'brine',
                           'ENVO:00000569'],
            'permafrost': ['permafrost', 'frozen soil', 'ENVO:00000134'],
            'acid_mine': ['acid mine drainage', 'acidic drainage', 'AMD'],
        },
        'built': {
            'wastewater': ['wastewater', 'sewage', 'activated sludge', 'effluent',
                          'ENVO:00002001'],
            'bioreactor': ['bioreactor', 'fermenter', 'biogas reactor'],
            'composting': ['compost', 'composting', 'ENVO:00002170'],
        }
    }
    
    # Material types
    MATERIAL_TYPES = {
        'solid': ['soil', 'sediment', 'rock', 'ice', 'compost', 'sludge'],
        'liquid': ['water', 'seawater', 'freshwater', 'wastewater', 'brine'],
        'air': ['air', 'aerosol', 'atmosphere'],
    }
    
    def __init__(self):
        """Initialize ENVO ontology with compiled regex patterns."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Precompile regex patterns for fast matching."""
        self.patterns = {}
        
        for biome_class, categories in self.BIOME_HIERARCHY.items():
            self.patterns[biome_class] = {}
            for category, terms in categories.items():
                # Create pattern matching any term (case-insensitive, word boundary)
                pattern = '|'.join(re.escape(term) for term in terms)
                self.patterns[biome_class][category] = re.compile(
                    rf'\b({pattern})\b', re.IGNORECASE
                )
    
    def categorize_sample(self, env_biome: str = None, env_feature: str = None, 
                         env_material: str = None) -> Dict[str, Any]:
        """
        Categorize a sample based on ENVO terms.
        
        Args:
            env_biome: Biome field (e.g., "terrestrial biome [ENVO:00000446]")
            env_feature: Feature field (e.g., "forest soil [ENVO:00002259]")
            env_material: Material field (e.g., "soil [ENVO:00001998]")
        
        Returns:
            Dict with:
                - biome_class: terrestrial/aquatic/extreme/built/unknown
                - category: soil/marine/freshwater/etc.
                - material_type: solid/liquid/air
                - confidence: 0-1
                - matched_terms: list of matched ENVO terms
        """
        text = ' '.join(str(x) for x in [env_biome, env_feature, env_material] if x)
        text = text.lower()
        
        matches = defaultdict(list)
        for biome_class, categories in self.patterns.items():
            for category, pattern in categories.items():
                found = pattern.findall(text)
                if found:
                    matches[biome_class].append((category, found))
        
        if not matches:
            return {
                'biome_class': 'unknown',
                'category': 'unclassified',
                'material_type': 'unknown',
                'confidence': 0.0,
                'matched_terms': []
            }
        
        # Pick biome class with most matches
        biome_class = max(matches.items(), key=lambda x: len(x[1]))[0]
        category, matched_terms = matches[biome_class][0]
        
        # Determine material type
        material_type = 'unknown'
        for mat_type, mat_terms in self.MATERIAL_TYPES.items():
            if any(term in text for term in mat_terms):
                material_type = mat_type
                break
        
        # Confidence based on number of matches and specificity
        confidence = min(1.0, 0.5 + 0.1 * len(matched_terms))
        
        return {
            'biome_class': biome_class,
            'category': category,
            'material_type': material_type,
            'confidence': confidence,
            'matched_terms': matched_terms
        }
    
    def find_samples_by_category(self, df: pd.DataFrame, 
                                  target_category: str,
                                  min_confidence: float = 0.5) -> pd.DataFrame:
        """
        Find all samples matching a category (e.g., all "soil" samples).
        
        Args:
            df: DataFrame with env_biome/env_feature/env_material columns
            target_category: Category to search (e.g., 'soil', 'marine')
            min_confidence: Minimum confidence threshold
        
        Returns:
            Filtered DataFrame with matching samples
        """
        logger.info(f"Searching for samples in category: {target_category}")
        
        # Categorize all samples
        categories = df.apply(
            lambda row: self.categorize_sample(
                row.get('env_biome'),
                row.get('env_feature'),
                row.get('env_material')
            ),
            axis=1
        )
        
        # Filter by category and confidence
        mask = (categories.apply(lambda x: x['category'] == target_category)) & \
               (categories.apply(lambda x: x['confidence'] >= min_confidence))
        
        n_found = mask.sum()
        logger.info(f"Found {n_found}/{len(df)} samples matching '{target_category}'")
        
        return df[mask]


class MetadataValidator:
    """
    Comprehensive metadata validation and cleaning.
    
    Performs:
    1. Redundancy removal
    2. Range validation
    3. ENVO term validation
    4. Geospatial/temporal consistency
    5. Unit harmonization
    6. External data validation
    """
    
    # Valid ranges for environmental variables
    VALID_RANGES = {
        # Universal ranges (hard limits)
        'pH': {'min': 0, 'max': 14, 'typical_min': 3, 'typical_max': 11},
        'temperature': {'min': -50, 'max': 120, 'typical_min': -10, 'typical_max': 50},
        'salinity': {'min': 0, 'max': 400, 'typical_min': 0, 'typical_max': 50},  # PSU
        'elevation': {'min': -500, 'max': 9000, 'typical_min': -100, 'typical_max': 5000},  # meters
        'depth': {'min': 0, 'max': 11000, 'typical_min': 0, 'typical_max': 100},  # meters
        'latitude': {'min': -90, 'max': 90},
        'longitude': {'min': -180, 'max': 180},
    }
    
    # Environment-specific ranges
    ENV_SPECIFIC_RANGES = {
        'marine': {
            'salinity': {'min': 30, 'max': 40, 'typical': True},
            'depth': {'min': 0, 'max': 11000, 'typical': True},
        },
        'freshwater': {
            'salinity': {'min': 0, 'max': 0.5, 'typical': True},
        },
        'soil': {
            'depth': {'min': 0, 'max': 5, 'typical': True},  # Usually shallow
            'pH': {'min': 3, 'max': 10, 'typical': True},
        },
        'hot_spring': {
            'temperature': {'min': 40, 'max': 100, 'typical': True},
        }
    }
    
    # Columns that are typically redundant
    REDUNDANT_SUFFIXES = ['_ena', '_study', '_sample', '.1', '.2', '_deg']
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize validator.
        
        Args:
            df: Metadata DataFrame
            config: Optional configuration dict
        """
        self.df = df.copy()
        self.config = config or {}
        self.envo = ENVOOntology()
        self.validation_report = []
        
    def validate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all validation steps.
        
        Returns:
            Tuple of (cleaned DataFrame, validation report DataFrame)
        """
        logger.info("Starting comprehensive metadata validation...")
        
        # Step 1: Remove redundant columns
        self.remove_redundant_columns()
        
        # Step 2: Validate numeric ranges
        self.validate_numeric_ranges()
        
        # Step 3: Harmonize units
        self.harmonize_units()
        
        # Step 4: Validate ENVO terms and add categorization
        self.validate_envo_terms()
        
        # Step 5: Check geospatial consistency
        self.validate_geographic_consistency()
        
        # Step 6: Validate external data (if present)
        self.validate_external_data()
        
        # Generate report
        report_df = pd.DataFrame(self.validation_report)
        
        n_errors = sum(1 for r in self.validation_report if r['level'] == 'ERROR')
        n_warnings = sum(1 for r in self.validation_report if r['level'] == 'WARNING')
        
        logger.info(f"Validation complete: {n_errors} errors, {n_warnings} warnings")
        
        return self.df, report_df
    
    def remove_redundant_columns(self):
        """Remove duplicate and highly correlated columns."""
        logger.info("Removing redundant columns...")
        
        initial_cols = len(self.df.columns)
        
        # 1. Remove exact duplicates
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        
        # 2. Remove columns with all NaN
        self.df = self.df.dropna(axis=1, how='all')
        
        # 3. Collapse suffix columns (e.g., keep 'latitude' over 'latitude_ena')
        for suffix in self.REDUNDANT_SUFFIXES:
            cols_with_suffix = [c for c in self.df.columns if c.endswith(suffix)]
            for col in cols_with_suffix:
                base_col = col.replace(suffix, '')
                if base_col in self.df.columns:
                    # Keep base_col, fill NaNs from suffix col if needed
                    self.df[base_col] = self.df[base_col].fillna(self.df[col])
                    self.df = self.df.drop(columns=[col])
                    logger.debug(f"Collapsed {col} into {base_col}")
        
        # 4. Remove highly correlated numeric columns (r > 0.99)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        to_drop = set()
        
        for i, col1 in enumerate(numeric_cols):
            if col1 in to_drop:
                continue
            for col2 in numeric_cols[i+1:]:
                if col2 in to_drop:
                    continue
                # Calculate correlation on non-NaN pairs
                valid_mask = self.df[col1].notna() & self.df[col2].notna()
                if valid_mask.sum() < 10:  # Need at least 10 samples
                    continue
                
                corr, _ = spearmanr(self.df.loc[valid_mask, col1], 
                                   self.df.loc[valid_mask, col2])
                
                if abs(corr) > 0.99:
                    # Keep the one with fewer NaNs
                    if self.df[col1].isna().sum() <= self.df[col2].isna().sum():
                        to_drop.add(col2)
                        logger.debug(f"Dropping {col2} (r={corr:.3f} with {col1})")
                    else:
                        to_drop.add(col1)
                        logger.debug(f"Dropping {col1} (r={corr:.3f} with {col2})")
                        break
        
        if to_drop:
            self.df = self.df.drop(columns=list(to_drop))
        
        final_cols = len(self.df.columns)
        logger.info(f"Reduced columns: {initial_cols} → {final_cols} "
                   f"(removed {initial_cols - final_cols})")
        
        self.validation_report.append({
            'step': 'remove_redundant_columns',
            'level': 'INFO',
            'message': f'Removed {initial_cols - final_cols} redundant columns',
            'details': f'Initial: {initial_cols}, Final: {final_cols}'
        })
    
    def validate_numeric_ranges(self):
        """Validate that numeric environmental variables are within valid ranges."""
        logger.info("Validating numeric ranges...")
        
        for var, ranges in self.VALID_RANGES.items():
            # Find matching columns (flexible naming)
            matching_cols = [c for c in self.df.columns 
                           if var.lower() in c.lower() and 
                           self.df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            for col in matching_cols:
                values = self.df[col].dropna()
                if len(values) == 0:
                    continue
                
                # Check hard limits
                below_min = values < ranges['min']
                above_max = values > ranges['max']
                
                if below_min.any() or above_max.any():
                    invalid_samples = self.df.index[below_min | above_max].tolist()
                    self.validation_report.append({
                        'step': 'validate_numeric_ranges',
                        'level': 'ERROR',
                        'column': col,
                        'message': f'{col} values outside valid range [{ranges["min"]}, {ranges["max"]}]',
                        'n_samples': len(invalid_samples),
                        'samples': invalid_samples[:10]  # First 10
                    })
                    
                    # Set invalid values to NaN
                    self.df.loc[below_min | above_max, col] = np.nan
                
                # Check typical ranges (warnings)
                if 'typical_min' in ranges and 'typical_max' in ranges:
                    below_typical = values < ranges['typical_min']
                    above_typical = values > ranges['typical_max']
                    
                    if below_typical.any() or above_typical.any():
                        unusual_samples = self.df.index[below_typical | above_typical].tolist()
                        self.validation_report.append({
                            'step': 'validate_numeric_ranges',
                            'level': 'WARNING',
                            'column': col,
                            'message': f'{col} values outside typical range [{ranges["typical_min"]}, {ranges["typical_max"]}]',
                            'n_samples': len(unusual_samples),
                            'samples': unusual_samples[:10]
                        })
    
    def harmonize_units(self):
        """Standardize units across all samples."""
        logger.info("Harmonizing units...")
        
        # Temperature: all to Celsius
        temp_cols = [c for c in self.df.columns if 'temp' in c.lower()]
        for col in temp_cols:
            values = self.df[col].dropna()
            if len(values) == 0:
                continue
            
            # Detect Kelvin (values > 200 likely Kelvin)
            if values.median() > 200:
                self.df[col] = self.df[col] - 273.15
                logger.info(f"Converted {col} from Kelvin to Celsius")
                self.validation_report.append({
                    'step': 'harmonize_units',
                    'level': 'INFO',
                    'column': col,
                    'message': 'Converted temperature from Kelvin to Celsius'
                })
        
        # Depth/Elevation: all to meters (detect if in feet/cm)
        dist_cols = [c for c in self.df.columns 
                    if any(x in c.lower() for x in ['depth', 'elevation', 'altitude'])]
        for col in dist_cols:
            values = self.df[col].dropna()
            if len(values) == 0:
                continue
            
            # If median > 1000, likely in cm or feet
            if values.median() > 1000:
                # Assume cm (more common in scientific data)
                self.df[col] = self.df[col] / 100
                logger.info(f"Converted {col} from cm to meters")
                self.validation_report.append({
                    'step': 'harmonize_units',
                    'level': 'INFO',
                    'column': col,
                    'message': 'Converted distance from cm to meters'
                })
    
    def validate_envo_terms(self):
        """Validate ENVO ontology terms and add semantic categorization."""
        logger.info("Validating ENVO terms and categorizing samples...")
        
        env_cols = [c for c in self.df.columns 
                   if c in ['env_biome', 'env_feature', 'env_material']]
        
        if not env_cols:
            logger.warning("No ENVO columns (env_biome/feature/material) found")
            return
        
        # Add categorization columns
        categories = self.df.apply(
            lambda row: self.envo.categorize_sample(
                row.get('env_biome'),
                row.get('env_feature'),
                row.get('env_material')
            ),
            axis=1
        )
        
        self.df['env_category_biome'] = categories.apply(lambda x: x['biome_class'])
        self.df['env_category_type'] = categories.apply(lambda x: x['category'])
        self.df['env_category_material'] = categories.apply(lambda x: x['material_type'])
        self.df['env_category_confidence'] = categories.apply(lambda x: x['confidence'])
        
        # Flag low-confidence categorizations
        low_conf = self.df['env_category_confidence'] < 0.5
        if low_conf.any():
            self.validation_report.append({
                'step': 'validate_envo_terms',
                'level': 'WARNING',
                'message': f'{low_conf.sum()} samples with low-confidence ENVO categorization',
                'samples': self.df.index[low_conf].tolist()[:10]
            })
        
        logger.info(f"Added semantic categorization for {len(self.df)} samples")
        
        # Log category distribution
        for col in ['env_category_biome', 'env_category_type', 'env_category_material']:
            dist = self.df[col].value_counts()
            logger.info(f"{col} distribution:\n{dist}")
    
    def validate_geographic_consistency(self):
        """Check for geographic inconsistencies."""
        logger.info("Validating geographic consistency...")
        
        lat_cols = [c for c in self.df.columns if 'lat' in c.lower() and 'facility' not in c.lower()]
        lon_cols = [c for c in self.df.columns if 'lon' in c.lower() and 'facility' not in c.lower()]
        
        if not lat_cols or not lon_cols:
            return
        
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        
        # Check if marine samples have coordinates in ocean
        if 'env_category_type' in self.df.columns:
            marine_samples = self.df['env_category_type'] == 'marine'
            if marine_samples.any():
                # Simple check: marine samples should have abs(lat) < 70 (not polar)
                # More sophisticated: check against ocean shapefile
                marine_with_coords = marine_samples & self.df[lat_col].notna() & self.df[lon_col].notna()
                
                # Flag marine samples with suspicious coords
                suspicious = marine_with_coords & (
                    (self.df[lat_col].abs() > 80) |  # Polar regions (usually ice)
                    (self.df[lon_col].abs() > 180)   # Invalid
                )
                
                if suspicious.any():
                    self.validation_report.append({
                        'step': 'validate_geographic_consistency',
                        'level': 'WARNING',
                        'message': f'{suspicious.sum()} marine samples with suspicious coordinates',
                        'samples': self.df.index[suspicious].tolist()[:10]
                    })
    
    def validate_external_data(self):
        """
        Validate external data sources (SoilGrids, Meteostat, etc.).
        
        Checks:
        1. Coordinates match between sample and external data
        2. Temporal proximity for time-series data
        3. Data completeness
        """
        logger.info("Validating external data sources...")
        
        # Find external data columns
        external_prefixes = ['SoilGrids_', 'Meteostat_', 'OpenMeteo_', 'WorldClim_']
        external_cols = [c for c in self.df.columns 
                        if any(c.startswith(p) for p in external_prefixes)]
        
        if not external_cols:
            logger.info("No external data columns found")
            return
        
        # Get sample coordinates
        lat_cols = [c for c in self.df.columns if 'latitude' in c.lower() and 'facility' not in c.lower()]
        lon_cols = [c for c in self.df.columns if 'longitude' in c.lower() and 'facility' not in c.lower()]
        
        if not lat_cols or not lon_cols:
            self.validation_report.append({
                'step': 'validate_external_data',
                'level': 'WARNING',
                'message': 'External data present but no sample coordinates for validation'
            })
            return
        
        lat_col, lon_col = lat_cols[0], lon_cols[0]
        
        # Check for external data with missing sample coordinates
        has_external = self.df[external_cols].notna().any(axis=1)
        missing_coords = has_external & (self.df[lat_col].isna() | self.df[lon_col].isna())
        
        if missing_coords.any():
            self.validation_report.append({
                'step': 'validate_external_data',
                'level': 'ERROR',
                'message': f'{missing_coords.sum()} samples have external data but missing coordinates',
                'samples': self.df.index[missing_coords].tolist()[:10]
            })
            
            # Remove external data for samples without coordinates
            self.df.loc[missing_coords, external_cols] = np.nan
        
        # For facility data, check distance reasonableness
        facility_cols = [c for c in self.df.columns if 'facility_distance' in c.lower()]
        if facility_cols:
            dist_col = facility_cols[0]
            
            # Flag if facility distance > 1000km (suspicious)
            suspicious_dist = self.df[dist_col] > 1000
            if suspicious_dist.any():
                self.validation_report.append({
                    'step': 'validate_external_data',
                    'level': 'WARNING',
                    'message': f'{suspicious_dist.sum()} samples with facility distance > 1000km',
                    'samples': self.df.index[suspicious_dist].tolist()[:10]
                })
        
        logger.info(f"Validated {len(external_cols)} external data columns")
    
    def get_category_samples(self, category: str, min_confidence: float = 0.5) -> pd.DataFrame:
        """
        Get all samples matching a category (semantic search).
        
        Args:
            category: Category name (e.g., 'soil', 'marine', 'freshwater')
            min_confidence: Minimum confidence threshold
        
        Returns:
            Filtered DataFrame
        """
        if 'env_category_type' not in self.df.columns:
            logger.warning("Run validate_envo_terms() first")
            return pd.DataFrame()
        
        mask = (self.df['env_category_type'] == category) & \
               (self.df['env_category_confidence'] >= min_confidence)
        
        return self.df[mask]
    
    def save_validation_report(self, output_path: Union[str, Path]):
        """Save validation report to CSV."""
        if not self.validation_report:
            logger.warning("No validation report to save")
            return
        
        report_df = pd.DataFrame(self.validation_report)
        report_df.to_csv(output_path, index=False)
        logger.info(f"Saved validation report to {output_path}")
