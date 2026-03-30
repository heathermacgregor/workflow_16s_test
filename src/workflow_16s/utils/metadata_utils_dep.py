# ===================================== IMPORTS ====================================== #

# Standard Imports
import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
import aiohttp
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from rapidfuzz import fuzz

# Local Imports
from workflow_16s.config import AppConfig
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.constants import SAMPLE_ID_COLUMN

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

class MetadataManager:
    """A unified class to handle the cleaning, processing, and enrichment of metadata.

    This class provides a complete, modular pipeline for:
    1.  **Loading and Saving Data**: Handles TSV file I/O.
    2.  **Cleaning**: Standardizes formats, handles duplicates, converts types,
        and collapses redundant columns.
    3.  **Processing**: Extracts key information like geolocation and infers
        ontologies from unstructured text.
    4.  **Enrichment**: Uses external APIs to add supplementary data like
        publication DOIs, location names from coordinates, and ENVO labels.

    The primary entry point is the `run_pipeline()` method, which executes these
    steps in a logical order.
    """
    # ================================= CLASS ATTRIBUTES ================================= #

    # Pre-compiled regex for efficiency
    NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
    PH_PATTERN = re.compile(r'^ph[^a-zA-Z]|^ph$')

    # Default definitions for coordinates, column names, units, etc.
    DEFAULT_COORDINATE_SOURCES = {
        'lat': [
            'lat_study', 'lat_ena', 'lat.1', 'lat', 'biosample_geographic_location_(latitude)',
            'biosample_latitude', 'experiment_lat', 'run_lat', 'latitude'
        ],
        'lon': [
            'lon_study', 'lon.1', 'lon', 'biosample_geographic_location_(longitude)',
            'biosample_longitude', 'experiment_lon', 'run_lon', 'longitude'
        ],
        'pairs': [
            'location_ena', 'location_start', 'location_end', 'location_start_study',
            'location_end_study', 'lat_lon', 'location', 'biosample_lat_lon',
            'biosample_latitude_and_longitude', 'run_location', 'run_location_start',
            'run_location_end', 'experiment_location', 'experiment_location_start',
            'experiment_location_end'
        ]
    }
    DEFAULT_COLUMN_MAPPINGS = {
        'env_biome': 'environment_biome', 'env_feature': 'environment_feature',
        'env_material': 'environment_material'
    }
    DEFAULT_UNIT_PATTERNS = {
        'celsius': re.compile(r'_(?:celsius|cel|c)$', re.IGNORECASE),
        'fahrenheit': re.compile(r'_(?:fahrenheit|far|f)$', re.IGNORECASE),
        'kelvin': re.compile(r'_(?:kelvin|k)$', re.IGNORECASE),
        'meters': re.compile(r'_(?:meters|meter|m)$', re.IGNORECASE),
        'feet': re.compile(r'_(?:feet|ft)$', re.IGNORECASE)
    }
    DEFAULT_CONVERSIONS: Dict[str, Tuple[str, Callable[[pd.Series], pd.Series]]] = {
        'fahrenheit': ('celsius', lambda f: (pd.to_numeric(f, errors='coerce') - 32) * 5 / 9),
        'kelvin': ('celsius', lambda k: pd.to_numeric(k, errors='coerce') - 273.15),
        'feet': ('meters', lambda ft: pd.to_numeric(ft, errors='coerce') * 0.3048),
    }
    DEFAULT_MEASUREMENT_STANDARDS = {
        'temp': 'celsius', 'depth': 'meters', 'altitude': 'meters'
    }

    # ================================= INITIALIZATION ================================= #

    def __init__(
        self, metadata: pd.DataFrame, config: AppConfig,
        sample_id_column: str = SAMPLE_ID_COLUMN
    ):
        if metadata.empty:
            raise ValueError("Cannot process an empty metadata DataFrame.")

        self.df = metadata.copy()
        self.original_df_for_enrichment: Optional[pd.DataFrame] = None
        self.config = config
        self.sample_id_column = self.config.metadata.columns.sample_id or sample_id_column
        self.initial_shape = self.df.shape
        self.report: Dict[str, Any] = {
            'initial_shape': self.initial_shape, 'actions': [],
            'columns_dropped': {'unwanted': [], 'duplicate': [], 'merged': []},
            'numeric_coercions': {}, 'categorical_standardizations': {},
            'unit_standardizations': {}
        }
        self.ncbi_api_key = self.config.credentials.ncbi_api_key or None
        self.ncbi_semaphore = asyncio.Semaphore(10)
        self.ncbi_pacing_lock = asyncio.Lock()      # Ensure paced requests
        self.last_ncbi_request_time = 0             # Track the time of the last request

        self._define_ontology_maps()
        logger.info(f"Initialized MetadataManager with shape {self.df.shape}.")

    # ================================= MAIN EXECUTOR ================================== #

    async def run_pipeline(self) -> pd.DataFrame:
        """
        Executes the full cleaning, processing, and enrichment pipeline.
        
        Returns:
            A fully cleaned, processed, and enriched metadata DataFrame.
        """
        logger.info("[!] Starting metadata processing pipeline...")
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)

        # Core cleaning and standardization (Synchronous)
        self._run_cleaning_steps()

        # Data extraction and ontology Inference (Synchronous)
        self._run_processing_steps()
        if self.df.empty:
            logger.warning("DataFrame is empty after processing steps. Returning original DataFrame.")
            return self.df

        # 3. External Data Enrichment (Asynchronous)
        await self._run_enrichment_steps()

        logger.info(f"[X] Metadata processing pipeline complete. Final shape: {self.df.shape}")
        return self.df.copy()

    # ============================== PIPELINE STAGES =============================== #

    def _run_cleaning_steps(self) -> None:
        """Executes foundational cleaning tasks."""
        logger.info("--- Running Stage 1: Cleaning and Standardization ---")
        steps = [
            ("Dropping unwanted columns", self._drop_unwanted_columns),
            ("Removing duplicate columns", self._clean_duplicate_columns),
            ("Cleaning sample IDs and removing duplicate rows", self._clean_sample_ids),
            ("Coercing specified columns to numeric", self._clean_numeric_columns),
            ("Standardizing categorical values", self._standardize_categorical_values),
            ("Applying custom filters", self._apply_custom_filters),
            ("Standardizing columns with units", self._standardize_units),
            ("Collapsing suffix columns", self._collapse_all_suffixes),
            ("Consolidating pH columns", self._collapse_ph_columns),
            ("Standardizing column names", self._standardize_column_names)
        ]
        self._execute_steps(steps)

    def _run_processing_steps(self) -> None:
        """Executes data extraction and inference tasks."""
        logger.info("--- Running Stage 2: Processing and Inference ---")
        
        self.original_df_for_enrichment = self.df.copy()
        steps = [
            ("Extracting and validating geolocation", self._process_geolocation),
            ("Inferring ontology terms", self._process_ontology),
            ("Processing contamination status", self._process_contamination_status),
            ("Ensuring ENA accession columns exist", self._process_ena_accessions),
            ("Standardizing date formats", self._standardize_dates)
        ]
        self._execute_steps(steps)

    async def _run_enrichment_steps(self) -> None:
        """Executes tasks that enrich data using external sources concurrently."""
        logger.info("--- Running Stage 3: Enrichment (Async) ---")
        async with aiohttp.ClientSession() as session:
            # Run these sequentially, but internal operations of each step are run concurrently
            await self._enrich_location_from_coords(session)
            await self._convert_envo_codes(session)
            await self._find_publications(session, api_key=self.ncbi_api_key)

    def _execute_steps(self, steps: List[Tuple[str, Callable]]) -> None:
        """Generic step executor to run a list of synchronous functions."""
        for name, func in steps:
            try:
                # logger.info(f"Executing: {name}...")
                func()
                self.report['actions'].append(name)
            except Exception as e:
                logger.error(f"Error during '{name}': {e}", exc_info=True)
                raise

    # =========================== I/O STATIC METHODS ============================= #

    @staticmethod
    def import_tsv(metadata_path: Union[str, Path]) -> pd.DataFrame:
        """Loads a DataFrame from a TSV file."""
        return pd.read_csv(metadata_path, sep='\t', low_memory=False, dtype=str)

    @staticmethod
    def export_tsv(metadata: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Exports a DataFrame to a TSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Metadata successfully exported to {output_path}")

    @staticmethod
    def import_merged_metadata(metadata_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """Loads and merges multiple metadata TSV files into one DataFrame."""
        dfs: List[pd.DataFrame] = []
        with get_progress_bar() as progress:
            task = progress.add_task("Loading metadata files", total=len(metadata_paths))
            for path in metadata_paths:
                try:
                    df = MetadataManager.import_tsv(path)
                    df.columns = df.columns.str.lower().str.strip()
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to load metadata file {path}: {e!r}")
                finally: progress.update(task, advance=1)

        if not dfs:
            raise FileNotFoundError("No valid metadata files could be loaded.")
        return pd.concat(dfs, ignore_index=True)

    # ========================== CLEANING & STANDARDIZATION ========================== #

    def _drop_unwanted_columns(self) -> None:
        cols_to_drop = self.config.metadata.columns_to_drop or []
        # Use set intersection for efficient identification of existing columns
        existing_cols_to_drop = list(set(cols_to_drop) & set(self.df.columns))
        if existing_cols_to_drop:
            self.df.drop(columns=existing_cols_to_drop, inplace=True)
            self.report['columns_dropped']['unwanted'] = existing_cols_to_drop
            logger.info(f"Dropped {len(existing_cols_to_drop)} unwanted columns.")

    def _clean_duplicate_columns(self) -> None:
        if self.df.columns.duplicated().any():
            duplicated_cols = self.df.columns[self.df.columns.duplicated()].unique().tolist()
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            self.report['columns_dropped']['duplicate'] = duplicated_cols
            logger.warning(f"Removed duplicate columns: {duplicated_cols}")

    def _clean_sample_ids(self) -> None:
        if self.sample_id_column not in self.df.columns:
            alternatives = ['#sampleid', 'sample_id', 'sample id',
                            'sample name', 'run_accession']
            found_col = next((
                alt for alt in alternatives if alt in self.df.columns
            ), None)
            if found_col:
                logger.warning(f"'{self.sample_id_column}' not found. Creating it from '{found_col}'.")
                self.df[self.sample_id_column] = self.df[found_col]
            else:
                raise KeyError(f"Required sample ID column '{self.sample_id_column}' not found.")

        original_count = len(self.df)
        self.df[self.sample_id_column] = self.df[self.sample_id_column].astype(str).str.lower().str.strip()
        self.df.dropna(subset=[self.sample_id_column], inplace=True)
        self.df = self.df[self.df[self.sample_id_column] != '']
        self.df.drop_duplicates(
            subset=[self.sample_id_column], keep='first', inplace=True
        )
        removed_count = original_count - len(self.df)
        if removed_count > 0:
            self.report['duplicate_rows_removed'] = removed_count
            logger.warning(f"Removed {removed_count} rows with duplicate or missing sample IDs.")

    def _clean_numeric_columns(self) -> None:
        numeric_cols = self.config.metadata.force_numeric_columns or []
        for col in numeric_cols:
            if col in self.df.columns and self.df[col].dtype == 'object':
                initial_nans = self.df[col].isna().sum()
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                coerced_count = self.df[col].isna().sum() - initial_nans
                if coerced_count > 0:
                    self.report['numeric_coercions'][col] = coerced_count
                    logger.debug(f"Coerced {coerced_count} values to NaN in '{col}'.")

    def _standardize_categorical_values(self) -> None:
        mappings = self.config.metadata.mappings or {}
        for col, value_map in mappings.items():
            if col in self.df.columns:
                cleaned_series = self.df[col].astype(str).str.lower().str.strip()
                replaced_series = cleaned_series.replace(value_map)
                if not cleaned_series.equals(replaced_series):
                    self.df[col] = replaced_series
                    self.report['categorical_standardizations'][col] = value_map
                    logger.debug(f"Standardized values in column '{col}'.")

    def _apply_custom_filters(self) -> None:
        if 'empo_3' in self.df.columns:
            initial_rows = len(self.df)
            values_to_remove = ['animal distal gut', 'animal corpus', 'animal secretion']
            mask = self.df['empo_3'].astype(str).str.lower().isin(values_to_remove)
            self.df = self.df[~mask]
            rows_removed = initial_rows - len(self.df)
            if rows_removed > 0:
                logger.info(f"Filtered {rows_removed} rows based on 'empo_3' values.")
                self.report['custom_filters_applied'] = {
                    'column': 'empo_3', 'rows_removed': rows_removed
                }

    def _parse_column_unit(self, col_name: str) -> Tuple[Optional[str], Optional[str]]:
        for unit, pattern in self.DEFAULT_UNIT_PATTERNS.items():
            match = pattern.search(col_name)
            if match:
                return col_name[:match.start()], unit
        return None, None

    def _standardize_units(self) -> None:
        column_groups: Dict[str, List[Tuple[str, str]]] = {}
        for col in self.df.columns:
            base_name_raw, unit = self._parse_column_unit(col)
            if base_name_raw and unit:
                measurement_key = next((key for key in self.DEFAULT_MEASUREMENT_STANDARDS
                                      if key in base_name_raw), base_name_raw)
                column_groups.setdefault(measurement_key, []).append((col, unit))

        for base_name, cols_with_units in column_groups.items():
            if len(cols_with_units) < 2: continue
            
            target_unit = self.DEFAULT_MEASUREMENT_STANDARDS.get(base_name)
            if not target_unit: continue
            
            target_col_name = f"{base_name}_{target_unit}"
            logger.info(f"Merging {[c[0] for c in cols_with_units]} into '{target_col_name}'")
            
            merged_series = pd.Series(np.nan, index=self.df.index, dtype=float)
            for col_name, unit in cols_with_units:
                source_series = pd.to_numeric(self.df[col_name], errors='coerce')
                if unit == target_unit:
                    converted_series = source_series
                elif unit in self.DEFAULT_CONVERSIONS and self.DEFAULT_CONVERSIONS[unit][0] == target_unit:
                    conversion_func = self.DEFAULT_CONVERSIONS[unit][1]
                    converted_series = conversion_func(source_series)
                else:
                    logger.warning(f"Cannot convert '{unit}' to '{target_unit}' for '{col_name}'. Skipping.")
                    continue
                merged_series.update(converted_series)

            self.df[target_col_name] = merged_series
            cols_to_drop = [c for c, _ in cols_with_units]
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.report['columns_dropped']['merged'].extend(cols_to_drop)
            self.report['unit_standardizations'][target_col_name] = cols_to_drop

    def _collapse_all_suffixes(self) -> None:
        suffixes = self.config.metadata.suffixes_to_collapse or []
        for suffix in suffixes:
            self._collapse_suffix_columns(suffix)

    def _collapse_suffix_columns(self, suffix: str) -> None:
        suffix_cols = [col for col in self.df.columns if col.endswith(suffix)]
        cols_to_drop = []
        for col in suffix_cols:
            base_col = col[:-len(suffix)]
            if base_col in self.df.columns:
                self.df[base_col] = self.df[base_col].combine_first(self.df[col])
                cols_to_drop.append(col)
            else:
                self.df.rename(columns={col: base_col}, inplace=True)
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            self.report['columns_dropped']['merged_suffix'] = self.report['columns_dropped'].get('merged_suffix', []) + cols_to_drop

    def _collapse_ph_columns(self) -> None:
        ph_cols = [col for col in self.df.columns if self.PH_PATTERN.match(col) and 'std' not in col]
        if len(ph_cols) > 1:
            if 'ph' not in ph_cols:
                self.df['ph'] = np.nan
                ph_cols.insert(0, 'ph')
            
            for col in ph_cols:
                self.df['ph'].fillna(pd.to_numeric(self.df[col], errors='coerce'), inplace=True)

            cols_to_drop = [c for c in ph_cols if c != 'ph']
            self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            self.report['columns_dropped']['merged_ph'] = cols_to_drop

    def _standardize_column_names(self) -> None:
        mappings = self.config.metadata.mappings or self.DEFAULT_COLUMN_MAPPINGS
        self.df.rename(columns=mappings, inplace=True)

    def _standardize_dates(self) -> None:
        date_cols = [c for c in self.df.columns if 'date' in c.lower() or 'time' in c.lower()]
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce').dt.strftime('%Y-%m-%d')

    # ========================== PROCESSING & INFERENCE ============================ #

    def _define_ontology_maps(self) -> None:
        """Defines keyword maps for inferring ontology terms."""
        self.ONTOLOGY_MAP = {
            'empo_1': {
                'Host-associated': [
                    'host', 'symbiont', 'microbiome', 'human', 'animal'
                ],
                'Free-living': [
                    'free living', 'environmental', 'soil', 'water', 'sediment', 'air'
                ]
            },
            'empo_2': {
                'Animal': ['animal', 'human', 'insect', 'mammal', 'gut', 'feces', 'skin'],
                'Plant': ['plant', 'rhizosphere', 'root', 'leaf', 'flower'],
                'Fungus': ['fungus', 'fungal'],
                'Aquatic': ['aquatic', 'water', 'marine', 'freshwater', 'sediment', 'ocean'],
                'Terrestrial': ['terrestrial', 'soil', 'land', 'desert', 'forest']
            },
            'empo_3': {
                'Gut': ['gut', 'feces', 'fecal', 'intestinal'],
                'Soil': ['soil', 'rhizosphere', 'terrestrial'],
                'Water': ['water', 'aquatic', 'marine', 'freshwater'],
                'Sediment': ['sediment'], 'Skin': ['skin']
            },
            'env_biome': {
                'Urban': ['urban', 'city'],
                'Agricultural': ['agricultural', 'farm', 'crop'],
                'Forest': ['forest'],
                'Grassland': ['grassland', 'savanna'],
                'Aquatic': ['aquatic', 'marine', 'freshwater', 'lake', 'river', 'ocean']
            },
            'env_feature': {
                'Anthropogenic': ['anthropogenic', 'human-made', 'built environment'],
                'Natural': ['natural', 'wild']
            },
            'env_material': {
                'Soil': ['soil', 'loam', 'clay', 'silt'],
                'Water': ['water'],
                'Sediment': ['sediment', 'mud'],
                'Air': ['air']
            }
        }

    def _process_geolocation(self) -> None:
        """Extracts and validates latitude/longitude, filtering out invalid rows."""
        initial_count = len(self.df)
        
        lat = pd.Series(np.nan, index=self.df.index)
        lon = pd.Series(np.nan, index=self.df.index)

        lat_sources = [
           c for c in self.DEFAULT_COORDINATE_SOURCES['lat'] if c in self.df.columns
        ]
        lon_sources = [
            c for c in self.DEFAULT_COORDINATE_SOURCES['lon'] if c in self.df.columns
        ]
        if lat_sources:
            for col in lat_sources:
                lat.fillna(
                    pd.to_numeric(self.df[col], errors='coerce'), inplace=True
                )
        if lon_sources:
            for col in lon_sources:
                lon.fillna(
                    pd.to_numeric(self.df[col], errors='coerce'), inplace=True
                )
        
        missing_mask = lat.isna() | lon.isna()
        if missing_mask.any():
            pair_sources = [
                c for c in self.DEFAULT_COORDINATE_SOURCES['pairs'] if c in self.df.columns
            ]
            for source in pair_sources:
                if not missing_mask.any(): break
                to_process = self.df.loc[missing_mask, source].dropna()
                if to_process.empty: continue
                
                extracted = to_process.astype(str).apply(self._extract_coords_from_string).apply(pd.Series)
                if not extracted.empty:
                    extracted.columns=['new_lat', 'new_lon']
                    lat.update(extracted['new_lat'])
                    lon.update(extracted['new_lon'])
                    missing_mask = lat.isna() | lon.isna()

        self.df['lat'] = pd.to_numeric(lat, errors='coerce')
        self.df['lon'] = pd.to_numeric(lon, errors='coerce')
        valid_mask = (self.df['lat'].between(-90, 90)) & (self.df['lon'].between(-180, 180))
        self.df = self.df[valid_mask].reset_index(drop=True)
        dropped_count = initial_count - len(self.df)
        logger.info(f"Geolocation: {initial_count} initial -> {len(self.df)}"
                    f" valid. ({dropped_count} dropped).")

    def _extract_coords_from_string(
        self, s: str
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(s, str): return None, None
        dd_regex = r'([-+]?[1-8]?\d(?:\.\d+)?|[-+]?90(?:\.0+)?),\s*([-+]?180(?:\.0+)?|[-+]?(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)'
        match = re.search(dd_regex, s)
        if match:
            try: return float(match.group(1)), float(match.group(2))
            except (ValueError, IndexError): pass
        
        if '°' in s:
            parts = re.findall(r'(\d{1,3}(?:[°\.\d\s\'"]*))\s*([NSEW])', s, re.IGNORECASE)
            if len(parts) >= 2:
                lat_str, lat_dir = parts[0]
                lon_str, lon_dir = parts[1]
                lat = self._dms_to_dd(f"{lat_str} {lat_dir}")
                lon = self._dms_to_dd(f"{lon_str} {lon_dir}")
                if lat is not None and lon is not None: return lat, lon
        return None, None

    def _dms_to_dd(self, dms_str: str) -> Optional[float]:
        dms_str = dms_str.strip().upper()
        try:
            parts = re.split(r'[°\'"]+', dms_str)
            d = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 and parts[1].strip() else 0.0
            s = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            dd = d + m / 60.0 + s / 3600.0
            if re.search(r'[SW]', dms_str): dd *= -1
            return dd
        except (ValueError, IndexError): return None

    def _infer_ontology_term(self, text: str, term_map: Dict) -> str:
        text = text.lower()
        for term, keywords in term_map.items():
            if any(keyword in text for keyword in keywords): return term
        return 'Unknown'

    def _process_ontology(self) -> None:
        if self.df.empty: return
        search_text_series = self.df.select_dtypes(include='object').fillna('').astype(str).agg(' '.join, axis=1)
        for term_category, term_map in self.ONTOLOGY_MAP.items():
            if term_category not in self.df.columns or self.df[term_category].isnull().all():
                self.df[term_category] = search_text_series.apply(
                    self._infer_ontology_term, term_map=term_map
                )
                logger.debug(f"nferred ontology for '{term_category}'.")

    def _process_contamination_status(self) -> None:
        if 'nuclear_contamination_status' in self.df.columns:
            true_values = ['true', 'yes', '1', 'contaminated']
            mask = self.df['nuclear_contamination_status'].astype(str).str.lower().isin(true_values)
            self.df['nuclear_contamination_status'] = mask
        else: self.df['nuclear_contamination_status'] = False

    def _process_ena_accessions(self) -> None:
        acc_cols = ['ena_study_acc', 'ena_sample_acc', 'ena_experiment_acc', 'ena_run_acc']
        for col in acc_cols:
            if col not in self.df.columns: self.df[col] = 'N/A'
            else: self.df[col].fillna('N/A', inplace=True)

    # =============================== ENRICHMENT (ASYNC) =================================== #

    async def _enrich_location_from_coords(self, session: aiohttp.ClientSession) -> None:
        if 'location' not in self.df.columns: self.df['location'] = np.nan
        if 'lat' not in self.df.columns or 'lon' not in self.df.columns: return

        rows_to_check = self.df[self.df['location'].isnull() & self.df['lat'].notna()]
        if rows_to_check.empty: return

        logger.info(f"Found {len(rows_to_check)} rows to enrich with geocoding...")
        semaphore = asyncio.Semaphore(1) # Nominatim policy: 1 request/sec
        
        geolocator = Nominatim(
            user_agent="metadata_analysis_script_v3", adapter_factory=session.get
        )

        tasks = [
            self._fetch_single_location(geolocator, semaphore, index, row['lat'], row['lon'])
            for index, row in rows_to_check.iterrows()
        ]
        results = await asyncio.gather(*tasks)

        for index, location_str in results:
            if location_str: self.df.loc[index, 'location'] = location_str

    async def _fetch_single_location(
        self, geolocator, semaphore, index, lat, lon
    ) -> Tuple[int, Optional[str]]:
        try:
            async with semaphore:
                location = await geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
                if location and hasattr(location, 'raw') and 'address' in location.raw:
                    addr = location.raw['address']
                    city = addr.get('city', addr.get('town', addr.get('village', '')))
                    country = addr.get('country', '')
                    return index, ", ".join(filter(None, [city, country]))
        except (GeocoderTimedOut, GeocoderServiceError, asyncio.TimeoutError) as e:
            logger.warning(f"Geocoding service error for index {index}: {e}")
        return index, None

    async def _convert_envo_codes(self, session: aiohttp.ClientSession) -> None:
        envo_cols = ['env_material', 'env_feature', 'env_biome']
        all_codes_to_lookup = set()
        envo_pattern = re.compile(r'(ENVO[:_]\d{7})', re.IGNORECASE)
        
        # More efficient lookup: Concat all relevant columns, get unique values, then scan
        existing_cols = [c for c in envo_cols if c in self.df.columns]
        if not existing_cols:
            return

        all_unique_values = pd.concat(
            [self.df[c].dropna() for c in existing_cols]
        ).astype(str).unique()

        for val in all_unique_values:
            matches = envo_pattern.findall(val)
            for match in matches:
                all_codes_to_lookup.add(match.upper().replace('_', ':'))
        
        if not all_codes_to_lookup: return
        logger.info(f"Found {len(all_codes_to_lookup)} unique ENVO codes to look up.")
        
        tasks = [
            self._fetch_envo_label(session, code) for code in all_codes_to_lookup
        ]
        results = await asyncio.gather(*tasks)
        
        code_to_label_map = {code: label for code, label in results if label}

        # Apply replacement to all columns that were checked
        for col in existing_cols:
            self.df[col] = self.df[col].astype(str).replace(code_to_label_map,
                                                            regex=True)

    async def _fetch_envo_label(self, session, code) -> Tuple[str, Optional[str]]:
        url = "https://www.ebi.ac.uk/ols/api/ontologies/envo/terms"
        iri = f"http://purl.obolibrary.org/obo/{code.replace(':', '_')}"
        params = {'iri': iri}
        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    terms = data.get('_embedded', {}).get('terms', [])
                    if terms and 'label' in terms[0]:
                        return code, terms[0]['label']
        except Exception as e:
            logger.warning(f"OLS API request failed for {code}: {e}")
        return code, None

    async def _find_publications(
        self, session: aiohttp.ClientSession, api_key: Optional[str] = None
    ) -> None:
        if 'publication_doi' not in self.df.columns: self.df['publication_doi'] = np.nan
        
        acc_pattern = r'^(run|sample|exp|study|proj|sra|ena|ddbj|bio)_?(acc|alias)$|^acc$'
        all_acc_cols = [c for c in self.df.columns if re.search(acc_pattern, c, re.I)]
        if not all_acc_cols: return
        
        priority = ['project', 'study', 'biosample', 'sra', 'ena',
                    'sample', 'experiment', 'run']
        sorted_acc_cols = sorted(all_acc_cols,
            key=lambda c: next((i for i, p in enumerate(priority)
                                if p in c.lower()), len(priority)))
        
        self.df['search_accession'] = self.df[sorted_acc_cols].bfill(axis=1).iloc[:, 0]
        rows_to_search = self.df[self.df['publication_doi'].isnull() & self.df['search_accession'].notna()]
        if rows_to_search.empty: return

        unique_accessions = rows_to_search['search_accession'].unique()
        logger.info(f"Found {len(unique_accessions)} unique accessions to check for publications.")
        
        tasks = [
            self._fetch_single_doi(session, acc, api_key) for acc in unique_accessions
        ]
        results = await asyncio.gather(*tasks)

        accession_to_doi = {acc: doi for acc, doi in results if doi}
        
        if accession_to_doi:
            doi_map = self.df['search_accession'].map(accession_to_doi)
            self.df['publication_doi'].fillna(doi_map, inplace=True)
        
        self.df.drop(columns=['search_accession'], inplace=True)

    async def _fetch_single_doi(
        self, session, accession, api_key
    ) -> Tuple[str, Optional[str]]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        api_key_str = f"&api_key={api_key}" if api_key else ""
        
        async def get_xml(url):
            max_retries = 5 # Number of times to retry before giving up
            
            for attempt in range(max_retries):
                async with self.ncbi_semaphore:
                    async with self.ncbi_pacing_lock:
                        now = time.monotonic()
                        elapsed = now - self.last_ncbi_request_time
                        # BUG FIX: Use 10/sec limit w/ key, 3/sec limit w/o key
                        wait_time = (0.11 if api_key else 0.34) - elapsed
                        if wait_time > 0: await asyncio.sleep(wait_time)
                        self.last_ncbi_request_time = time.monotonic()

                    try:
                        async with session.get(url, timeout=60) as resp:
                            if resp.status == 429:
                                # Use the server's suggested wait time, or default to exponential backoff
                                retry_after = int(resp.headers.get("Retry-After", 2 * (attempt + 1)))
                                logger.warning(
                                    f"Rate limited for {accession}. Retrying in {retry_after} seconds..."
                                    f"(Attempt {attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(retry_after)
                                continue # Retry

                            resp.raise_for_status()
                            return ET.fromstring(await resp.text())

                    except Exception as e:
                        logger.warning(f"NCBI request for {accession} failed: {e}", exc_info=True)
                        # For other client errors, wait a bit before the next retry
                        await asyncio.sleep(1 * (attempt + 1))
                
            # If all retries fail, log it and return None
            logger.error(f"All {max_retries} retries failed for NCBI request: {accession}")
            return None

        pmids = []
        # Stage 1: Search formal links
        for db in ['bioproject', 'sra', 'biosample']:
            uid_root = await get_xml(f"{base_url}esearch.fcgi?db={db}&term={accession}&retmode=xml{api_key_str}")
            if uid_root is not None and (id_elem := uid_root.find('.//Id')) is not None and id_elem.text:
                uid = id_elem.text
                link_root = await get_xml(f"{base_url}elink.fcgi?dbfrom={db}&db=pubmed&id={uid}&retmode=xml{api_key_str}")
                if link_root is not None:
                    pmids = [id_elem.text for id_elem in link_root.findall(".//LinkSetDb[DbTo='pubmed']//Id") if id_elem.text]
                if pmids: break
        
        # Stage 2: If no links, search PubMed directly
        if not pmids:
            pm_root = await get_xml(f"{base_url}esearch.fcgi?db=pubmed&term={accession}[accn]&retmode=xml{api_key_str}")
            if pm_root is not None:
                pmids = [id_elem.text for id_elem in pm_root.findall('.//Id') if id_elem.text]

        # Get DOI from PMID
        if pmids:
            summary_root = await get_xml(f"{base_url}esummary.fcgi?db=pubmed&id={pmids[0]}&retmode=xml{api_key_str}")
            if summary_root is not None and (doi_elem := summary_root.find(".//Item[@Name='DOI']")) is not None and doi_elem.text:
                logger.info(f"SUCCESS: Found DOI {doi_elem.text} for {accession}")
                return accession, doi_elem.text
        
        return accession, None

    # ========================== EXPLORATORY & REPORTING =========================== #

    def suggest_categorical_mappings(
        self, similarity_threshold: int = 90, max_unique_values: int = 100
    ) -> Dict[str, Dict[str, str]]:
        """Analyzes categorical columns and suggests mappings for standardization."""
        logger.info("Generating suggestions for categorical value mappings...")
        suggested_mappings = {}
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        with get_progress_bar() as progress:
            task = progress.add_task("Analyzing columns", total=len(categorical_cols))
            for col in categorical_cols:
                try:
                    unique_count = self.df[col].nunique()
                    if not (2 <= unique_count <= max_unique_values): continue

                    value_counts = self.df[col].astype(str).str.lower(
                        ).str.strip().dropna().value_counts()
                    unique_values = value_counts.index.tolist()
                    
                    groups, processed_values = [], set()
                    for val in unique_values:
                        if val in processed_values: continue
                        current_group = {
                            o for o in unique_values
                            if fuzz.ratio(val, o) >= similarity_threshold
                        }
                        groups.append(list(current_group))
                        processed_values.update(current_group)

                    col_mapping = {}
                    for group in groups:
                        if len(group) > 1:
                            canonical_val = max(group,
                                                key=lambda v: value_counts.get(v, 0))
                            for val in group:
                                if val != canonical_val: col_mapping[val] = canonical_val
                    
                    if col_mapping: suggested_mappings[col] = col_mapping
                except Exception as e: logger.error(f"Error analyzing column '{col}': {e}", exc_info=True)
                finally: progress.update(task, advance=1)
                
        return suggested_mappings

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Returns a comprehensive report of all cleaning actions performed."""
        self.report['final_shape'] = self.df.shape
        self.report['summary'] = {
            'rows_initial': self.initial_shape[0],
            'cols_initial': self.initial_shape[1],
            'rows_final': self.df.shape[0],
            'cols_final': self.df.shape[1],
            'rows_removed': self.initial_shape[0] - self.df.shape[0],
            'cols_removed': self.initial_shape[1] - self.df.shape[1],
            'duplicate_rows_removed': self.report.get('duplicate_rows_removed', 0)
        }
        return self.report

# ================================== EXECUTOR FUNCTION ==================================== #

async def process_metadata(
    df: pd.DataFrame, output_path: Union[str, Path], config: Optional[AppConfig] = None
) -> pd.DataFrame:
    """High-level async executor function to run the full metadata processing pipeline."""
    if config is None: config = AppConfig() # type: ignore

    try:
        manager = MetadataManager(metadata=df, config=config)
        cleaned_df = await manager.run_pipeline()

        if not cleaned_df.empty:
            MetadataManager.export_tsv(cleaned_df, output_path)
        else:
            logger.warning("Pipeline resulted in an empty DataFrame. No file was saved.")
            return df

        # --- NEW REPORTING LOGIC ---
        report = manager.get_cleaning_report()
        
        # Define the path for the report file
        output_path = Path(output_path)
        report_path = output_path.parent / f"{output_path.stem}_cleaning_report.json"
        
        # Save the report to a JSON file
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Log a single, clean message pointing to the report
        logger.info(f"Metadata cleaning complete. A detailed report was saved to: {report_path}")
        # --- END NEW LOGIC ---
        
        return cleaned_df
    
    except Exception as e:
        logger.error(
            f"An error occurred during the metadata processing workflow: {e}", exc_info=True
        )
        return df # Return original dataframe on failure
        
        
def import_tsv(metadata_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(metadata_path, sep='\t', low_memory=False)


def export_tsv(metadata: pd.DataFrame, output_path: Union[str, Path]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, sep='\t', index=True)
    
    
def standardize_lat_lon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds and renames latitude/longitude columns to 'lat' and 'lon'.

    Searches for common variations and renames the first match found for each.
    The search order is: exact match, 'latitude'/'longitude', variations.
    """
    # Create a copy to avoid modifying the original DataFrame in place
    df = df.copy()
    
    # --- Standardize Latitude ---
    if 'lat' not in df.columns:
        lat_found = False
        # Define search patterns for latitude
        lat_patterns = [
            (r'^latitude$', 'exact'),
            (r'.*_lat$', 'suffix'),
            (r'^latitude_.*', 'prefix')
        ]
        for col in df.columns:
            for pattern, _ in lat_patterns:
                if re.match(pattern, col, re.IGNORECASE):
                    logger.info(f"Found latitude-like column '{col}'. Renaming to 'lat'.")
                    df.rename(columns={col: 'lat'}, inplace=True)
                    lat_found = True
                    break
            if lat_found:
                break

    # --- Standardize Longitude ---
    if 'lon' not in df.columns:
        lon_found = False
        # Define search patterns for longitude
        lon_patterns = [
            (r'^longitude$', 'exact'),
            (r'^lon$', 'exact'), # Handles 'lon ' with trailing space
            (r'.*_lon$', 'suffix'),
            (r'^longitude_.*', 'prefix')
        ]
        for col in df.columns:
            for pattern, _ in lon_patterns:
                # Use strip() to handle names like 'lon '
                if re.match(pattern, col.strip(), re.IGNORECASE):
                    logger.info(f"Found longitude-like column '{col}'. Renaming to 'lon'.")
                    df.rename(columns={col: 'lon'}, inplace=True)
                    lon_found = True
                    break
            if lon_found:
                break
                
    return df

# ==============================================================================
# STANDALONE HELPER FUNCTIONS (Required by Ingestion)
# ==============================================================================

import scanpy as sc
import anndata as ad
import numpy as np

def filter_samples_and_features(
    adata: ad.AnnData, 
    min_counts_per_sample: Any = 100, 
    min_counts_per_feature: int = 2,
    min_cells_per_feature: int = 1,
    *args, **kwargs
) -> ad.AnnData:
    """
    Filters low-quality samples and rare features (ASVs).
    Handles cases where 'min_counts_per_sample' receives an AppConfig object.
    """
    if adata is None:
        return None

    # --- 1. Resolve Argument Types ---
    # If the second argument is not an int, it's likely the AppConfig object passed by ingestion.py
    target_min_sample = 100 # Default fallback
    
    if isinstance(min_counts_per_sample, (int, float)):
        target_min_sample = int(min_counts_per_sample)
    elif hasattr(min_counts_per_sample, 'preprocessing'): 
        # Extract value from AppConfig object
        try:
            target_min_sample = int(min_counts_per_sample.preprocessing.filter.min_sequencing_depth)
            min_counts_per_feature = int(min_counts_per_sample.preprocessing.filter.min_counts_feature)
            logger.info(f"Extracted min_sequencing_depth={target_min_sample} from passed Config object.")
        except Exception as e:
            logger.warning(f"Could not extract filtering param from config: {e}. Using default {target_min_sample}.")
    else:
        logger.warning(f"Received invalid type {type(min_counts_per_sample)} for min_counts. Using default {target_min_sample}.")

    logger.info(f"Filtering: min_counts_sample={target_min_sample}, min_counts_feature={min_counts_per_feature}, min_cells_feature={min_cells_per_feature}")
    
    # --- 2. Apply Filters ---
    try:
        # Filter Samples (Rows)
        sc.pp.filter_cells(adata, min_counts=target_min_sample)
        
        # Filter Features (Columns)
        sc.pp.filter_genes(adata, min_counts=min_counts_per_feature)
        sc.pp.filter_genes(adata, min_cells=min_cells_per_feature)
        
        logger.info(f"Filtered data shape: {adata.shape}")
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        
    return adata

def clean_metadata(adata: ad.AnnData, *args, **kwargs) -> ad.AnnData:
    """Standardizes metadata in .obs (bytes->str, strip whitespace, unify NaNs)."""
    logger.info("Cleaning metadata...")
    adata.obs.index = adata.obs.index.astype(str)
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object' or isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            if len(adata.obs) > 0 and isinstance(adata.obs[col].iloc[0], bytes):
                 adata.obs[col] = adata.obs[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
            series = adata.obs[col].astype(str).str.strip()
            adata.obs[col] = series.replace(['nan', 'NaN', 'None', '<NA>', '', 'NoneType'], np.nan)
    return adata

def parse_taxonomy(adata: ad.AnnData, taxonomy_col: str = 'Taxon') -> ad.AnnData:
    """Parses a taxonomy string column into separate rank columns."""
    if taxonomy_col not in adata.var.columns:
        # Case-insensitive fallback
        for c in adata.var.columns:
            if c.lower() == taxonomy_col.lower():
                taxonomy_col = c
                break
        else:
            logger.warning(f"Taxonomy column '{taxonomy_col}' not found. Skipping parse.")
            return adata

    logger.info(f"Parsing taxonomy from column '{taxonomy_col}'...")
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    try:
        tax_series = adata.var[taxonomy_col].astype(str)
        tax_df = tax_series.str.split(';', expand=True)
        tax_df = tax_df.apply(lambda x: x.str.strip())
        for i, rank in enumerate(ranks):
            if i < tax_df.shape[1]:
                adata.var[rank] = tax_df[i].replace(['', 'None', 'nan', '<NA>', 'NoneType'], 'Unassigned')
    except Exception as e:
        logger.error(f"Failed to parse taxonomy: {e}")
    return adata

def validate_metadata(adata: ad.AnnData, config=None) -> ad.AnnData:
    """Wrapper to clean metadata and ensure priority columns exist."""
    adata = clean_metadata(adata)
    required = ['latitude', 'longitude', 'facility_match']
    for col in required:
        if col not in adata.obs.columns:
            adata.obs[col] = False if col == 'facility_match' else np.nan
    return adata
