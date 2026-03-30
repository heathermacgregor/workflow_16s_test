# workflow_16s/metadata/manager.py

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from workflow_16s.config import AppConfig
from workflow_16s.constants import SAMPLE_ID_COLUMN
from workflow_16s.utils.logger import get_logger, with_logger
from workflow_16s.utils.progress import get_progress_bar
from .constants import (
    DEFAULT_COLUMN_MAPPINGS, DEFAULT_CONVERSIONS, DEFAULT_COORDINATE_SOURCES,
    DEFAULT_MEASUREMENT_STANDARDS, DEFAULT_UNIT_PATTERNS, ONTOLOGY_MAP,
    PH_PATTERN
)
from .enrichment import MetadataEnricher

# Import ENA enrichment pipeline
try:
    from workflow_16s.api.sequence.ena import ENAEnrichmentPipeline
    HAS_ENA_PIPELINE = True
except ImportError:
    HAS_ENA_PIPELINE = False

@with_logger
class MetadataManager:
    """A unified class to handle the cleaning, processing, and enrichment of metadata.

    This class provides a complete, modular pipeline for:
    1.  **Loading and Saving Data**: Handles TSV file I/O.
    2.  **Cleaning**: Standardizes formats, handles duplicates, converts types,
        and collapses redundant columns.
    3.  **Processing**: Extracts key information like geolocation and infers
        ontologies from unstructured text.
    4.  **Enrichment**: Delegates to MetadataEnricher for external API calls.

    The primary entry point is the `run_pipeline()` method, which executes these
    steps in a logical order.
    """

    NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+') # Re-compile for self
    PH_PATTERN = PH_PATTERN
    DEFAULT_COORDINATE_SOURCES = DEFAULT_COORDINATE_SOURCES
    DEFAULT_COLUMN_MAPPINGS = DEFAULT_COLUMN_MAPPINGS
    DEFAULT_UNIT_PATTERNS = DEFAULT_UNIT_PATTERNS
    DEFAULT_CONVERSIONS = DEFAULT_CONVERSIONS
    DEFAULT_MEASUREMENT_STANDARDS = DEFAULT_MEASUREMENT_STANDARDS
    ONTOLOGY_MAP = ONTOLOGY_MAP
    
    CORE_COLUMNS = ['run_accession', 'sample_accession', 'lat', 'lon', 'collection_date']
    
    TARGET_SCHEMA = {
        "ph": ["ph", "ph_level", "soil_ph", "water_ph", "ph_sensor"],
        "temperature": ["temp", "temperature_c", "temp_c", "water_temp", "soil_temp", "air_temp"],
        "env_type": ["environment", "sample_type", "water_body", "biome"],
        "salinity": ["sal", "salinity_ppt", "salinity_psu", "salt_concentration", "conductivity"],
        "depth": ["depth", "depth_m", "sampling_depth", "water_depth", "altitude_m"],
        "oxygen": ["do", "dissolved_oxygen", "o2_concentration", "oxygen_saturation"],
        "host_age": ["age", "host_age", "subject_age", "age_years"],
        "host_sex": ["sex", "gender", "host_sex"]
    }


    def __init__(
        self, metadata: pd.DataFrame, config: AppConfig,
        sample_id_column: str = SAMPLE_ID_COLUMN
    ):
        if metadata.empty: raise ValueError("Cannot process an empty metadata DataFrame.")
        
        self.logger = get_logger("workflow_16s")
        
        self.config = config
        self.sample_id_column = self.config.metadata.columns.sample_id or sample_id_column
        self.ncbi_api_key = self.config.credentials.ncbi_api_key or None
        self.df = metadata.copy()
        self.initial_shape = self.df.shape
        self.original_df_for_enrichment: Optional[pd.DataFrame] = None
        
        self.report: Dict[str, Any] = {
            'initial_shape': self.initial_shape, 'actions': [],
            'columns_dropped': {'unwanted': [], 'duplicate': [], 'merged': []},
            'numeric_coercions': {}, 'categorical_standardizations': {},
            'unit_standardizations': {}
        }
        
        self.logger.info(f"Initialized MetadataManager with shape {self.df.shape}.")
        
    def harmonize(self, similarity_threshold: int = 85) -> pd.DataFrame:
        """
        Collapses sparse, chaotic ENA metadata into a dense standard schema.
        Uses a mix of exact alias matching and fuzzy string distance.
        """
        self.logger.info(f"🧬 Harmonizing {len(self.df.columns)} raw columns into standard schema...")
        
        # Start with the core columns you already know are good
        core_cols = self.CORE_COLUMNS
        harmonized_df = self.df[self.df.columns.intersection(core_cols)].copy()

        # Iterate through our desired standard fields
        for standard_key, aliases in self.TARGET_SCHEMA.items():
            found_data = pd.Series(index=self.df.index, dtype=object)
            matched_raw_cols = []

            for raw_col in self.df.columns:
                raw_col_lower = raw_col.lower().strip()
                
                # 1. Exact Match in Synonym Ring
                is_alias = raw_col_lower in aliases
                
                # 2. Fuzzy Match (Catches typos or variations)
                fuzzy_score = fuzz.ratio(raw_col_lower, standard_key)
                is_fuzzy = fuzzy_score >= similarity_threshold

                if is_alias or is_fuzzy:
                    # Combine existing found data with the new column (filling NaNs)
                    found_data = found_data.combine_first(self.df[raw_col])
                    matched_raw_cols.append(raw_col)

            if not found_data.isna().all():
                harmonized_df[standard_key] = found_data
                self.logger.debug(f"✅ Harmonized '{standard_key}' from: {matched_raw_cols}")

        for col in self.df.columns:
            if col not in matched_raw_cols and col not in harmonized_df.columns:
                if self.df[col].notna().mean() > 0.01:  # Keep columns that are at least 1% complete
                    harmonized_df[col] = self.df[col]

        self.logger.info(f"✅ Final harmonized metadata shape: {harmonized_df.shape}")
        return harmonized_df

    async def run_pipeline(self) -> pd.DataFrame:
        """
        Executes the full cleaning, processing, and enrichment pipeline.

        Pipeline stages:
        1. **Cleaning**: Standardizes formats, handles duplicates, converts types
        2. **Processing**: Extracts geolocation, infers ontologies, standardizes dates
        3. **Enrichment**:
           - External API calls (geocoding, ENVO codes, publications)
           - ENA/SRA metadata enrichment (location, collection dates from sequence archives)

        Returns:
            A fully cleaned, processed, and enriched metadata DataFrame.
        """
        self.logger.info("[!] Starting metadata processing pipeline...")
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)

        # Core cleaning and standardization (Synchronous)
        self._run_cleaning_steps()

        # Data extraction and ontology Inference (Synchronous)
        self._run_processing_steps()
        if self.df.empty:
            self.logger.warning("DataFrame is empty after processing steps. Returning original DataFrame.")
            return self.df

        # 3. External Data Enrichment (Asynchronous)
        await self._run_enrichment_steps()

        self.logger.info(f"[X] Metadata processing pipeline complete. Final shape: {self.df.shape}")
        return self.df.copy()

    def _run_cleaning_steps(self) -> None:
        """Executes foundational cleaning tasks."""
        self.logger.info("--- Running Stage 1: Cleaning and Standardization ---")
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
        self.logger.info("--- Running Stage 2: Processing and Inference ---")
        
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
        """
        Delegates tasks that enrich data using external sources to MetadataEnricher.
        Also enriches ENA metadata (sample accessions with location/date information).
        """
        self.logger.info("--- Running Stage 3: Enrichment (Async) ---")
        async with aiohttp.ClientSession() as session:
            # Instantiate the enricher and pass it the session and API key
            enricher = MetadataEnricher(
                session=session,
                ncbi_api_key=self.ncbi_api_key
            )

            # Delegate the enrichment tasks, passing the DataFrame
            await enricher.enrich_location_from_coords(self.df)
            await enricher.convert_envo_codes(self.df)
            await enricher.find_publications(self.df)

        # Add ENA metadata enrichment (location, dates from ENA/SRA)
        await self._run_ena_enrichment()

    async def _run_ena_enrichment(self) -> None:
        """
        Enrich samples with ENA/SRA metadata (location, collection dates).

        This step:
        1. Checks if ENA enrichment is enabled in config
        2. Verifies email credential is available
        3. Enriches samples with location and date information
        4. Gracefully handles errors and missing data
        """
        # Check if ENA enrichment is enabled
        if not self.config.apis.enabled or not self.config.apis.sequence.ena.enabled:
            self.logger.debug("ENA enrichment disabled in config")
            return

        # Check if we have email configured
        ena_email = self.config.credentials.ena_email or self.config.credentials.email
        if not ena_email:
            self.logger.warning(
                "ENA enrichment requires ena_email or email credential. Skipping ENA enrichment."
            )
            return

        try:
            from workflow_16s.api.sequence.ena import ENAEnrichmentPipeline

            self.logger.info("Starting ENA metadata enrichment...")

            async with ENAEnrichmentPipeline(self.config) as pipeline:
                enriched_df = await pipeline.enrich_samples(self.df)

                # Merge enriched columns with existing DataFrame
                # Only add new columns or fill missing values
                for col in enriched_df.columns:
                    if col not in ['run_accession', 'sample_accession', '#sampleid', 'sample_id']:
                        # Only add column if it doesn't exist or if we're filling missing values
                        if col not in self.df.columns:
                            self.df[col] = enriched_df[col]
                        else:
                            # Fill missing values in existing column from enriched data
                            mask = self.df[col].isna()
                            if mask.any():
                                self.df.loc[mask, col] = enriched_df.loc[mask, col]

            self.logger.info("✅ ENA enrichment completed successfully")

        except ImportError as e:
            self.logger.warning(f"ENA enrichment module not available: {e}")
        except Exception as e:
            self.logger.warning(f"ENA enrichment encountered an error (continuing pipeline): {e}", exc_info=True)

    def _execute_steps(self, steps: List[Tuple[str, Callable]]) -> None:
        """Generic step executor to run a list of synchronous functions."""
        for name, func in steps:
            try:
                func()
                self.report['actions'].append(name)
            except Exception as e:
                self.logger.error(f"Error during '{name}': {e}", exc_info=True)
                raise

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
        get_logger("workflow_16s").info(f"Metadata successfully exported to {output_path}")

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
                    self.logger.error(f"Failed to load metadata file {path}: {e!r}")
                finally: progress.update(task, advance=1)

        if not dfs:
            raise FileNotFoundError("No valid metadata files could be loaded.")
        return pd.concat(dfs, ignore_index=True)

    def _drop_unwanted_columns(self) -> None:
        cols_to_drop = self.config.metadata.columns_to_drop or []
        existing_cols_to_drop = list(set(cols_to_drop) & set(self.df.columns))
        if existing_cols_to_drop:
            self.df.drop(columns=existing_cols_to_drop, inplace=True)
            self.report['columns_dropped']['unwanted'] = existing_cols_to_drop
            self.logger.info(f"Dropped {len(existing_cols_to_drop)} unwanted columns.")

    def _clean_duplicate_columns(self) -> None:
        if self.df.columns.duplicated().any():
            duplicated_cols = self.df.columns[self.df.columns.duplicated()].unique().tolist()
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            self.report['columns_dropped']['duplicate'] = duplicated_cols
            self.logger.warning(f"Removed duplicate columns: {duplicated_cols}")

    def _clean_sample_ids(self) -> None:
        if self.sample_id_column not in self.df.columns:
            alternatives = ['#sampleid', 'sample_id', 'sample id',
                            'sample name', 'run_accession']
            found_col = next((
                alt for alt in alternatives if alt in self.df.columns
            ), None)
            if found_col:
                self.logger.warning(f"'{self.sample_id_column}' not found. Creating it from '{found_col}'.")
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
            self.logger.warning(f"Removed {removed_count} rows with duplicate or missing sample IDs.")

    def _clean_numeric_columns(self) -> None:
        numeric_cols = self.config.metadata.force_numeric_columns or []
        for col in numeric_cols:
            if col in self.df.columns and self.df[col].dtype == 'object':
                initial_nans = self.df[col].isna().sum()
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                coerced_count = self.df[col].isna().sum() - initial_nans
                if coerced_count > 0:
                    self.report['numeric_coercions'][col] = coerced_count
                    self.logger.debug(f"Coerced {coerced_count} values to NaN in '{col}'.")

    def _standardize_categorical_values(self) -> None:
        mappings = self.config.metadata.mappings or {}
        for col, value_map in mappings.items():
            if col in self.df.columns:
                cleaned_series = self.df[col].astype(str).str.lower().str.strip()
                replaced_series = cleaned_series.replace(value_map)
                if not cleaned_series.equals(replaced_series):
                    self.df[col] = replaced_series
                    self.report['categorical_standardizations'][col] = value_map
                    self.logger.debug(f"Standardized values in column '{col}'.")

    def _apply_custom_filters(self) -> None:
        if 'empo_3' in self.df.columns:
            initial_rows = len(self.df)
            values_to_remove = ['animal distal gut', 'animal corpus', 'animal secretion']
            mask = self.df['empo_3'].astype(str).str.lower().isin(values_to_remove)
            self.df = self.df[~mask]
            rows_removed = initial_rows - len(self.df)
            if rows_removed > 0:
                self.logger.info(f"Filtered {rows_removed} rows based on 'empo_3' values.")
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
                measurement_key = next((
                    key for key in self.DEFAULT_MEASUREMENT_STANDARDS
                    if key in base_name_raw
                ), base_name_raw)
                column_groups.setdefault(measurement_key, []).append((col, unit))

        for base_name, cols_with_units in column_groups.items():
            if len(cols_with_units) < 2: continue
            
            target_unit = self.DEFAULT_MEASUREMENT_STANDARDS.get(base_name)
            if not target_unit: continue
            
            target_col_name = f"{base_name}_{target_unit}"
            self.logger.info(f"Merging {[c[0] for c in cols_with_units]} into '{target_col_name}'")
            
            merged_series = pd.Series(np.nan, index=self.df.index, dtype=float)
            for col_name, unit in cols_with_units:
                source_series = pd.to_numeric(self.df[col_name], errors='coerce')
                if unit == target_unit:
                    converted_series = source_series
                elif unit in self.DEFAULT_CONVERSIONS and self.DEFAULT_CONVERSIONS[unit][0] == target_unit:
                    conversion_func = self.DEFAULT_CONVERSIONS[unit][1]
                    converted_series = conversion_func(source_series)
                else:
                    self.logger.warning(f"Cannot convert '{unit}' to '{target_unit}' for '{col_name}'. Skipping.")
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
            # --- Identify columns to parse for text coordinates ---
            pair_sources = [
                c for c in self.DEFAULT_COORDINATE_SOURCES['pairs'] if c in self.df.columns
            ]
            
            # 1. If 'location' exists, definitely check it
            if 'location' in self.df.columns and 'location' not in pair_sources:
                pair_sources.append('location')

            # 2. If 'lat' exists but is full of NaNs (because it contained text), check it too
            if 'lat' in self.df.columns and lat.isna().all():
                if 'lat' not in pair_sources:
                    pair_sources.append('lat')
                    
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
        self.logger.info(f"Geolocation: {initial_count} initial -> {len(self.df)}"
                    f" valid. ({dropped_count} dropped).")

    def _extract_coords_from_string(
        self, s: str
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(s, str): return None, None
        
        # 1. Robust "Decimal Direction" pattern with Scientific Notation support
        dd_dir_regex = r'([\d\.-]+(?:[eE][-+]?\d+)?)\s*([NS])\s*([\d\.-]+(?:[eE][-+]?\d+)?)\s*([EW])'
        
        match = re.search(dd_dir_regex, s, re.IGNORECASE)
        if match:
            try:
                lat = float(match.group(1))
                if match.group(2).upper() == 'S': lat *= -1
                
                lon = float(match.group(3))
                if match.group(4).upper() == 'W': lon *= -1
                
                return lat, lon
            except ValueError: pass

        # 2. Try standard comma-separated Decimal Degrees (e.g. "40.1, -74.2")
        dd_regex = r'([-+]?[1-8]?\d(?:\.\d+)?|[-+]?90(?:\.0+)?),\s*([-+]?180(?:\.0+)?|[-+]?(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)'
        match = re.search(dd_regex, s)
        if match:
            try: return float(match.group(1)), float(match.group(2))
            except (ValueError, IndexError): pass
        
        # 3. Try standard DMS with symbols (e.g. "40° 12' N")
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
                self.logger.debug(f"Inferred ontology for '{term_category}'.")

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

    def suggest_categorical_mappings(
        self, similarity_threshold: int = 90, max_unique_values: int = 100
    ) -> Dict[str, Dict[str, str]]:
        """Analyzes categorical columns and suggests mappings for standardization."""
        self.logger.info("Generating suggestions for categorical value mappings...")
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
                except Exception as e:
                    self.logger.error(f"Error analyzing column '{col}': {e}", exc_info=True)
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


@with_logger
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
            manager.logger.warning("Pipeline resulted in an empty DataFrame. No file was saved.")
            return df

        report = manager.get_cleaning_report()
        output_path = Path(output_path)
        report_path = output_path.parent / f"{output_path.stem}_cleaning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        manager.logger.info(f"Metadata cleaning complete. A detailed report was saved to: {report_path}")
        return cleaned_df
    
    except Exception as e:
        manager.logger.error(
            f"An error occurred during the metadata processing workflow: {e}", exc_info=True
        )
        return df # Return original dataframe on failure
