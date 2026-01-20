# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
import numpy as np
import re
from biom import load_table
from biom.table import Table
import h5py

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level, _ProcessingMixin
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.utils.io import (
    export_h5py
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import load_nfc_facilities, match_facilities_to_samples

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

def collapse_ena_columns(df, ena_suffix='_ena'):
    # Create a copy of the DataFrame to avoid modifying the original
    new_df = df.copy()
    
    # Identify all columns ending with the suffix
    ena_columns = [col for col in new_df.columns if col.endswith(ena_suffix)]
    
    # Sort columns by length in descending order to handle nested suffixes
    ena_columns_sorted = sorted(ena_columns, key=len, reverse=True)
    
    for ena_col in ena_columns_sorted:
        # Skip if the column is exactly the suffix (e.g., '_ena')
        if ena_col == ena_suffix:
            continue
        
        # Determine the base column name by removing the suffix
        base_col = ena_col[:-len(ena_suffix)]
        
        if base_col in new_df.columns:
            # Combine values: prioritize base_col, fill missing from ena_col
            new_df[base_col] = new_df[base_col].combine_first(new_df[ena_col])
        else:
            # Create base_col from ena_col if it doesn't exist
            new_df[base_col] = new_df[ena_col]
        
        # Drop the ena_col after processing
        new_df = new_df.drop(columns=[ena_col])
    
    return new_df

def collapse_ph_columns(df):
    """
    Collapses all columns in the dataframe that start with 'ph' followed by a non-alphabet character or exactly 'ph'
    into a single 'ph' column. The first non-null value from these columns is retained for each row.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with 'ph' columns collapsed into a single 'ph' column
    """
    # Compile regex pattern to match columns starting with 'ph' followed by non-alphabet or exactly 'ph'
    pattern = re.compile(r'^ph[^a-zA-Z]|^ph$')
    ph_columns = [col for col in df.columns if pattern.match(col)]
    
    # Return original dataframe if no ph_columns found
    if not ph_columns:
        return df
    
    # Prioritize exact 'ph' column if present
    if 'ph' in ph_columns:
        ph_columns.remove('ph')
        ph_columns = ['ph'] + sorted(ph_columns)
    else:
        ph_columns = sorted(ph_columns)
    
    # Create a temporary DataFrame with the selected columns
    temp_df = df[ph_columns]
    
    # Backfill values along rows and take the first column to get the first non-null value
    new_ph = temp_df.bfill(axis=1).iloc[:, 0]
    
    # Drop original ph_columns and add the new coalesced column
    df = df.drop(columns=ph_columns)
    df['ph'] = new_ph
    
    return df

def check_coordinate_completeness(df):
    """
    Check completeness of latitude and longitude coordinates in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to check
        
    Returns:
        dict: Dictionary containing:
            - 'total_rows': Total number of rows in DataFrame
            - 'complete_coordinates': Count of rows with both latitude and longitude
            - 'missing_coordinates': Count of rows missing at least one coordinate
            - 'missing_latitude': Count of rows missing latitude
            - 'missing_longitude': Count of rows missing longitude
            - 'completeness_percentage': Percentage of rows with complete coordinates
    """
    # Initialize result dictionary
    result = {
        'total_rows': len(df),
        'complete_coordinates': 0,
        'missing_coordinates': 0,
        'missing_latitude': 0,
        'missing_longitude': 0,
        'completeness_percentage': 0.0
    }
    
    # Check if required columns exist
    has_lat = 'latitude_deg' in df.columns
    has_lon = 'longitude_deg' in df.columns
    
    if not (has_lat and has_lon):
        result['missing_coordinates'] = len(df)
        return result
    
    # Calculate completeness metrics
    lat_missing = df['latitude_deg'].isna()
    lon_missing = df['longitude_deg'].isna()
    
    result['complete_coordinates'] = ((~lat_missing) & (~lon_missing)).sum()
    result['missing_coordinates'] = (lat_missing | lon_missing).sum()
    result['missing_latitude'] = lat_missing.sum()
    result['missing_longitude'] = lon_missing.sum()
    
    # Calculate completeness percentage
    if result['total_rows'] > 0:
        result['completeness_percentage'] = round(
            (result['complete_coordinates'] / result['total_rows']) * 100, 2
        )
    
    return result

def print_columns_with_missing_location_data(
    df: pd.DataFrame,
    lon_col: str = 'longitude_deg',
    lat_col: str = 'latitude_deg'
) -> None:
    """
    Prints columns that contain data for rows missing both longitude and latitude values.
    
    Args:
        df: Input DataFrame
        lon_col: Name of longitude column (default: 'longitude_deg')
        lat_col: Name of latitude column (default: 'latitude_deg')
    """
    # Check if the required columns exist
    missing_cols = [col for col in [lon_col, lat_col] if col not in df.columns]
    if missing_cols:
        logger.info(f"Warning: Columns {missing_cols} not found in DataFrame. Skipping location-based analysis.")
        return
    # Identify rows missing both coordinates
    missing_loc_mask = df[lon_col].isna() & df[lat_col].isna()
    missing_loc_rows = df[missing_loc_mask]
    
    if missing_loc_rows.empty:
        logger.info("No rows missing both longitude and latitude values")
        return
    
    # Find columns with at least one non-null value in missing location rows
    cols_with_data = [
        col for col in df.columns
        if col not in [lon_col, lat_col] 
        and missing_loc_rows[col].notna().any()
    ]
    
    if not cols_with_data:
        logger.info("No additional data exists for rows without coordinates")
        return
    
    logger.info("Columns containing data for rows without coordinates:")
    logger.info("-" * 50)
    
    # Print column names and sample values
    for col in cols_with_data:
        # Get non-null values from the column
        sample_values = missing_loc_rows[col].dropna().unique()
        
        logger.info(f"{col}:")
        logger.info(f"  • Non-null count: {missing_loc_rows[col].notna().sum()}")
        logger.info(f"  • Sample values: {sample_values[:5]}... (Total unique: {len(sample_values)})")
        logger.info("-" * 50)

def import_metadata_tsv(
    tsv_path: Union[str, Path],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """Load and standardize a sample metadata TSV file.
    
    Args:
        tsv_path:       Path to metadata TSV file.
        column_renames: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Standardized metadata DataFrame.
    
    Raises:
        FileNotFoundError: If specified path doesn't exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")
    

    df = pd.read_csv(tsv_path, sep='\t')
  
    df.columns = df.columns.str.lower()

    sample_id_col = next((col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns), None)
    df['SAMPLE ID'] = (df[sample_id_col] if sample_id_col else [f"{tsv_path.parents[5].name}_x{i}" for i in range(1, len(df)+1)])

    dataset_id_col = next((col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns), None)
    df['DATASET ID'] = (df[dataset_id_col] if dataset_id_col else tsv_path.parents[5].name)
  
    for col in constants.DEFAULT_GROUP_COLUMNS:
        col_name = col.get('name')
        if col.get('type') == 'bool' and col_name and col_name not in df.columns:
            df[col_name] = False

    if column_renames is None:
        column_renames = []      
    for old, new in column_renames:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df

def import_merged_metadata_tsv(
    tsv_paths: List[Union[str, Path]],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """Merge multiple metadata files into a single DataFrame.
    
    Args:
        tsv_paths:      List of paths to metadata TSV files.
        column_renames: List of (old_name, new_name) tuples for column renaming.
        verbose:        Verbosity flag.
    
    Returns:
        Concatenated metadata DataFrame.
    
    Raises:
        FileNotFoundError: If no valid metadata files could be loaded.
    """
    dfs: List[pd.DataFrame] = []
    with get_progress_bar() as progress:
        task_desc = "Loading metadata files"
        task = progress.add_task(_format_task_desc(task_desc), total=len(tsv_paths))
        for tsv_path in tsv_paths:
            try:
                dfs.append(import_metadata_tsv(tsv_path, column_renames))
            except Exception as e:
                logger.error(f"Loading metadata failed for {tsv_path}: {e!r}")
            finally:
                progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError("No valid metadata files loaded. Check paths and file formats.")

    return pd.concat(dfs, ignore_index=True)
  
########################################################################################

def import_table_biom(
    biom_path: Union[str, Path], 
    as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """Load a BIOM table from file.
    
    Args:
        biom_path: Path to .biom file.
        as_type:   Output format ('table' or 'dataframe').
    
    Returns:
        BIOM Table object or pandas DataFrame.
    
    Raises:
        ValueError: For invalid 'as_type' values.
    """
    try:
        with h5py.File(biom_path) as f:
            table = Table.from_hdf5(f)
    except:
        table = load_table(biom_path)
        
    if as_type == 'table':
        return table
    elif as_type == 'dataframe':
        return table_to_df(table)
    else:
        raise ValueError(
            f"Invalid output type: {as_type}. Use 'table' or 'dataframe'"
        )


def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """Merge multiple BIOM tables into a single unified table.
    
    Args:
        biom_paths: List of paths to .biom files.
        as_type:    Output format ('table' or 'dataframe').
        verbose:    Verbosity flag.
    
    Returns:
        Merged BIOM Table or DataFrame.
    
    Raises:
        ValueError: If no valid tables are loaded.
    """
    tables: List[Table] = []
    with get_progress_bar() as progress:
        task_desc = "Loading feature tables"
        task = progress.add_task(_format_task_desc(task_desc), total=len(biom_paths))
        for path in biom_paths:
            try:
                tables.append(import_table_biom(path, 'table'))
            except Exception as e:
                logger.error(f"BIOM load failed for {path}: {e}")
            finally:
                progress.update(task, advance=1)

    if not tables:
        raise ValueError("No valid BIOM tables loaded")

    # ACTUALLY MERGE THE TABLES USING REDUCE
    merged_table = reduce(lambda t1, t2: t1.merge(t2), tables)
    
    return merged_table if as_type == 'table' else table_to_df(merged_table)

########################################################################################

# Precompile regex patterns for efficiency
NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
LETTER_PATTERN = re.compile(r'[NnSsEeWw]')

def extract_lat_lon(s):
    """
    Extract latitude and longitude from a string using precompiled regex patterns.
    
    Args:
        s (str): Input string containing coordinate information
        
    Returns:
        tuple: (latitude, longitude) as floats, or (None, None) if extraction fails
    """
    if not isinstance(s, str):
        return None, None
        
    # Find all number matches
    matches = list(NUM_PATTERN.finditer(s))
    if len(matches) < 2:
        return None, None
        
    try:
        num1 = float(matches[0].group())
        num2 = float(matches[1].group())
    except (ValueError, TypeError):
        return None, None
        
    # Check for directional letters near the numbers
    letters = []
    for match in matches[:2]:
        start, end = match.span()
        window = s[max(0, start-3):min(len(s), end+3)]
        letter_match = LETTER_PATTERN.search(window)
        letters.append(letter_match.group().upper() if letter_match else None)
    
    # Process coordinates based on directional letters
    coords = {}
    for i, (num, letter) in enumerate(zip([num1, num2], letters)):
        if not letter:
            continue
        if letter in ['N', 'S']:
            coords['lat'] = num if letter == 'N' else -num
        else:  # E or W
            coords['lon'] = num if letter == 'E' else -num
    
    # Return results based on what we found
    if 'lat' in coords and 'lon' in coords:
        return coords['lat'], coords['lon']
    if 'lat' in coords:
        return coords['lat'], num2
    if 'lon' in coords:
        return num1, coords['lon']
    
    return num1, num2

def fill_missing_coordinates(df):
    """
    Fill missing latitude/longitude values by extracting from alternative columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with location information
        
    Returns:
        pd.DataFrame: DataFrame with filled coordinates
    """
    # Ensure required columns exist
    if 'latitude_deg' not in df.columns:
        df['latitude_deg'] = np.nan
    if 'longitude_deg' not in df.columns:
        df['longitude_deg'] = np.nan
        
    # Define potential source columns
    lat_sources = ['lat', 'lat_study', 'latitude_deg_ena', 'latitude_deg.1']
    lon_sources = ['lon', 'lon_study', 'longitude_deg.1']
    pair_sources = [
        'lat_lon', 'location', 'location_ena', 'location_start', 'location_end',
        'location_start_study', 'location_end_study'
    ]
    
    # Filter to existing columns only
    existing_lat = [col for col in lat_sources if col in df.columns]
    existing_lon = [col for col in lon_sources if col in df.columns]
    existing_pair = [col for col in pair_sources if col in df.columns]
    
    # Identify rows needing processing
    missing_mask = df['latitude_deg'].isna() | df['longitude_deg'].isna()
    missing_count = missing_mask.sum()
    
    if not missing_count:
        return df  # Return early if nothing to process
    
    print(f"Processing {missing_count} rows with missing coordinates...")
    
    # Process missing rows
    for idx in df.index[missing_mask]:
        row = df.loc[idx]
        new_lat, new_lon = row['latitude_deg'], row['longitude_deg']
        
        # First try pair columns (contain both coordinates)
        for col in existing_pair:
            if pd.isna(new_lat) or pd.isna(new_lon):
                if pd.notna(row[col]) and row[col] != '':
                    lat_val, lon_val = extract_lat_lon(str(row[col]))
                    if pd.isna(new_lat) and lat_val is not None:
                        new_lat = lat_val
                    if pd.isna(new_lon) and lon_val is not None:
                        new_lon = lon_val
        
        # Then try single-value columns
        if pd.isna(new_lat):
            for col in existing_lat:
                if pd.notna(row[col]) and row[col] != '':
                    try:
                        new_lat = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
        
        if pd.isna(new_lon):
            for col in existing_lon:
                if pd.notna(row[col]) and row[col] != '':
                    try:
                        new_lon = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
        
        # Update the dataframe
        df.at[idx, 'latitude_deg'] = new_lat
        df.at[idx, 'longitude_deg'] = new_lon
    
    return df

def update_table_and_metadata(
    table: Table,
    metadata: pd.DataFrame,
    sample_col: str = constants.DEFAULT_META_ID_COLUMN
) -> Tuple[Table, pd.DataFrame]:
    """Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata:      Sample metadata DataFrame.
        sample_column: Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    # Handle empty metadata case
    if metadata.empty:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_col])
    
    norm_metadata = _normalize_metadata(metadata, sample_col)
    
    # Handle empty metadata after normalization
    if norm_metadata.empty:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_col])
    
    biom_mapping = _create_biom_id_mapping(table)
    
    shared_ids = [sid for sid in norm_metadata[sample_col] if sid in biom_mapping]
    
    # Handle case with no shared IDs
    if not shared_ids:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_col])
    
    filtered_metadata = norm_metadata[norm_metadata[sample_col].isin(shared_ids)]
    original_ids = [biom_mapping[sid] for sid in filtered_metadata[sample_col]]
    
    # Filter the BIOM table
    filtered_table = table.filter(original_ids, axis='sample', inplace=False)
    
    return filtered_table, filtered_metadata


def _normalize_metadata(metadata: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    """Normalize sample IDs and remove duplicates."""
    # Handle empty metadata
    if metadata.empty:
        return metadata
    
    # Validate sample column exists
    if sample_col not in metadata.columns:
        raise ValueError(f"Sample column '{sample_col}' not found in metadata")
    
    metadata[sample_col] = metadata[sample_col].astype(str).str.lower()
    return metadata.drop_duplicates(subset=[sample_col])


def _create_biom_id_mapping(table: Table) -> Dict[str, str]:
    """Create lowercase to original-case ID mapping for BIOM table samples."""
    # Handle empty table
    if table.is_empty():
        return {}
    
    mapping: Dict[str, str] = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = orig_id.lower()
        if lower_id in mapping:
            raise ValueError(
                f"Duplicate lowercase sample ID: '{lower_id}' "
                f"(from '{orig_id}' and '{mapping[lower_id]}')"
            )
        mapping[lower_id] = orig_id
    return mapping
  
########################################################################################

class DownstreamDataLoader:
    ModeConfig = {
        "asv": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    def __init__(
        self,
        config: Dict,
        mode: str,
        project_dir: SubDirs,
        existing_subsets: Any = None,
        verbose: bool = False
    ):
        self.config, self.project_dir, self.verbose = config, project_dir, verbose
        self.existing_subsets = existing_subsets

        self._load_nfc_facilities()
        
        self.tables: Dict = {'raw': {}}
        self.metadata: Dict = {'raw': {}}
        
        # Load the ASV feature table if the target subfragment is specified (so, not 'any')
        if not self.config["target_subfragment_mode"] == 'any':
            self._load_table_and_metadata('asv')
        # Load the taxonomically-assigned feature table at genus level     
        self._load_table_and_metadata('genus')
    
    def _load_table_and_metadata(self, table_level: str = 'genus') -> None:
        table_level, table_dir, _ = self.ModeConfig['genus']
        table, metadata = self._filter_and_align(
            self._load_table_biom(table_level, table_dir), 
            self._match_facilities_to_samples(self._load_metadata_df(table_level, table_dir))
        )
        self._log_results(table_level, table, metadata)
        self.tables['raw'][table_level], self.metadata['raw'][table_level] = table, metadata

    def _filter_and_align(self, table, metadata) -> Tuple:
        table, metadata = update_table_and_metadata(
            table, metadata, 
            self.config.get("metadata_id_column", constants.DEFAULT_META_ID_COLUMN)
        )
        # Log alignment results
        if table.is_empty() or metadata.empty:
            logger.warning(f"Alignment resulted in empty table for {table_level}")
        return table, metadata
        
    # BIOM FEATURE TABLE    
    def _load_table_biom(self, table_level, table_dir) -> Table:
        table_biom_paths = self._find_table_biom_paths(table_level, table_dir)  
        if not table_biom_paths:
            raise FileNotFoundError("No BIOM files found")
        return import_merged_table_biom(table_biom_paths, "table", self.verbose)

    def _find_table_biom_paths(self, table_level, table_dir) -> List[Path]:
        if self.existing_subsets == None:
            # Use wildcard for subfragment when in 'any' mode or not in 'genus' mode
            subfragment_part = (
              "*" if self.config["target_subfragment_mode"] == 'any' or table_level == 'asv'
              else self.config["target_subfragment_mode"]
            )
            pattern = "/".join([
                "*", "*", "*", subfragment_part, 
                "FWD_*_REV_*", table_dir, "feature-table.biom"
            ])
            globbed = glob.glob(
                str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
                recursive=True
            )
            if self.verbose:
                logger.info(f"Found {len(globbed)} feature tables")
            return [Path(p) for p in globbed]
        else:
            table_biom_paths = [paths[table_dir] for subset_id, paths in self.existing_subsets.items()]
            if self.verbose:
                logger.info(f"Found {len(table_biom_paths)} feature tables")
            return table_biom_paths

    # METADATA TSV
    def _load_metadata_df(self, table_level, table_dir) -> pd.DataFrame:
        metadata = import_merged_metadata_tsv(self._find_metadata_paths(table_level, table_dir), None)
        # Remove duplicate columns
        if metadata.columns.duplicated().any():
            duplicated_columns = metadata.columns[metadata.columns.duplicated()].tolist()
            logger.debug(
                f"Found duplicate columns in metadata: {duplicated_columns}. "
                "Removing duplicates."
            )
            metadata = metadata.loc[:, ~metadata.columns.duplicated()]
        # Attempt to fill in missing latitude/longitude
        metadata = fill_missing_coordinates(metadata)
        metadata = collapse_ph_columns(metadata)
        metadata = collapse_ena_columns(metadata)
        return metadata

    def _find_metadata_paths(self, table_level, table_dir) -> List[Path]:
        tsv_paths: List[Path] = []
        if self.existing_subsets == None:
            for biom_path in self._find_table_biom_paths(table_level, table_dir):
                dataset_dir = biom_path.parent if biom_path.is_file() else biom_path
                tail = dataset_dir.parts[-6:-1]
                tsv_path = Path(self.project_dir.metadata_per_dataset).joinpath(
                    *tail, "sample-metadata.tsv"
                )
                if tsv_path.exists():
                    tsv_paths.append(tsv_path)
            if self.verbose:
                logger.info(f"Found {len(tsv_paths)} metadata files")
            return tsv_paths
        else:
            metadata_paths = [paths["metadata"] 
                              for subset_id, paths in self.existing_subsets.items()]
            if self.verbose:
                (f"Found {len(metadata_paths)} metadata files")
            return metadata_paths

    # SPECIAL CASE: NFC FACILITIES
    def _load_nfc_facilities(self) -> None:
        # If enabled, find samples within a threshold distance from NFC facilities
        if self.config.get("nfc_facilities", {}).get("enabled", False):
            self.nfc_facilities = load_nfc_facilities(
                cfg=self.config,
                output_dir=self.project_dir.final
            )
        else:
            self.nfc_facilities = None

    def _match_facilities_to_samples(self, metadata) -> pd.DataFrame:
        if not self.config.get("nfc_facilities", {}).get("enabled", False):
            return metadata
        else:
            return match_facilities_to_samples(self.config, metadata, self.nfc_facilities)

    def _log_results(self, table_level, table, metadata) -> None:
        table_size = "Empty" if table.is_empty() else f"{table.shape[0]} features × {table.shape[1]} samples"
        meta_size = "Empty" if metadata.empty else f"{metadata.shape[0]} samples × {metadata.shape[1]} cols"
        
        logger.info(
            f"{'Loaded metadata:':<30}{meta_size}"
        )
        feature_type = "genera" if table_level == "genus" else "ASVs"
        logger.info(
            f"{'Loaded features:':<30}{table_size} {feature_type}"
        )
