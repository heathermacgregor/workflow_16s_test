# ===================================== IMPORTS ====================================== #

# Standard Imports
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import pandas as pd
import numpy as np
from tqdm import tqdm 

# Local Imports
from workflow_16s.constants import (
    DEFAULT_GROUP_COLUMNS, SET_SAMPLE_ID_COLUMN, SET_SAMPLE_ID_COLUMN, SET_DATASET_ID_COLUMN
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

fastq_columns = ['fastq_aspera', 'fastq_bytes', 'fastq_ftp', 'fastq_galaxy', 'fastq_md5']
bam_columns = ['bam_aspera', 'bam_bytes', 'bam_ftp', 'bam_galaxy', 'bam_md5']
cols_to_drop = fastq_columns + bam_columns
cols_to_rename = {}

def import_metadata_tsv(
    tsv_path: Union[str, Path],
    group_columns: List[Dict] = DEFAULT_GROUP_COLUMNS,
    columns_to_rename: Optional[List[Tuple[str, str]]] = None,
    set_sample_id_column: str = SET_SAMPLE_ID_COLUMN,
    set_dataset_id_column: str = SET_DATASET_ID_COLUMN
) -> pd.DataFrame:
    """Load and standardize a sample metadata TSV file.
    
    Args:
        tsv_path:          Path to metadata TSV file.
        group_columns:     [Placeholder]
        columns_to_rename: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Standardized metadata DataFrame.
    
    Raises:
        FileNotFoundError: If specified path doesn't exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")
    
    # Load DataFrame from TSV
    df = pd.read_csv(tsv_path, sep='\t')
  
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
            logger.debug(f"Dropped `{col}`")

    sample_id_col = next((col 
                          for col in ['#sampleid', 'run_accession', 'sample_id', 'sample id', 'sample-id'] 
                          if col in df.columns), None)
    df[set_sample_id_column] = (df[sample_id_col] 
                                if sample_id_col 
                                else [f"{tsv_path.parents[5].name}_x{i}" 
                                      for i in range(1, len(df)+1)])

    dataset_id_col = next((col 
                           for col in ['dataset_id', 'project_accession', 'dataset_name'] 
                           if col in df.columns), None)
    df[set_dataset_id_column] = (df[dataset_id_col] 
                        if dataset_id_col 
                        else tsv_path.parents[5].name)
  
    for col in group_columns:
        name = col.get('name')
        type = col.get('type')
        if type == 'bool' and name and name not in df.columns:
            df[name] = False
            logger.debug(f"Set `{name}` to FALSE")
        if type == 'bool' and name and name in df.columns:
            df[name] = df[name].fillna(False)
            logger.debug(f"Set NaN values in `{name}` to FALSE")

    if columns_to_rename is not None:
        for old, new in columns_to_rename:
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
                logger.debug(f"Renamed `{old}` to `{new}`")

    return df


def import_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(tsv_path, sep='\t')

    
def export_tsv(metadata: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Export a sample metadata DataFrame to a TSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, sep='\t', index=True)

# ==================================================================================== #

def get_group_column_values(
    group_column: Union[str, Dict], 
    metadata: pd.DataFrame
) -> List[Any]:
    """Extract values from group column.
    
    Args:
        group_column: [Placeholder]
        metadata:     [Placeholder]
        
    Returns:
        List [Placeholder]
    """
    if isinstance(group_column, dict):
        if 'values' in group_column and group_column['values']:
            return group_column['values']
        if 'type' in group_column and group_column['type'] == 'bool':
            return [True, False]
        if 'name' in group_column and group_column['name'] in metadata.columns:
            return metadata[group_column['name']].drop_duplicates().tolist()
    elif isinstance(group_column, str):
        if group_column in metadata.columns:
            return metadata[group_column].drop_duplicates().tolist()
    return []


def import_merged_metadata_tsv(
    tsv_paths: List[Union[str, Path]],
    columns_to_rename: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """Merge multiple metadata files into a single DataFrame.
    
    Args:
        tsv_paths:         List of paths to metadata TSV files.
        columns_to_rename: List of (old_name, new_name) tuples for column renaming.
    
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
                dfs.append(import_metadata_tsv(tsv_path, group_columns=DEFAULT_GROUP_COLUMNS, columns_to_rename=columns_to_rename))
            except Exception as e:
                logger.error(f"Loading metadata failed for {tsv_path}: {e!r}")
            finally:
                progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError("No valid metadata files loaded. Check paths and file formats.")

    return pd.concat(dfs, ignore_index=True)

# ==================================================================================== #

class MetadataCleaner:
    """
    A comprehensive metadata cleaning utility for biological/environmental datasets.
    
    This class handles common metadata cleaning tasks including:
    - Removing duplicate columns and rows
    - Normalizing sample IDs
    - Collapsing redundant columns with different suffixes/prefixes
    - Extracting and standardizing coordinate information
    - Consolidating pH measurements from multiple columns
    
    Example:
        config = {"metadata_id_column": "sample_id"}
        cleaner = MetadataCleaner(config, metadata_df)
        cleaned_df = cleaner.run_all()
    """
    
    # Precompile regex patterns for efficiency
    NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
    LETTER_PATTERN = re.compile(r'[NnSsEeWw]')
    PH_PATTERN = re.compile(r'^ph[^a-zA-Z]|^ph$')
    
    # Define coordinate source columns with priority order (higher index = higher priority)
    COORDINATE_SOURCES = {
        'lat': ['lat_study', 'latitude_deg_ena', 'latitude_deg.1', 'lat'],
        'lon': ['lon_study', 'longitude_deg.1', 'lon'], 
        'pairs': ['location_ena', 'location_start', 'location_end', 
                 'location_start_study', 'location_end_study', 'lat_lon', 'location']
    }
    
    # Default column mappings for standardization
    DEFAULT_COLUMN_MAPPINGS = {
        'env_biome': 'environment_biome',
        'env_feature': 'environment_feature', 
        'env_material': 'environment_material',
    }

    def __init__(self, config: Dict[str, Any], metadata: pd.DataFrame, 
                 sample_id_column: str = "sample_id"):
        """Initialize the MetadataCleaner.
        
        Args:
            config: Configuration dictionary for cleaning parameters
            metadata: DataFrame containing metadata to clean
            sample_id_column: Name of the sample ID column
            
        Raises:
            ValueError: If metadata is empty or sample_id_column not found
        """
        if metadata.empty:
            raise ValueError("Cannot process empty metadata DataFrame")
            
        self.config = config
        self.sample_id_column = config.get("metadata_id_column", sample_id_column)
        self.df = metadata.copy()  # Work on a copy to avoid modifying original
        
        # Validate sample ID column exists
        if self.sample_id_column not in self.df.columns:
            raise ValueError(f"Sample ID column '{self.sample_id_column}' not found in metadata")
            
        logger.info(f"Initialized MetadataCleaner with {len(self.df)} rows and {len(self.df.columns)} columns")

    def run_all(self) -> pd.DataFrame:
        """Execute the complete metadata cleaning pipeline.
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting metadata cleaning pipeline...")
        
        # Sort columns for consistent output
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)
        
        # Execute cleaning steps in order
        cleaning_steps = [
            ("Removing duplicate columns", self._clean_columns),
            ("Cleaning sample IDs", self._clean_sample_ids),
            ("Collapsing _ena columns", lambda: self._collapse_suffix_columns('_ena')),
            ("Collapsing _study columns", lambda: self._collapse_suffix_columns('_study')),  
            ("Collapsing _deg columns", lambda: self._collapse_suffix_columns('_deg')),
            ("Collapsing .1 columns", lambda: self._collapse_suffix_columns('.1')),
            ("Consolidating pH columns", self._collapse_ph_columns),
            ("Standardizing column names", self._standardize_columns),
            ("Filling missing coordinates", self._fill_missing_coordinates),
        ]
        
        for step_name, step_func in cleaning_steps:
            try:
                logger.debug(f"Executing: {step_name}")
                step_func()
            except Exception as e:
                logger.error(f"Error in {step_name}: {str(e)}")
                raise
        #self.df['nuclear_contamination_status'] = self.df['nuclear_contamination_status'].fillna(False)
        logger.info(f"Cleaning complete. Final shape: {self.df.shape}")
        return self.df

    def _clean_columns(self) -> None:
        """Remove duplicate columns, keeping the first occurrence."""
        if not self.df.columns.duplicated().any():
            return
            
        duplicated_columns = self.df.columns[self.df.columns.duplicated()].unique().tolist()
        logger.warning(f"Found duplicate columns: {duplicated_columns}. Removing duplicates.")
        
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

    def _clean_sample_ids(self) -> None:
        """Normalize sample IDs and remove duplicate rows."""
        original_count = len(self.df)
        
        # Normalize sample IDs to lowercase strings
        self.df[self.sample_id_column] = (
            self.df[self.sample_id_column]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        
        # Remove duplicate sample IDs, keeping first occurrence
        self.df = self.df.drop_duplicates(subset=[self.sample_id_column])
        
        removed_count = original_count - len(self.df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} duplicate sample IDs")

    def _collapse_suffix_columns(self, suffix: str) -> None:
        """Collapse columns with a given suffix into their base columns.
        
        Args:
            suffix: The suffix to look for (e.g., '_ena', '_study')
        """
        if not suffix:
            return
            
        # Find columns with the suffix
        suffix_columns = [col for col in self.df.columns if col.endswith(suffix)]
        if not suffix_columns:
            return
            
        logger.debug(f"Collapsing {len(suffix_columns)} columns with suffix '{suffix}'")
        
        # Process each suffix column
        columns_to_drop = []
        for col in suffix_columns:
            if col == suffix:  # Skip if column name is exactly the suffix
                continue
                
            base_col = col[:-len(suffix)]
            
            if base_col in self.df.columns:
                # Combine values: prioritize base column, fill missing from suffix column
                self.df[base_col] = self.df[base_col].combine_first(self.df[col])
            else:
                # Rename suffix column to base column
                self.df[base_col] = self.df[col]
                
            columns_to_drop.append(col)
        
        # Drop processed suffix columns
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)

    def _collapse_ph_columns(self) -> None:
        """Consolidate all pH-related columns into a single 'ph' column.
        Matches columns that start with 'ph' followed by a non-letter character,
        or are exactly 'ph'. Prioritizes exact 'ph' column if present."""
        ph_columns = [col for col in self.df.columns if self.PH_PATTERN.match(col)]
        
        if not ph_columns:
            return
            
        logger.debug(f"Consolidating {len(ph_columns)} pH columns: {ph_columns}")
        
        # Prioritize exact 'ph' column
        if 'ph' in ph_columns:
            ph_columns.remove('ph')
            ph_columns = ['ph'] + sorted(ph_columns)
        else:
            ph_columns = sorted(ph_columns)
        
        # Use vectorized operations for better performance
        ph_data = self.df[ph_columns]
        
        # Get first non-null value across columns for each row
        consolidated_ph = ph_data.bfill(axis=1).iloc[:, 0]
        
        # Update DataFrame
        self.df = self.df.drop(columns=ph_columns)
        self.df['ph'] = consolidated_ph

    def _standardize_columns(self) -> None:
        """Standardize column names using predefined mappings."""
        mappings = self.config.get('column_mappings', self.DEFAULT_COLUMN_MAPPINGS)
        
        for old_name, new_name in mappings.items():
            if old_name in self.df.columns and new_name in self.df.columns:
                # Combine columns if both exist
                self.df[old_name] = self.df[old_name].combine_first(self.df[new_name])
                self.df = self.df.drop(columns=[new_name])

    def _extract_coordinates_from_string(self, coord_string: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract latitude and longitude from a coordinate string.
        
        Supports formats like:
        - "40.7128N 74.0060W"
        - "40.7128, -74.0060"  
        - "N40.7128 W74.0060"
        
        Args:
            coord_string: String containing coordinate information
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if extraction fails
        """
        if not isinstance(coord_string, str) or not coord_string.strip():
            return None, None
            
        # Find all numbers in the string
        number_matches = list(self.NUM_PATTERN.finditer(coord_string))
        if len(number_matches) < 2:
            return None, None
        
        try:
            num1 = float(number_matches[0].group())
            num2 = float(number_matches[1].group())
        except (ValueError, TypeError):
            return None, None
        
        # Look for directional indicators near each number
        directions = []
        for match in number_matches[:2]:
            start, end = match.span()
            # Check a small window around each number for direction letters
            window = coord_string[max(0, start-3):min(len(coord_string), end+3)]
            direction_match = self.LETTER_PATTERN.search(window)
            directions.append(direction_match.group().upper() if direction_match else None)
        
        # Process coordinates based on directional indicators
        lat, lon = None, None
        
        for i, (num, direction) in enumerate(zip([num1, num2], directions)):
            if direction in ['N', 'S']:
                lat = num if direction == 'N' else -abs(num)
            elif direction in ['E', 'W']:
                lon = num if direction == 'E' else -abs(num)
        
        # If we found specific directions, use them
        if lat is not None and lon is not None:
            return lat, lon
        
        # Otherwise, assume first number is latitude, second is longitude
        return num1, num2

    def _fill_missing_coordinates(self) -> None:
        """Fill missing latitude and longitude values using vectorized operations."""
        # Ensure coordinate columns exist
        if 'latitude_deg' not in self.df.columns:
            self.df['latitude_deg'] = np.nan
        if 'longitude_deg' not in self.df.columns:
            self.df['longitude_deg'] = np.nan
        
        # Get existing source columns
        existing_lat_sources = [col for col in self.COORDINATE_SOURCES['lat'] 
                               if col in self.df.columns]
        existing_lon_sources = [col for col in self.COORDINATE_SOURCES['lon'] 
                               if col in self.df.columns]  
        existing_pair_sources = [col for col in self.COORDINATE_SOURCES['pairs'] 
                                if col in self.df.columns]
        
        if not (existing_lat_sources or existing_lon_sources or existing_pair_sources):
            logger.debug("No coordinate source columns found")
            return
        
        # Identify rows with missing coordinates
        missing_lat = self.df['latitude_deg'].isna()
        missing_lon = self.df['longitude_deg'].isna()
        missing_any = missing_lat | missing_lon
        
        if not missing_any.any():
            logger.debug("No missing coordinates to fill")
            return
        
        logger.info(f"Filling coordinates for {missing_any.sum()} rows...")
        
        # Fill latitude values
        if missing_lat.any():
            self._fill_coordinate_column(
                target_col='latitude_deg',
                source_cols=existing_lat_sources,
                missing_mask=missing_lat
            )
        
        # Fill longitude values  
        if missing_lon.any():
            self._fill_coordinate_column(
                target_col='longitude_deg', 
                source_cols=existing_lon_sources,
                missing_mask=missing_lon
            )
        
        # Fill from coordinate pair columns
        if existing_pair_sources:
            self._fill_from_coordinate_pairs(existing_pair_sources)

    def _fill_coordinate_column(self, target_col: str, source_cols: List[str], 
                               missing_mask: pd.Series) -> None:
        """Fill a coordinate column from source columns using vectorized operations."""
        if not source_cols:
            return
            
        # Process source columns in priority order (last has highest priority)
        for source_col in source_cols:
            if source_col not in self.df.columns:
                continue
                
            # Convert source column to numeric, errors become NaN
            source_values = pd.to_numeric(self.df[source_col], errors='coerce')
            
            # Fill missing values in target column
            fill_mask = missing_mask & source_values.notna()
            if fill_mask.any():
                self.df.loc[fill_mask, target_col] = source_values[fill_mask]
                missing_mask = missing_mask & ~fill_mask  # Update mask
                
                logger.debug(f"Filled {fill_mask.sum()} {target_col} values from {source_col}")

    def _fill_from_coordinate_pairs(self, pair_sources: List[str]) -> None:
        """Extract coordinates from string columns containing coordinate pairs."""
        missing_lat = self.df['latitude_deg'].isna()
        missing_lon = self.df['longitude_deg'].isna()
        missing_any = missing_lat | missing_lon
        
        if not missing_any.any():
            return
            
        for source_col in pair_sources:
            if source_col not in self.df.columns:
                continue
                
            # Get rows that still need coordinates and have data in source column
            needs_processing = (
                missing_any & 
                self.df[source_col].notna() & 
                (self.df[source_col] != '')
            )
            
            if not needs_processing.any():
                continue
                
            logger.debug(f"Extracting coordinates from {source_col} for {needs_processing.sum()} rows")
            
            # Extract coordinates for rows that need processing
            extracted_coords = (
                self.df.loc[needs_processing, source_col]
                .apply(self._extract_coordinates_from_string)
            )
            
            # Separate latitude and longitude
            extracted_lats = extracted_coords.apply(lambda x: x[0] if x[0] is not None else np.nan)
            extracted_lons = extracted_coords.apply(lambda x: x[1] if x[1] is not None else np.nan)
            
            # Fill missing latitudes
            lat_fill_mask = needs_processing & missing_lat & extracted_lats.notna()
            if lat_fill_mask.any():
                self.df.loc[lat_fill_mask, 'latitude_deg'] = extracted_lats[lat_fill_mask]
                missing_lat = missing_lat & ~lat_fill_mask
            
            # Fill missing longitudes
            lon_fill_mask = needs_processing & missing_lon & extracted_lons.notna()  
            if lon_fill_mask.any():
                self.df.loc[lon_fill_mask, 'longitude_deg'] = extracted_lons[lon_fill_mask]
                missing_lon = missing_lon & ~lon_fill_mask
                
            # Update missing_any mask
            missing_any = missing_lat | missing_lon
            
            if not missing_any.any():
                break  # All coordinates filled

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get statistics about the cleaning process.
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns), 
            'missing_coordinates': (
                self.df['latitude_deg'].isna() | self.df['longitude_deg'].isna()
            ).sum() if 'latitude_deg' in self.df.columns else 0,
            'columns': sorted(self.df.columns.tolist())
        }

# ==================================================================================== #

def clean_metadata(config: Dict, metadata: pd.DataFrame):
    """[Placeholder]"""
    cleaner = MetadataCleaner(
        config=config, 
        metadata=metadata
    )
    cleaner.run_all()
    return cleaner.df

# ==================================================================================== #

class MetadataSummarizer:
    """Summarizes metadata column prefixes and associated columns."""

    def __init__(self, config: Dict, metadata: pd.DataFrame):
        self.config = config
        self.df = metadata
        self.prefix_df = self._find_column_prefixes()

    def _find_column_prefixes(self) -> pd.DataFrame:
        all_columns = list(self.df.columns)
        unique_prefixes = sorted(set(col.split('_')[0] for col in all_columns))
        summary_data = []

        for prefix in unique_prefixes:
            columns_with_prefix = [col for col in all_columns if col == prefix or col.startswith(f"{prefix}_")]
            summary_data.append({
                "prefix": prefix,
                "n_columns": len(columns_with_prefix),
                "columns": ", ".join(columns_with_prefix)
            })

        # Convert to DataFrame
        prefix_df = pd.DataFrame(summary_data).sort_values(by="n_columns", ascending=False).reset_index(drop=True)
        return prefix_df
