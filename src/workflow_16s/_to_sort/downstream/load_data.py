# ===================================== IMPORTS ====================================== #

# Standard Imports
import glob
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s.constants import MODE, SAMPLE_ID_COLUMN, TAXONOMIC_LEVELS
from workflow_16s.nuclear_fuel_cycle.nuclear_fuel_cycle import (
    update_nfc_facilities_data
)
from workflow_16s.utils.biom import (
    import_biom, import_merged_biom_table, export_h5py, sample_id_map
)
from workflow_16s.utils.dir import Dir, ProjectDir
from workflow_16s.utils.metadata import (
    clean_metadata, import_tsv, import_merged_metadata_tsv, MetadataSummarizer
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

def align_table_and_metadata(
    table: Table,
    metadata: pd.DataFrame,
    sample_id_col: str = SAMPLE_ID_COLUMN
) -> Tuple[Table, pd.DataFrame]:
    """Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata:      Sample metadata DataFrame.
        sample_id_col: Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    # Handle empty metadata
    if metadata.empty:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_id_col])
    
    biom_mapping = sample_id_map(table)
    shared_ids = [id for id in metadata[sample_id_col] if id in biom_mapping]
    
    # Handle no shared IDs
    if not shared_ids:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_id_col])
    
    filtered_metadata = metadata[metadata[sample_id_col].isin(shared_ids)]
    original_ids = [biom_mapping[id] for id in filtered_metadata[sample_id_col]]
    filtered_table = table.filter(original_ids, axis='sample', inplace=False)
    return filtered_table, filtered_metadata
    
# ==================================================================================== #

class DownstreamDataLoader:
    """Loads and aligns BIOM feature tables and metadata for downstream analysis.
    
    Handles loading of both ASV and genus-level feature tables along with 
    corresponding metadata files. Supports filtering based on nuclear fuel cycle 
    facilities and alignment of samples between tables and metadata.

    Attributes:
        config:                  Configuration dictionary containing processing 
                                 parameters.
        target_subfragment_mode: Processing mode for subfragments ('asv', 'genus', 
                                 or 'any').
        metadata_id_column:      Column name in metadata containing sample IDs.
        verbose:                 Verbosity flag.
        project_dir:             Project directory object containing path definitions.
        existing_subsets:        Pre-existing data subsets from upstream processing.
        tables:                  Dictionary storing loaded feature tables.
        metadata:                Dictionary storing loaded metadata.
        nfc_facilities:          DataFrame containing nuclear fuel cycle facility data 
                                 if enabled.
        table_paths:             List of paths to found BIOM table files.
    """
    ModeConfig = {
        "asv": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    def __init__(
        self,
        config: Dict,
        project_dir: Any, 
        existing_subsets: Any = None
    ):
        self.config = config
        self.target_subfragment_mode = self.config.get("target_subfragment_mode", MODE)
        self.metadata_id_column = self.config.get("metadata_id_column", SAMPLE_ID_COLUMN)
      
        self.verbose = self.config.get("verbose", False)
      
        self.project_dir = project_dir
        self.existing_subsets = existing_subsets

        # Initialize storage for feature tables and metadata
        self.tables: Dict = {'raw': {}}
        self.metadata: Dict = {'raw': {}}
        self.nfc_facilities = None
        self.table_paths = None

    def run(self):
        """Executes the data loading process."""
        # Load the ASV feature table if the target subfragment is specified (so, not 'any')
        if not self.target_subfragment_mode == 'any':
            self._load_table_and_metadata('asv')  
        # Load the taxonomically-assigned feature table at the genus level     
        self._load_table_and_metadata('genus')
      
    def _load_table_and_metadata(self, mode: str = 'genus') -> None:
        """Loads feature table and metadata for a specific processing mode."""
        # Get the level and subdirectory to search for the feature table
        level, subdir, _ = self.ModeConfig[mode]
        # Load the feature table and sample metadata
        table = self._load_biom_table(level, subdir)
        metadata = self._load_metadata(level, subdir)
        # Clean the sample metadata
        metadata = clean_metadata(self.config, metadata)
        # Summarize the sample metadata
        MetadataSummarizer(self.config, metadata)          
        table, metadata = self._filter_and_align(table, metadata, level)
        self._log_results(table, metadata, level)
        
        # Find NFC facility data and create a new column indicating whether
        # samples are within a certain threshold distance from a NFC facility
        if self.config.get("nfc_facilities", {}).get("enabled", False):
            logger.info("Finding NFC facilities...")
            try:
                self.nfc_facilities, metadata = self._load_nfc_facilities(metadata)
            except Exception as e:
                logger.error(f"Failed finding NFC facilities: {e}\n"
                             f"Traceback: {traceback.format_exc()}")
                
        self.tables['raw'][level], self.metadata['raw'][level] = table, metadata

    def _filter_and_align(self, table, metadata, level) -> Tuple[Table, pd.DataFrame]:
        """Filters and aligns feature table with metadata using sample IDs."""
        table, metadata = align_table_and_metadata(
            table, metadata, self.metadata_id_column
        )
        if table.is_empty() or metadata.empty:
            logger.warning(f"Alignment resulted in empty table for level '{level}'")
        return table, metadata
        
    def _load_biom_table(self, level, subdir) -> Table:
        """Loads and merges BIOM tables from found file paths."""
        table_paths = self._get_table_paths(level, subdir)  
        if not table_paths:
            raise FileNotFoundError("No BIOM table filepaths found")
        return import_merged_biom_table(biom_paths=table_paths)

    def _load_metadata(self, level, subdir) -> pd.DataFrame:
        """Loads and merges metadata from TSV files."""
        metadata_paths = self._get_metadata_paths(level, subdir)
        if not metadata_paths:
            raise FileNotFoundError("No metadata TSV filepaths found")
        
        columns_to_rename = self.config.get("columns_to_rename", {}) or {}
        
        metadata = import_merged_metadata_tsv(
            tsv_paths=metadata_paths, 
            columns_to_rename=columns_to_rename  
        )
        return metadata

    def _get_table_paths(self, level: str, subdir: str) -> List[Path]:
        """Discovers BIOM table file paths based on processing mode."""
        # If there are existing subsets of datasets from upstream processing loaded
        if self.existing_subsets is not None:
            table_paths = [paths[subdir] for subset_id, paths in self.existing_subsets.items()]
        # If there are NOT existing subsets, search in the directory files matching a pattern
        else:
            if self.config["target_subfragment_mode"] == 'any':
                subfragment = "*"
            else:
                subfragment = self.config["target_subfragment_mode"]
            qiime_data_dir = Path(self.project_dir.qiime_data_per_dataset)   
            pattern = "/".join([
                str(qiime_data_dir), "*", "*", "*", subfragment, 
                "FWD_*_REV_*", subdir, "feature-table.biom"
            ])
            table_paths = glob.glob(pattern, recursive=True)
        table_paths = [Path(p) for p in table_paths]
        if self.verbose:
            n = len(table_paths)
            logger.info(f"Found {n} feature tables")
        self.table_paths = table_paths 
        return table_paths 

    def _get_metadata_paths(self, level, subdir) -> List[Path]:
        """Discovers metadata file paths corresponding to found BIOM tables."""
        tsv_paths: List[Path] = []
        # If there are existing subsets of datasets from upstream processing loaded
        if self.existing_subsets is not None:
            tsv_paths = [paths["metadata"] for subset_id, paths in self.existing_subsets.items()]
        # If there are NOT existing subsets, find metadata files corresponding to each biom table file
        else:
            table_paths = self.table_paths if self.table_paths is not None else self._get_table_paths(level, subdir)
            for table_path in table_paths:
                dataset_dir = table_path.parent if table_path.is_file() else table_path
                tail = dataset_dir.parts[-6:-1]
                metadata_dir = Path(self.project_dir.metadata_per_dataset)
                tsv_path = metadata_dir.joinpath(*tail, "sample-metadata.tsv")
                if tsv_path.exists():
                    tsv_paths.append(tsv_path)
                  
        if self.verbose:
            (f"Found {len(tsv_paths)} metadata files")
        return tsv_paths

    # SPECIAL CASE: LOAD NFC FACILITIES DATA
    def _load_nfc_facilities(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and processes nuclear fuel cycle facilities data."""
        return update_nfc_facilities_data(config=self.config, metadata=metadata)

    def _log_results(self, table, metadata, level) -> None:
        """Logs summary statistics for loaded data."""
        table_size = "Empty" if table.is_empty() else f"{table.shape[0]} features × {table.shape[1]} samples"
        metadata_size = "Empty" if metadata.empty else f"{metadata.shape[0]} samples × {metadata.shape[1]} cols"
        feature_type = "genera" if level == "genus" else "ASVs"
        logger.info(f"{'Loaded metadata:':<30}{metadata_size}")
        logger.info(f"{'Loaded features:':<30}{table_size} {feature_type}")


class ExistingDataLoader:
    levels = TAXONOMIC_LEVELS
    
    def __init__(self, config: Dict, project_dir: Any):
        self.config = config
        self.project_dir = project_dir
        self.tables = defaultdict(lambda: defaultdict(lambda: {}))
        self.metadata = defaultdict(lambda: defaultdict(lambda: {}))
        self.nfc_facilities = None
    
    def _transform_enabled(self, config_key: str):
        # Explicitly convert config value to boolean
        return bool(self.config.get("features", {}).get(config_key, True))
        
    def run(self) -> None:      
        """Load existing tables and metadata files."""
        steps = [
            ("filter", "filtered"),
            ("normalize", "normalized"),
            ("clr_transform", "clr_transformed")
        ]
        n_steps = sum([self._transform_enabled(key) for key, _ in steps])
        
        with get_progress_bar() as progress:
            task_desc = "Checking existing features and metadata files"
            task_id = progress.add_task(_format_task_desc(task_desc), total=n_steps)   
            
            for key, table_type in steps:
                if self._transform_enabled(key):
                    for level in self.levels.keys():
                        base = Path(self.project_dir.data) / "merged"
                        table_path = base / "table" / table_type / f"{level}.biom"
                        metadata_path = base / "metadata" / table_type / f"{level}.tsv"
                        
                        try:
                            table = import_biom(table_path)
                            metadata = import_tsv(metadata_path)
                            self.tables[table_type][level] = table
                            self.metadata[table_type][level] = metadata
                        except Exception as e:
                            error_msg = (f"Failed to load required data files:\n"
                                         f"  Table: {table_path}\n"
                                         f"  Metadata: {metadata_path}\n"
                                         f"  Error: {str(e)}\n"
                                         f"  Traceback: {traceback.format_exc()}")
                            logger.error(error_msg)
                            raise RuntimeError(error_msg) from e
                        finally:
                            progress.update(task_id, advance=1)   
                            
        # If enabled, find samples within a threshold distance from NFC facilities
        if self.config.get("nfc_facilities", {}).get("enabled", False):
            logger.info("Finding NFC facilities...")
            try:
                self.nfc_facilities, *_ = self._load_nfc_facilities(self.metadata['raw']['genus'])
            except Exception as e:
                error_msg = (f"Failed to load NFC facilities data: {e}\n"
                             f"Traceback: {traceback.format_exc()}")
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
    def _load_nfc_facilities(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and processes nuclear fuel cycle facilities data."""
        return update_nfc_facilities_data(config=self.config, metadata=metadata)

# ==================================================================================== #

def load_data(config: Dict, project_dir: Any, existing_subsets: Any = None):
    """Convenience function to initialize and run the data loader.
    
    Args:
        config:           Configuration dictionary for data loading parameters.
        project_dir:      Project directory object containing path definitions.
        existing_subsets: Pre-existing data subsets from upstream processing.
        
    Returns:
        Initialized and executed DownstreamDataLoader instance.
    """
    loader = DownstreamDataLoader(
        config=config,
        project_dir=project_dir,
        existing_subsets=existing_subsets
    )
    loader.run()
    return loader


def load_existing_data(config: Dict, project_dir: Any):
    """Convenience function to initialize and run the data loader.
    
    Args:
        config:           Configuration dictionary for data loading parameters.
        project_dir:      Project directory object containing path definitions.
        
    Returns:
        Initialized and executed ExistingDataLoader instance.
    """
    data = ExistingDataLoader(config=config, project_dir=project_dir)
    data.run()
    return data
