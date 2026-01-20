# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-Party Imports
import pandas as pd
from biom.table import Table

# Local Imports from workflow_16s
from workflow_16s.constants import TAXONOMIC_LEVELS, TAXONOMY_PREFIXES
from workflow_16s.downstream import Data
from workflow_16s.logger import get_logger
from workflow_16s.utils.biom_utils import export_h5py
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, sync_samples
)
from workflow_16s.utils.metadata_utils import export_tsv
from workflow_16s.utils.progress import get_progress_bar

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =============================== DATA PROCESSING ==================================== #

class DataProcessor:
    """Handles downstream data preparation, transformation, and export."""

    levels = TAXONOMIC_LEVELS

    def __init__(self, config: Dict, data: Data) -> None:
        self.config = config
        self.data = data
        self.verbose = self.config.get("verbose", False)
        project_dir = self.config.get("project_dir")
        if not project_dir:
            raise ValueError("'project_dir' must be specified in the config.")
        self.project_dir = Path(project_dir)

    def run(self, output_dir: Optional[Path] = None) -> None:
        """
        Executes the full data preparation and transformation pipeline.

        Args:
            output_dir: The directory to save results. If None, defaults to
                        '[project_dir]/data'.
        """
        if self.data.metadata is None:
            raise ValueError("Metadata must be loaded before running the processor.")
        
        # If no specific output directory is given, use a default.
        if output_dir is None:
            output_dir = self.project_dir / "data"

        self._collapse_taxonomy()
        self._create_raw_presence_absence()

        for level in self.levels:
            self._apply_transformations(level)

        self._save_all_data(output_dir)

    def get_merged_dataframe(self, table_type: str, level: str) -> pd.DataFrame:
        """
        Converts a BIOM table to a DataFrame and merges it with the master metadata.
        """
        if self.data.metadata is None:
            raise RuntimeError("Cannot merge without loaded metadata.")

        table = self.data.tables[table_type][level]
        features_df = table.to_dataframe(dense=True).T
        
        # Merge using the DataFrame's index, which should be the sample ID
        merged_df = self.data.metadata.merge(
            features_df, 
            left_index=True, 
            right_index=True, 
            how="inner"
        )
        logger.info(f"Merged DataFrame for '{table_type}/{level}' created with shape: {merged_df.shape}")
        return merged_df

    def _collapse_taxonomy(self) -> None:
        """Collapses the base feature table to all taxonomic levels with validation."""
        base_level = "asv" if "asv" in self.data.tables["raw"] else "genus"
        if base_level not in self.data.tables["raw"]:
             logger.warning(f"Base table '{base_level}' not found in raw data. Skipping taxonomy collapse.")
             return
        
        base_table, _ = self._fetch_data("raw", base_level)

        with get_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Collapsing taxonomy[/]", total=len(self.levels))
            for level in self.levels:
                progress.update(task_id, description=f"Collapsing to {level.title()}")
                
                if level == base_level or level not in self.levels:
                    table = base_table
                else:
                    table = collapse_taxa(base_table, level)
                    table = self._validate_and_clean_taxonomy(table, level)

                self.data.tables["raw"][level] = table
                progress.update(task_id, advance=1)
                
    def _validate_and_clean_taxonomy(self, table: Table, level: str) -> Table:
        """Cleans taxonomic labels and merges duplicate features by summing their counts."""
        ordered_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        try:
            level_index = ordered_levels.index(level)
        except ValueError:
            return table

        invalid_prefixes = tuple(TAXONOMY_PREFIXES[i] for i in range(level_index + 1, len(ordered_levels)))
        
        map_for_cleaning = {}
        anomalies_found = 0
        for obs_id in table.ids(axis='observation'):
            if any(prefix in obs_id for prefix in invalid_prefixes):
                anomalies_found += 1
                clean_id = re.split(f"({'|'.join(invalid_prefixes)})", obs_id, 1)[0]
                map_for_cleaning[obs_id] = clean_id.rstrip(';')
        
        if anomalies_found > 0:
            logger.warning(f"Cleaned {anomalies_found} anomalous IDs in '{level}' table, merging any duplicates.")
            full_mapping = {obs_id: map_for_cleaning.get(obs_id, obs_id) for obs_id in table.ids(axis='observation')}
            return table.collapse(lambda id_, md: full_mapping[id_], axis='observation')
            
        return table

    def _apply_transformations(self, level: str) -> None:
        """Applies filtering, normalization, and CLR to a table."""
        if level not in self.data.tables["raw"]:
            return # Skip if the raw table for this level doesn't exist
            
        table, metadata = self._fetch_data("raw", level)
        steps = [
            ("filter", filter, "filtered"),
            ("normalize", normalize, "normalized"),
            ("clr_transform", clr, "clr_transformed"),
        ]
        
        current_table = table
        for key, func, out_type in steps:
            if self._is_enabled(key):
                try:
                    transformed_table = func(current_table)
                    if self.data.metadata is not None:
                        # Sync samples in case filtering removed some
                        transformed_table, self.data.metadata = sync_samples(transformed_table, self.data.metadata)
                    self.data.tables[out_type][level] = transformed_table
                    # Use the transformed table as input for the next step (e.g., normalize the filtered table)
                    if out_type == "filtered":
                        current_table = transformed_table
                except Exception as e:
                    logger.error(f"'{key}' transformation failed for '{level}': {e}")
                    
    def _create_raw_presence_absence(self) -> None:
        """Creates presence/absence tables from all raw tables."""
        if not self._is_enabled("presence_absence"): return
        
        logger.info("Generating presence/absence tables from raw data...")
        for level, table in self.data.tables["raw"].items():
            pa_table = presence_absence(table)
            self.data.tables["presence_absence"][level] = pa_table

    def _save_all_data(self, output_dir: Path) -> None:
        """Saves all generated tables and the final metadata file to the specified directory."""
        if self.data.metadata is not None:
            meta_dir = output_dir / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            export_tsv(self.data.metadata, meta_dir / "processed_metadata.tsv")

        tasks = []
        for table_type, levels in self.data.tables.items():
            for level, table in levels.items():
                table_dir = output_dir / "tables" / table_type
                table_dir.mkdir(parents=True, exist_ok=True)
                out_path = table_dir / f"{level}.biom"
                tasks.append((table, out_path))
        
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(export_h5py, tbl, pth) for tbl, pth in tasks]
            self._monitor_futures(futures, "Exporting BIOM tables")
        logger.info(f"All processed data saved to: {output_dir}")
            
    def _monitor_futures(self, futures: List, description: str):
        """Monitors future completion with a progress bar."""
        with get_progress_bar() as progress:
            task = progress.add_task(f"[cyan]{description}[/]", total=len(futures))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"A task in '{description}' failed: {e}")
                progress.update(task, advance=1)
                
    def _fetch_data(self, table_type: str, level: str) -> Tuple[Table, pd.DataFrame]:
        """Retrieves a table and the master metadata."""
        if self.data.metadata is None: raise RuntimeError("Metadata is not loaded.")
        table = self.data.tables.get(table_type, {}).get(level)
        if table is None: raise ValueError(f"Table for type '{table_type}' at level '{level}' not found.")
        return table, self.data.metadata

    def _is_enabled(self, config_key: str) -> bool:
        """Checks if a feature processing step is enabled in the config."""
        return bool(self.config.get("features", {}).get(config_key, True))