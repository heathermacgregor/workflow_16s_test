# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.downstream.input import update_table_and_metadata
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

class PrepData:
    ModeConfig = {
        "any": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    def __init__(
        self,
        config: Dict,
        tables: Dict,
        metadata: Dict,
        mode: str,
        project_dir: Union[str, Path],
        verbose: bool = False
    ) -> None:
        self.config, self.project_dir, self.mode = config, project_dir, mode
        self.tables, self.metadata = tables, metadata
        self.verbose = verbose
        # Collapse raw tables at all taxonomy levels
        self._collapse_taxonomy("raw")

        # For each level, apply preprocessing and create presence/absence
        for level in constants.levels:
            try:
                self._apply_preprocessing(level)
                for table_type in ["filtered", "normalized", "clr_transformed"]:
                    if table_type in self.tables and level in self.tables[table_type]:
                        self._create_presence_absence(table_type, level)
            except Exception as e:
                logger.error(f"Preprocessing failed for {level}: {e}")
                
        # Save tables
        self._save_tables()

    def _collapse_taxonomy(self, table_type: str = "raw") -> None:
        # Get base level from mode config (e.g. "asv" or "genus")
        base_level = self.ModeConfig[self.mode][0]
        base_table = self.tables.setdefault(table_type, {}).get(base_level)
        base_metadata = self.metadata.setdefault(table_type, {}).get(base_level)

        if base_table is None or base_metadata is None:
            raise ValueError(
                f"Missing base table or metadata for {base_level} level in {table_type}"
            )

        with get_progress_bar() as progress:
            ct_desc = "Collapsing taxonomy"
            ct_task = progress.add_task(_format_task_desc(ct_desc), total=len(constants.levels))   
            for level in constants.levels:
                # Use level-specific description
                level_desc = f"{ct_desc} {table_type} → {level.title()}"
                progress.update(ct_task, description=_format_task_desc(level_desc))
                
                if level == base_level:
                    # Use base table directly without collapsing
                    table = base_table
                    metadata = base_metadata
                else:
                    # Collapse from base table to target level
                    table = collapse_taxa(base_table, level, progress, ct_task)
                    table, metadata = update_table_and_metadata(table, base_metadata)

                # Store results
                self.tables.setdefault(table_type, {})[level] = table
                self.metadata.setdefault(table_type, {})[level] = metadata
                progress.update(ct_task, advance=1)
                
            progress.update(ct_task, description=_format_task_desc(ct_desc))

    def _apply_preprocessing(self, level: str) -> None:
        features_config = self.config.get("features", {})
        table = self.tables.get("raw", {}).get(level)
        metadata = self.metadata.get("raw", {}).get(level)
        if table is None or metadata is None:
            raise ValueError(f"Missing raw table or metadata for level: {level}")

        steps = [
            ("filter", filter, "filtered"),
            ("normalize", normalize, "normalized"),
            ("clr_transform", clr, "clr_transformed")
        ]
    
        for config_key, func, table_type in steps:
            # Explicitly convert config value to boolean
            if bool(features_config.get(config_key, True)):
                initial_samples, initial_features = table.shape
                try:
                    table = func(table)
                except Exception as e:
                    logger.error(f"Preprocessing function failed for {config_key} at {level}: {e}")
                    continue
                    
                table, metadata = update_table_and_metadata(table, metadata)
                # Set defaults
                self.tables.setdefault(table_type, {})[level] = table
                self.metadata.setdefault(table_type, {})[level] = metadata
                
                logger.info( 
                    f"Preprocessing: {initial_samples} → {table.shape[0]} samples, " 
                    f"{initial_features} → {table.shape[1]} features" 
                )
        
    def _create_presence_absence(self, table_type: str, level: str) -> None:
        if not bool(self.config.get("features", {}).get("presence_absence", False)):
            return
        try:
            table = self.tables[table_type][level]
            metadata = self.metadata[table_type][level]
            pa_table = presence_absence(table)
            pa_table, pa_metadata = update_table_and_metadata(pa_table, metadata)
            self.tables.setdefault(f"{table_type}_presence_absence", {})[level] = pa_table
            self.metadata.setdefault(f"{table_type}_presence_absence", {})[level] = pa_metadata
        except Exception as e:
            logger.error(f"Presence/absence table failed for {level} ({table_type}): {e}")

    def _save_tables(self) -> None:
        # Create directory if it doesn't exist
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)
    
        # Prepare export tasks
        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                out = tdir / f"{level}.biom"  # Simplified filename
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out))
    
        # Use ThreadPoolExecutor for parallel exports
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for table, out_path in export_tasks:
                future = executor.submit(export_h5py, table, out_path)
                futures[future] = (table_type, level, out_path)
    
            # Track progress
            with get_progress_bar() as progress:
                task = progress.add_task(
                    _format_task_desc("Exporting tables"), 
                    total=len(export_tasks)
                )
                for future in as_completed(futures):
                    table_type, level, out_path = futures[future]
                    try:
                        future.result()
                        if self.verbose:
                            logger.debug(f"Exported {table_type}/{level} to {out_path}")
                    except Exception as e:
                        logger.error(f"Failed to export {out_path}: {str(e)}")
                    finally:
                        progress.update(task, advance=1)
