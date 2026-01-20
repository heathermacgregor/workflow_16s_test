# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s.constants import MODE, TAXONOMIC_LEVELS
from workflow_16s.downstream.load_data import align_table_and_metadata
from workflow_16s.utils.biom import collapse_taxa, export_h5py, presence_absence
from workflow_16s.utils.data import clr, filter, normalize
from workflow_16s.utils.dir import Dir, ProjectDir
from workflow_16s.utils.metadata import export_tsv
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

class DownstreamDataPrepper:
    """A class that handles downstream data preparation.
    
    - Generates feature tables collapsed at each taxonomic level
    - Performs various transformations (filtering, normalization, CLR) to feature tables
    - Generates presence/absence tables
    - Exports processed BIOM tables

    Attributes:
        config:      Configuration dictionary for processing parameters.
        metadata:    Metadata associated with the feature tables.
        tables:      Feature tables at different processing stages.
        project_dir: Project directory structure.
        mode:        Processing mode ('any' or 'genus').
        verbose:     Verbosity flag.
        output_dir:  Output directory for processed tables.
    """
    ModeConfig = {
        "any": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    levels = TAXONOMIC_LEVELS
    def __init__(
        self,
        config: Dict,
        metadata: Dict,
        tables: Dict,
        project_dir: Any
    ) -> None:
        self.config = config 
        self.mode = "genus" if not self.config.get("target_subfragment_mode", MODE) == 'any' else "any"
        self.verbose = self.config.get("verbose", False)
      
        self.output_dir = Path(project_dir.data)
        
        self.metadata, self.tables = metadata, tables        

    def run(self):
        """Execute the full data preparation pipeline."""
        # Collapse raw tables at all taxonomy levels
        self._collapse_taxonomy("raw")

        for level in self.levels:
            try:
                self._apply_transformations(level) # Preprocessing 
                for table_type in ["filtered", "normalized", "clr_transformed"]:
                    if table_type in self.tables and level in self.tables[table_type]:
                        self._create_presence_absence(table_type, level)
            except Exception as e:
                logger.error(f"Data prep failed for {level}: {e}")
                
        self._save_tables()
        self._save_metadata()

    def log(self, msg):
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)
        
    def _fetch_data(self, table_type: str, level: str) -> Tuple[pd.DataFrame, Table]:
        """Retrieve table and metadata for specified processing stage and taxonomic level."""
        metadata = self.metadata.get(table_type, {}).get(level)
        table = self.tables.get(table_type, {}).get(level)
        if table is None or metadata is None:
            raise ValueError(
                f"Missing table or metadata for level '{level}' and table type '{table_type}'"
            )
        return table, metadata
       
    def _collapse_taxonomy(self, table_type: str = "raw") -> None:
        """Collapse taxonomy tables from base level to all taxonomic levels."""
        # Get base level from mode config (e.g. "asv" or "genus")
        base_level = "genus" if self.config.get("target_subfragment_mode", MODE) == 'any' else "asv"
        base_table, base_metadata = self._fetch_data(table_type, base_level)

        with get_progress_bar() as progress:
            task_desc = "Collapsing taxonomy"
            task_desc_fmt = _format_task_desc(task_desc)
            task_id = progress.add_task(task_desc_fmt, total=len(self.levels))   
            for level in self.levels:
                # Use level-specific description
                level_desc = f"{task_desc} {table_type} → {level.title()}"
                level_desc_fmt = _format_task_desc(level_desc)
                progress.update(task_id, description=level_desc_fmt)
                
                if level == base_level: # Use base table directly without collapsing
                    table, metadata = base_table, base_metadata
                else: # Collapse from base table to target level
                    table = collapse_taxa(base_table, level)
                    table, metadata = align_table_and_metadata(table, base_metadata)

                # Store results
                self.tables.setdefault(table_type, {})[level] = table
                self.metadata.setdefault(table_type, {})[level] = metadata
                progress.update(task_id, advance=1)
            progress.update(task_id, description=task_desc_fmt)
            
    def _transform_enabled(self, config_key: str):
        """Explicitly convert config value to boolean."""
        return bool(self.config.get("features", {}).get(config_key, True))
        
    def _apply_transformations(self, level: str) -> None:      
        """Apply transformations (filtering, normalization, CLR) to specified taxonomic level."""
        table, metadata = self._fetch_data("raw", level)
        steps = [
            ("filter", filter, "filtered"),
            ("normalize", normalize, "normalized"),
            ("clr_transform", clr, "clr_transformed")
        ]
        n_steps = sum([self._transform_enabled(key) for key, _, _ in steps])
        with get_progress_bar() as progress:
            task_desc = "Transforming_features"
            task_desc_fmt = _format_task_desc(task_desc)
            task_id = progress.add_task(task_desc_fmt, total=n_steps)   
            for key, func, table_type in steps:
                if self._transform_enabled(key):
                    samples_n_0, features_n_0 = table.shape
                    try:
                        transformation_desc = f"{task_desc} {level.title()} → {key}"
                        transformation_desc_fmt = _format_task_desc(transformation_desc)
                        progress.update(task_id, description=transformation_desc_fmt)
                        table = func(table)
                        table, metadata = align_table_and_metadata(table, metadata)
                        # Store results
                        self.tables.setdefault(table_type, {})[level] = table
                        self.metadata.setdefault(table_type, {})[level] = metadata

                        samples_n_1, features_n_1 = table.shape
                        samples_msg = f"{samples_n_0} → {samples_n_1} samples"
                        if samples_n_1 < samples_n_0:
                            samples_n_lost = samples_n_0 - samples_n_1
                            samples_perc_lost = samples_n_lost / samples_n_0
                            samples_msg += f"(-{samples_n_lost} or -{samples_perc_lost}%)"
                        features_msg = f"{features_n_0} → {features_n_1} features"
                        if features_n_1 < features_n_0:
                            features_n_lost = features_n_0 - features_n_1
                            features_perc_lost = features_n_lost / features_n_0
                            features_msg += f"(-{features_n_lost} or -{features_perc_lost}%)"
                        self.log(f"Preprocessing: {samples_msg}, {features_msg}")
                    except Exception as e:
                        logger.error(f"Preprocessing function failed for {key} at {level}: {e}")
                    finally:    
                        progress.update(task_id, advance=1)
        
    def _create_presence_absence(self, table_type: str, level: str) -> None:
        """Create presence/absence tables from specified table type and taxonomic level."""
        if not bool(self.config.get("features", {}).get("presence_absence", False)):
            return
        try:
            metadata = self.metadata[table_type][level]
            table = self.tables[table_type][level]
            
            pa_table = presence_absence(table, level)
            pa_table, pa_metadata = align_table_and_metadata(pa_table, metadata)
            # Store results
            self.tables.setdefault(f"{table_type}_presence_absence", {})[level] = pa_table
            self.metadata.setdefault(f"{table_type}_presence_absence", {})[level] = pa_metadata
        except Exception as e:
            logger.error(f"Presence/absence table failed for {level} ({table_type}): {e}")

    def _save_tables(self) -> None:
        """Export all processed tables to BIOM format files in parallel."""
        base = self.output_dir / "merged" / "table"    
        # Prepare export tasks
        export_tasks = []
        for table_type, levels in self.tables.items():
            table_dir = base / table_type
            table_dir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                out = table_dir / f"{level}.biom"  # Simplified filename
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out))
        # Use ThreadPoolExecutor for parallel exports
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for table, out_path in export_tasks:
                future = executor.submit(export_h5py, table, out_path)
                futures[future] = (table_type, level, out_path)
            with get_progress_bar() as progress:
                task_id = progress.add_task(_format_task_desc("Exporting tables"), total=len(export_tasks))
                for future in as_completed(futures):
                    table_type, level, out_path = futures[future]
                    try:
                        future.result()
                        if self.verbose:
                            logger.debug(f"Exported {table_type}/{level} table to {out_path}")
                    except Exception as e:
                        logger.error(f"Failed to export {out_path}: {str(e)}")
                    finally:
                        progress.update(task_id, advance=1)
                        
    def _save_metadata(self) -> None:
        """Export all processed tables to BIOM format files in parallel."""
        base = self.output_dir / "merged" / "metadata"    
        # Prepare export tasks
        export_tasks = []
        for table_type, levels in self.metadata.items():
            metadata_dir = base / table_type
            metadata_dir.mkdir(parents=True, exist_ok=True)
            for level, metadata in levels.items():
                out = metadata_dir / f"{level}.tsv"  # Simplified filename
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((metadata, out))
        # Use ThreadPoolExecutor for parallel exports
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for table, out_path in export_tasks:
                future = executor.submit(export_tsv, table, out_path)
                futures[future] = (table_type, level, out_path)
            with get_progress_bar() as progress:
                task_id = progress.add_task(_format_task_desc("Exporting metadata"), total=len(export_tasks))
                for future in as_completed(futures):
                    table_type, level, out_path = futures[future]
                    try:
                        future.result()
                        if self.verbose:
                            logger.debug(f"Exported {table_type}/{level} metadata to {out_path}")
                    except Exception as e:
                        logger.error(f"Failed to export {out_path}: {str(e)}")
                    finally:
                        progress.update(task_id, advance=1)      
                        
# ==================================================================================== #

def prep_data(config: Dict, metadata: Dict, tables: Dict, project_dir: Any):
    """Prepare data for downstream analysis by applying transformations and exports.

    Args:
        config:      Configuration dictionary for processing parameters.
        metadata:    Metadata associated with the feature tables.
        tables:      Dictionary containing feature tables at different stages.
        project_dir: Project directory structure object.

    Returns:
        DownstreamDataPrepper: The processed data prepper instance.
    """
    prepper = DownstreamDataPrepper(
        config=config,
        metadata=metadata,
        tables=tables,
        project_dir=project_dir
    )
    prepper.run()
    return prepper
  
