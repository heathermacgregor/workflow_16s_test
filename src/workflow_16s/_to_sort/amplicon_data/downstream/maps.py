# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd

# Local Imports
from workflow_16s import constants
from workflow_16s.figures.merged import sample_map_categorical
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =================================== FUNCTIONS ====================================== #

class Maps:
    """Generates sample map plots and stores them internally"""
    def __init__(
        self, 
        config: Dict, 
        metadata: pd.DataFrame,
        output_dir: Path, 
        verbose: bool = False
    ):
        self.config, self.verbose = config, verbose
        self.metadata = metadata
        self.output_dir = output_dir
      
        self.maps_config = self.config['maps']
        self.color_columns = self.maps_config.get(
            'color_columns',
            [
                constants.DEFAULT_DATASET_COLUMN,
                constants.DEFAULT_GROUP_COLUMN,
                "env_feature",
                "env_material",
                "country",
            ],
        )
        
        self.figures: Dict[str, Any] = {}

    def generate_sample_maps(
        self, 
        nfc_facility_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.maps_config.get('enabled', False):
            return {}
        
        # Handle NFC facility data
        if 'nfc_facility_data' in kwargs:
            if self.verbose:
                logger.warning(
                    "Duplicate 'nfc_facility_data' argument in kwargs. "
                    "Using explicit value."
                )
            del kwargs['nfc_facility_data']
        
        color_columns = self.color_columns
        if nfc_facility_data is not None:
            color_columns.append('facility_match')

        metadata = self.metadata
        # Get valid color columns
        valid_columns = [col for col in color_columns if col in metadata.columns]
        missing = set(color_columns) - set(valid_columns)
        if missing and self.verbose:
            logger.warning(f"Missing columns in metadata: {', '.join(missing)}")

        # Set index to the metadata ID column for consistency
        metadata = metadata.set_index(self.config.get('metadata_id_column', '#sampleid'))

        # Plot sample maps colored by each (valid) color column
        with get_progress_bar() as progress:
            plot_desc = "Plotting sample maps"
            plot_desc_fmt = _format_task_desc(plot_desc)
            plot_task = progress.add_task(plot_desc_fmt, total=len(valid_columns))
            for col in valid_columns:
                col_desc = f"{plot_desc} → {col}"
                col_desc_fmt = _format_task_desc(col_desc)
                progress.update(plot_task, description=col_desc_fmt)
                self.figures[col], _ = sample_map_categorical(
                    metadata=metadata,
                    nfc_facilities_data=nfc_facility_data,
                    output_dir=self.output_dir,
                    color_col=col,
                    **kwargs,
                )
                progress.update(plot_task, advance=1)
            progress.update(plot_task, description=plot_desc_fmt)
        return self.figures
        
