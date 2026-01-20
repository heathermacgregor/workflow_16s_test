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
        meta: pd.DataFrame,
        output_dir: Path, 
        verbose: bool = False
    ):
        self.config, self.verbose = config, verbose
        self.meta = meta
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
        nfc_facility_data: pd.DataFrame = None,
        **kwargs
    ) -> Dict[str, Any]:
        if 'nfc_facility_data' in kwargs:
            if self.verbose:
                logger.warning(
                    "Duplicate 'nfc_facility_data' argument in kwargs. "
                    "Using explicit value."
                )
            del kwargs['nfc_facility_data']
        if not self.maps_config.get('enabled', False):
            return {}
        color_columns = self.color_columns
        # Safely check for valid facilities data
        has_valid_facility_data = (
            nfc_facility_data is not None 
            and not nfc_facility_data.empty
        )
        if has_valid_facility_data:
            color_columns.append('facility_match')
        meta = self.meta
        valid_columns = [col for col in color_columns if col in meta]
        missing = set(self.color_columns) - set(valid_columns)
        if missing and self.verbose:
            logger.warning(f"Missing columns in metadata: {', '.join(missing)}")

        meta = meta.set_index('#sampleid')
      
        with get_progress_bar() as progress:
            plot_desc = f"Plotting sample maps"
            plot_task = progress.add_task(
              _format_task_desc(plot_desc), 
              total=len(valid_columns)
            )

            for col in valid_columns:
                col_desc = f"Plotting sample maps → {col}"
                progress.update(
                    plot_task, 
                    description=_format_task_desc(col_desc)
                )
                
                self.figures[col], _ = sample_map_categorical(
                    metadata=meta,
                    nfc_facilities_data=nfc_facility_data,
                    output_dir=self.output_dir,
                    color_col=col,
                    **kwargs,
                )
                
                progress.update(plot_task, advance=1)
            progress.update(
                plot_task, 
                description=_format_task_desc(plot_desc)
            )
        return self.figures
      
