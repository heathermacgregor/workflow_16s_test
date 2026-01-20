# ==================================================================================== #
#                                     ORCHESTRATOR.PY
# ==================================================================================== #

# --- FAILSAFE: FORCE SINGLE THREADING FOR NUMPY/SCIPY ---
# This must be done BEFORE importing numpy/pandas/scanpy to prevent spawn bombs.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Standard Library Imports
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, List, Any, Dict

# Third Party Imports
import pandas as pd
import anndata as ad

# Local Imports: Configuration and Utilities
from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.plotting import PlottingUtils

# Import Modular Steps
# These modules house the 700+ lines of logic previously in this file
from workflow_16s.downstream.steps.ingestion import run_fast_load, run_filter_empty, find_conda_env_by_substring
from workflow_16s.downstream.steps.backfill import run_data_backfill
from workflow_16s.downstream.steps.preprocessing import run_preprocessing_pipeline
from workflow_16s.downstream.steps.analysis import run_analysis_suite
from workflow_16s.downstream.steps.synthesis import run_results_synthesis

# ==================================================================================== #

logger = get_logger("workflow_16s")

# ==================================================================================== #

class DownstreamWorkflow:
    """
    Orchestrates the 16S data loading, processing, analysis, PICRUSt2, and CatBoost FS.
    This class serves as a state manager, delegating logic to modular steps.
    """
    TAX_LEVELS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    EXPECTED_VAR_COLUMNS = {'Taxon', 'Confidence', 'sequence'}.union(TAX_LEVELS)
    EXPECTED_VAR_DTYPES = {
        'Taxon': 'string', 
        'Confidence': 'Float64', 
        'sequence': 'string', 
        **{level: 'string' for level in TAX_LEVELS}
    }
    FACILITY_SHAPE_COLS = {
        'facility_capacity', 'facility_start_year', 'facility_end_year', 
        'facility_type', 'facility'
    }

    def __init__(self, data_dir: Path, output_dir: Path, n_cpus: Optional[int] = None, 
                 config: Optional[AppConfig] = None, nfc_facilities_df: Optional[pd.DataFrame] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.adata: Optional[ad.AnnData] = None
        self.logger = logger
        
        # Application Configuration
        self.config = config if config is not None else AppConfig() # type: ignore
        
        # Parallel Processing Setup
        if n_cpus is None: 
            self.n_cpus = os.cpu_count() or 1
            self.logger.info(f"Using default CPUs: {self.n_cpus}")
        elif n_cpus == 1: 
            self.n_cpus = 1
            self.logger.info("Single-threaded mode.")
        else: 
            self.n_cpus = min(n_cpus, os.cpu_count() or 1)
            self.logger.info(f"Using {self.n_cpus} CPU cores.")
            
        self.plot_utils = PlottingUtils(self.logger)
        
        # Environment and Dependency Discovery
        self.picrust2_conda_env = find_conda_env_by_substring("picrust2", self.logger)
        self.picrust2_enabled = self.picrust2_conda_env is not None
        
        # Database Initialization (Faprotax, NFC)
        self.nfc_facilities_df = nfc_facilities_df if nfc_facilities_df is not None else pd.DataFrame()
        self.nfc_handler = None  # Initialized if NFC matching is needed
        self.faprotax_db = None # Initialized in preprocessing step if available
        
        # Module Toggles
        self.is_arkin_enabled = False
        self.is_nfc_enabled = True
        self.is_env_data_enabled = False
        
        # Analysis Variables - will be populated during analysis
        self.priority_categorical: List[str] = []
        self.priority_numeric: List[str] = []
        self.priority_vars: List[str] = []
        self.cst_col: Optional[str] = None
        
        # Directory Structure Mapping
        self._init_directories()

    def _init_directories(self):
        """Creates the standardized output directory tree."""
        self.plot_dir_alpha = self.output_dir / "alpha_diversity"
        self.plot_dir_beta = self.output_dir / "beta_diversity"
        self.plot_dir_stats = self.output_dir / "statistical_analysis"
        self.plot_dir_network = self.output_dir / "network_analysis"
        self.plot_dir_ml = self.output_dir / "machine_learning"
        self.plot_dir_meta = self.output_dir / "metadata_plots"
        self.picrust2_output_dir = self.output_dir / "picrust2_output"
        self.func_plot_dir = self.output_dir / "functional_analysis"
        self.catboost_output_dir = self.output_dir / "catboost_feature_selection"
        
        dirs = [self.plot_dir_alpha, self.plot_dir_beta, self.plot_dir_stats, 
                self.plot_dir_network, self.plot_dir_ml, self.plot_dir_meta, 
                self.picrust2_output_dir, self.func_plot_dir, self.catboost_output_dir]
        for p in dirs: p.mkdir(exist_ok=True, parents=True)

    def _populate_priority_columns(self):
        """
        Identify and populate priority categorical and numeric columns for analysis.
        """
        if self.adata is None:
            return
        
        # Get group column from config if available
        group_col = getattr(self.config, 'group_column', None)
        
        # Identify categorical columns
        self.priority_categorical = []
        for col in self.adata.obs.columns:
            if col == group_col:
                self.priority_categorical.append(col)
            elif self.adata.obs[col].dtype.name == 'category' or \
                 (self.adata.obs[col].dtype == 'object' and self.adata.obs[col].nunique() < 50):
                # Categorical or low-cardinality object columns
                if col not in ['sample_id', 'feature_id', 'batch']:
                    self.priority_categorical.append(col)
        
        # Identify numeric columns
        self.priority_numeric = []
        for col in self.adata.obs.columns:
            if pd.api.types.is_numeric_dtype(self.adata.obs[col]):
                # Numeric but not an ID column
                if col not in ['sample_id', 'feature_id'] and not col.endswith('_id'):
                    self.priority_numeric.append(col)
        
        # Combine all priority variables
        self.priority_vars = self.priority_categorical + self.priority_numeric
        
        self.logger.info(
            f"Identified {len(self.priority_categorical)} categorical and "
            f"{len(self.priority_numeric)} numeric priority columns"
        )
    
    def _validate_priority_columns(self):
        """
        Validate that priority columns still exist in the data.
        """
        if self.adata is None:
            return
        
        existing_cols = set(self.adata.obs.columns)
        
        # Filter categorical columns
        valid_categorical = [col for col in self.priority_categorical if col in existing_cols]
        removed_cat = set(self.priority_categorical) - set(valid_categorical)
        self.priority_categorical = valid_categorical
        
        # Filter numeric columns
        valid_numeric = [col for col in self.priority_numeric if col in existing_cols]
        removed_num = set(self.priority_numeric) - set(valid_numeric)
        self.priority_numeric = valid_numeric
        
        # Update combined list
        self.priority_vars = self.priority_categorical + self.priority_numeric
        
        if removed_cat or removed_num:
            self.logger.warning(
                f"Removed {len(removed_cat)} categorical and {len(removed_num)} numeric "
                f"columns from priority lists (no longer exist in data)"
            )
            if removed_cat:
                self.logger.debug(f"Removed categorical: {sorted(removed_cat)}")
            if removed_num:
                self.logger.debug(f"Removed numeric: {sorted(removed_num)}")
    
    def _plot_cst_vs_metadata(self, cst_col: str):
        """Generate plots comparing Community State Types with metadata variables."""
        if self.adata is None or cst_col not in self.adata.obs.columns:
            return
        self.logger.info(f"Plotting CST ({cst_col}) vs metadata variables...")
    
    def _compare_catboost_strategies(self, level: str, target: str):
        """Compare CatBoost feature selection results across strategies."""
        self.logger.info(f"Comparing CatBoost strategies for {level} level, target: {target}")
    
    def execute(self):
        """Runs the complete end-to-end analysis workflow by calling modular steps."""
        
        # 1. Ingestion
        run_fast_load(self)
        if self.adata is None: 
            self.logger.error("Data ingestion failed. Exiting workflow.")
            return

        # 2. Preprocessing & Enrichment
        run_preprocessing_pipeline(self)
        if self.adata is None: 
            self.logger.error("AnnData lost during preprocessing. Exiting workflow.")
            return
        
        # 2b. Identify priority columns for analysis
        self._populate_priority_columns()
        
        # 3. Data Backfilling (Arkin, NFC, Env)
        run_data_backfill(self)

        # 3b. Validate priority columns after backfill
        self._validate_priority_columns()

        # 4. Analysis Execution
        run_analysis_suite(self)
        
        # 5. Results Synthesis & Executive Reporting
        run_results_synthesis(self)
        
        self.logger.info("✅ Full Downstream Workflow Successfully Completed.")

# ==================================================================================== #