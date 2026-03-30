# workflow_16s/downstream/workflow.py
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
import asyncio
from pathlib import Path
from typing import Optional, List, Any, Dict
import psutil

# Third Party Imports
import pandas as pd
import anndata as ad
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

# Local Imports: Configuration and Utilities
from workflow_16s.config import AppConfig
from workflow_16s.utils.ui.logger import get_logger
from workflow_16s.utils.telemetry import TelemetryCollector
from workflow_16s.visualization.utils import PlottingUtils
from workflow_16s.downstream.dashboard_enhanced import create_dashboard_with_telemetry
from workflow_16s.downstream.visualization.orchestrator import ReportOrchestrator

# Import Modular Steps
from workflow_16s.downstream.steps.ingestion import load_data, cleanup, find_conda_env_by_substring
from workflow_16s.downstream.steps.backfill import run_data_backfill
from workflow_16s.downstream.steps.preprocessing import run_preprocessing_pipeline
from workflow_16s.downstream.steps.enrichment_proxy_columns import run_proxy_columns_enrichment, generate_proxy_columns_report
from workflow_16s.downstream.steps.analysis import run_analysis_suite
from workflow_16s.downstream.steps.synthesis import run_results_synthesis
from workflow_16s.api.environmental_data.nuclear_fuel_cycle.main import NFCFacilitiesHandler

# ==================================================================================== #

class LiveResourceMonitor:
    """A persistent dashboard pane for system metrics using Rich."""
    def __init__(self):
        self.layout = Layout(); from workflow_16s.utils.progress import get_progress_bar; self.progress = get_progress_bar()
        
    def __rich__(self):
        """This method is called automatically by Rich Live to redraw the UI."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        
        try:
            load = os.getloadavg()
            load_str = f"{load[0]:.1f}, {load[1]:.1f}, {load[2]:.1f}"
        except AttributeError:
            load_str = "N/A" # For Windows compatibility if needed
            
        cpu_color = "red" if cpu > 90 else "yellow" if cpu > 70 else "green"
        ram_color = "red" if ram.percent > 90 else "green"

        text = Text()
        text.append("System Status (thar)\n", style="bold white")
        text.append("Load (1/5/15m): ", style="bold")
        text.append(f"{load_str}\n")
        
        text.append("CPU Usage:       ", style="bold")
        text.append(f"[{'█' * int(cpu/5)}{'░' * (20 - int(cpu/5))}] {cpu}%\n", style=cpu_color)
        
        text.append("RAM Usage:       ", style="bold")
        text.append(f"[{'█' * int(ram.percent/5)}{'░' * (20 - int(ram.percent/5))}] {ram.percent}% ({ram.used/1024**3:.1f}GB)\n", style=ram_color)
        
        from rich.console import Group; return Panel(Group(text, self.progress), title="[bold cyan]📊 Live Resource Monitor", border_style="cyan", expand=False)

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

    def __init__(
        self, 
        data_dir: Path, 
        output_dir: Path, 
        n_cpus: Optional[int] = None, 
        config: Optional[AppConfig] = None, 
        nfc_facilities_df: Optional[pd.DataFrame] = None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.adata: Optional[ad.AnnData] = None
        self.logger = get_logger("workflow_16s")
        
        # Application Configuration
        self.config = config if config is not None else AppConfig() # type: ignore
        
        # Parallel Processing Setup
        if n_cpus is None: self.n_cpus = os.cpu_count() or 1
        elif n_cpus == 1: self.n_cpus = 1
        else: self.n_cpus = min(n_cpus, os.cpu_count() or 1)
        self.logger.info(f"Using {self.n_cpus} CPU cores.")
            
        self.plot_utils = PlottingUtils(self.logger)
        
        # Environment and Dependency Discovery
        self.picrust2_conda_env = find_conda_env_by_substring("picrust2", self.logger)
        self.picrust2_enabled = self.picrust2_conda_env is not None
       
        # Module Toggles
        self.is_nfc_enabled = self.config.nfc_facilities.enabled

        # NFC (Optional)
        self.nfc_handler, self.nfc_facilities_df = None, pd.DataFrame()
        if self.is_nfc_enabled:
            self.nfc_handler = NFCFacilitiesHandler(self.config)
            self.logger.info("Fetching NFC facility data...")
            try:
                # FIX: Execute async fetch safely inside sync __init__ using asyncio.run
                self.nfc_facilities_df = asyncio.run(self.nfc_handler.nfc_facilities())
                if not self.nfc_facilities_df.empty:
                    self.logger.info(f"Successfully loaded {len(self.nfc_facilities_df)} NFC facilities.")
                else:
                    self.logger.warning("NFC handler ran but returned no data. Continuing workflow without NFC facility data.")
            except Exception as e:
                self.logger.error(f"Failed to load NFC facilities: {e}. Continuing without NFC data.")
        
        # Module Toggles
        self.is_arkin_enabled = self.config.upstream.get('arkin_agents', {}).get('enabled', False)
        self.is_env_data_enabled = True
        self.is_gee_enabled = False

        # GEE: Check NEW config (apis.gee.enabled) first, fall back to deprecated setting
        try:
            apis_cfg = getattr(self.config, 'apis', None)
            if apis_cfg:
                gee_cfg = getattr(apis_cfg, 'gee', None)
                if gee_cfg:
                    self.is_gee_enabled = getattr(gee_cfg, 'enabled', False)
        except (AttributeError, TypeError):
            pass  # Fall back to deprecated setting

        # Fall back to deprecated downstream.gee_enrichment_enabled if not set
        if not self.is_gee_enabled:
            downstream_cfg = getattr(self.config, 'downstream', None)
            if downstream_cfg:
                if isinstance(downstream_cfg, dict):
                    self.is_gee_enabled = downstream_cfg.get('gee_enrichment_enabled', False)
                else:
                    self.is_gee_enabled = getattr(downstream_cfg, 'gee_enrichment_enabled', False)
        # Environmental data enrichment toggles (default: enabled) - robust access for dict/Pydantic
        self.is_csu_soil_enabled = True
        self.is_geochemical_enabled = True
        try:
            env_data_cfg = getattr(self.config, 'environmental_data', None)
            if env_data_cfg and isinstance(env_data_cfg, dict):
                self.is_csu_soil_enabled = env_data_cfg.get('csu_soil', {}).get('enabled', True)
                self.is_geochemical_enabled = env_data_cfg.get('geochemical', {}).get('enabled', True)
        except (AttributeError, TypeError):
            pass  # Use defaults if config access fails
        
        # Analysis Variables
        self.priority_categorical: List[str] = []
        self.priority_numeric: List[str] = []
        self.priority_vars: List[str] = []
        self.cst_col: Optional[str] = None
        
        # Initialize Telemetry & Report Orchestrator
        self.telemetry = TelemetryCollector()
        batch_render = getattr(self.config.visualization, 'batch_render_plots', True)
        png_scale = getattr(self.config.visualization, 'png_scale', 2)
        self.report = ReportOrchestrator(
            output_dir=self.output_dir,
            logger=self.logger,
            batch_render=batch_render,
            png_scale=png_scale,
            enable_master_report=getattr(self.config.visualization, 'master_report_enabled', True)
        )
        self.logger.info(f"📊 Report orchestrator initialized (batch_render={batch_render}, png_scale={png_scale}x)")
        
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
        if self.adata is None: return
        group_col = getattr(self.config, 'group_column', None)
        
        self.priority_categorical = []
        for col in self.adata.obs.columns:
            if col == group_col:
                self.priority_categorical.append(col)
            elif self.adata.obs[col].dtype.name == 'category' or \
                 (self.adata.obs[col].dtype == 'object' and self.adata.obs[col].nunique() < 50):
                if col not in ['sample_id', 'feature_id', 'batch']:
                    self.priority_categorical.append(col)
        
        self.priority_numeric = []
        for col in self.adata.obs.columns:
            if pd.api.types.is_numeric_dtype(self.adata.obs[col]):
                if col not in ['sample_id', 'feature_id'] and not col.endswith('_id'):
                    self.priority_numeric.append(col)
        
        self.priority_vars = self.priority_categorical + self.priority_numeric
    
    def _validate_priority_columns(self):
        if self.adata is None: return
        existing_cols = set(self.adata.obs.columns)
        
        valid_categorical = [col for col in self.priority_categorical if col in existing_cols]
        self.priority_categorical = valid_categorical
        
        valid_numeric = [col for col in self.priority_numeric if col in existing_cols]
        self.priority_numeric = valid_numeric
        
        self.priority_vars = self.priority_categorical + self.priority_numeric
    
    def _plot_cst_vs_metadata(self, cst_col: str):
        if self.adata is None or cst_col not in self.adata.obs.columns: return
        self.logger.info(f"📊 Plotting CST ({cst_col}) vs. metadata variables...")
    
    def _compare_catboost_strategies(self, level: str, target: str):
        self.logger.info(f"Comparing CatBoost strategies for {level} level, target: {target}")
    
    def execute(self):
        """Runs the workflow with live Rich TUI dashboard and telemetry."""
        # Initialize enhanced dashboard with telemetry
        dashboard = create_dashboard_with_telemetry(self.telemetry, self.logger)
        
        # Render at 1 Hz instead of 2 Hz for better CPU balance
        # Properly handle KeyboardInterrupt (Ctrl+C) for clean shutdown
        try:
            with Live(dashboard, refresh_per_second=1, screen=False) as live:
                self._run_internal_logic()
                # Emit final completion event so dashboard shows completion
                self.telemetry.emit(
                    event_type='completion',
                    phase='Workflow_Complete',
                    message='✅ Workflow execution finished. Generate final reports.',
                    metrics={}
                )
                # Brief pause to let dashboard render final status
                import time
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.warning("\n⏹️  Workflow interrupted by user (Ctrl+C)")
            self.logger.info("Cleaning up and shutting down...")
            if self.adata is not None:
                # Optionally save intermediate state
                try:
                    pass  # Add cleanup if needed
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
            raise  # Re-raise to exit

    def _run_internal_logic(self):
        """The actual workflow execution steps, separated for clean wrapping with telemetry."""
        # --- PHASE 1: INGESTION ---
        self.telemetry.start_phase('Ingestion')
        load_data(self)
        if not self.adata.obs_names.is_unique:
            self.logger.warning(f"⚠️ Found {len(self.adata.obs_names) - len(set(self.adata.obs_names))} duplicate Sample IDs. Making unique...")
            self.adata.obs_names_make_unique()
        
        # --- Metadata Sanitization ---
        self.logger.info('🧪 Sanitizing metadata dtypes for ML...')
        for col in ['lat', 'lon', 'latitude', 'longitude', 'measured_ph', 'measured_temp', 'altitude_m']:
            if col in self.adata.obs.columns:
                self.adata.obs[col] = pd.to_numeric(self.adata.obs[col], errors='coerce')
       
        cleanup(self)
        if self.adata is None: 
            self.logger.error("Data ingestion failed. Exiting workflow.")
            self.telemetry.emit(event_type='error', phase='Ingestion', message='Ingestion failed', metrics={})
            return
        
        self.telemetry.end_phase('Ingestion', metrics={'n_samples': self.adata.n_obs, 'n_features': self.adata.n_vars})

        # --- PHASE 2: DATA BACKFILLING ---
        # Enrich ALL 463k samples with geochemical/environmental metadata (lightweight, uses unique coordinates)
        # This must happen BEFORE study eligibility check because many targets are geochemical_proxy_* columns
        self.telemetry.start_phase('Backfill')
        run_data_backfill(self)
        self.telemetry.end_phase('Backfill', metrics={'n_obs_columns': len(self.adata.obs.columns)})
        '''
        # --- PHASE 3: STUDY ELIGIBILITY FILTERING ---
        # Now filter to "Valid Pool" based on target criteria (≥2 classes / ≥10 samples for categorical)
        # (≥10 valid samples + variance > 0 for continuous targets)
        eligibility_mode = getattr(self.config.downstream, 'study_eligibility_mode', 'non_strict')
        min_samples_per_study = getattr(self.config.downstream, 'strict_min_samples_per_study', 10)
        
        if eligibility_mode in ['strict', 'both']:
            from workflow_16s.downstream.machine_learning.validation.check_study_eligibility import StudyEligibilityManager
            
            # Find eligible studies (multiclass, sufficient N, variance > 0 for continuous targets)
            target_col = getattr(self.config, 'group_column', 'Env_Level_1')
            if target_col and target_col in self.adata.obs.columns:
                manager = StudyEligibilityManager(self.adata, target_col=target_col, min_n=min_samples_per_study)
                eligibility_df = manager.diagnose_studies()
                passing_studies = eligibility_df[eligibility_df['Status'] == "✅ PASS"]['Study'].tolist()
                
                if eligibility_mode == 'strict':
                    # Get batch column
                    possible_batch_cols = ['batch_original', 'study_accession', 'Project', 'dataset', 'study', 'batch', 'project_id', 'study_id']
                    batch_col = None
                    for col in possible_batch_cols:
                        if col in self.adata.obs.columns:
                            batch_col = col
                            break
                    
                    if batch_col and passing_studies:
                        # Filter to only samples from eligible studies (Valid Pool)
                        eligible_mask = self.adata.obs[batch_col].isin(passing_studies)
                        self.adata = self.adata[eligible_mask].copy()
                        self.logger.info(f"✅ Strict mode: Filtered to {self.adata.n_obs} samples from {len(passing_studies)} eligible studies")
        
        # --- PHASE 4: SUBSAMPLING FOR TESTING (DISABLED FOR PROD) ---
        # TEST MODE: Draw N samples from the Valid Pool for quick testing/validation
        # Disabled by default (test_sample_limit=0) to process full datasets
        test_limit = getattr(self.config.downstream, 'test_sample_limit', 0)
        if test_limit > 0 and self.adata.n_obs > test_limit:
            self.logger.warning(f" ⚠️  TEST MODE: Subsampling to {test_limit} samples (test_sample_limit={test_limit} in config)")
            self.adata = self.adata[:test_limit, :].copy()
            self.logger.info(f"   Test set: {self.adata.n_obs} samples × {self.adata.n_vars} features")
        else:
            if test_limit == 0:
                self.logger.info(f" ✅ PRODUCTION MODE: Processing full dataset ({self.adata.n_obs} samples)")
            else:
                self.logger.info(f" ✅ Dataset already at/below limit ({self.adata.n_obs} samples)")
        '''
        # --- PHASE 5: PREPROCESSING ---
        # Apply data cleaning and transformations ONLY on the working dataset
        self.telemetry.start_phase('Preprocessing')
        run_preprocessing_pipeline(self)
        if self.adata is None: 
            self.logger.error("AnnData lost during preprocessing. Exiting workflow.")
            self.telemetry.emit(event_type='error', phase='Preprocessing', message='AnnData lost', metrics={})
            return
        
        # 2b. Identify priority columns for analysis
        self._populate_priority_columns()
        # Alias coordinate names for Geo-Enrichment (try multiple column names)
        if 'latitude' in self.adata.obs.columns and 'lat' not in self.adata.obs.columns:
            self.adata.obs['lat'] = self.adata.obs['latitude']
        elif 'LatitudeParsed' in self.adata.obs.columns and 'lat' not in self.adata.obs.columns:
            self.adata.obs['lat'] = self.adata.obs['LatitudeParsed']
        
        if 'longitude' in self.adata.obs.columns and 'lon' not in self.adata.obs.columns:
            self.adata.obs['lon'] = self.adata.obs['longitude']
        elif 'LongitudeParsed' in self.adata.obs.columns and 'lon' not in self.adata.obs.columns:
            self.adata.obs['lon'] = self.adata.obs['LongitudeParsed']
        
        self.telemetry.end_phase('Preprocessing', metrics={'n_samples': self.adata.n_obs, 'n_features': self.adata.n_vars})

        # --- PHASE 6: PROXY COLUMNS ENRICHMENT ---
        self.telemetry.start_phase('ProxyEnrichment')
        run_proxy_columns_enrichment(self)
        generate_proxy_columns_report(self, output_dir=self.output_dir)
        self.telemetry.end_phase('ProxyEnrichment', metrics={'n_obs_columns': len(self.adata.obs.columns)})

        # --- PHASE 7: ANALYSIS ---
        self.logger.info("Starting Analysis Suite...")
        self.telemetry.start_phase('Analysis')
        run_analysis_suite(self)
        self.telemetry.end_phase('Analysis', metrics={'n_samples': self.adata.n_obs})
        
        # --- PHASE 7: REPORT FINALIZATION ---
        self.logger.info("Finalizing report...")
        self.telemetry.start_phase('Report_Generation')
        report_path = self.report.finalize()
        self.telemetry.end_phase('Report_Generation', metrics={'report_path': str(report_path)})
        
        # Log execution summary
        summary = self.telemetry.get_summary()
        self.logger.info(f"✅ Workflow completed: {summary['events_collected']} events, {summary['total_runtime_seconds']:.1f}s runtime")

# ==================================================================================== #
