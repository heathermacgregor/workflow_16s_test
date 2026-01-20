# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import threading  
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.amplicon_data.alpha_diversity import AlphaDiversity
from workflow_16s.amplicon_data.beta_diversity import Ordination
from workflow_16s.amplicon_data.helpers import _init_dict_level, _ProcessingMixin
from workflow_16s.amplicon_data.feature_selection import FeatureSelection
from workflow_16s.amplicon_data.maps import Maps
from workflow_16s.amplicon_data.preprocessing import _DataLoader, _TableProcessor
from workflow_16s.amplicon_data.statistical_analyses import (
    run_statistical_tests_for_group, TopFeaturesAnalyzer
)
from workflow_16s.amplicon_data.top_features import top_features_plots
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import find_nearby_nfc_facilities

# ==================================== FUNCTIONS ===================================== #

logger = logging.getLogger("workflow_16s")
# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# ================================= DEFAULT VALUES =================================== #

def check_if_two_vals(group_column, meta):
    meta_col_exists = (
        'name' in group_column and
        isinstance(group_column['name'], str) and
        group_column['name'] in list(meta.columns)
    )

    config_is_bool = (
        meta_col_exists and
        'type' in group_column and
        isinstance(group_column['type'], str) and
        group_column.get('type', '') == 'bool'
    )

    config_has_two_vals = (
        meta_col_exists and
        'values' in group_column and 
        isinstance(group_column['values'], list) and 
        len(group_column['values']) == 2
    )

    meta_col_has_two_vals = (
        meta_col_exists and
        meta[group_column['name']].nunique() == 2
    )

    if config_is_bool:
        return [True, False]

    elif config_has_two_vals:
        return group_column['values']

    elif meta_col_has_two_vals:
        return meta[group_column['name']].unique()

    # Optional: add explicit fallback
    return None


class AmpliconData:
    """Main class for orchestrating 16S amplicon data analysis pipeline."""
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        mode: str = constants.DEFAULT_MODE, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False
    ):
        self.config = config
        self.project_dir = project_dir
        self.mode = mode
        self.existing_subsets = existing_subsets
        self.verbose = verbose
        
        # Initialize result containers
        self.tables: Dict[str, Any] = {}
        self.maps: Optional[Dict[str, Any]] = None
        
        self.stats: Dict[str, Any] = {}
        self.top_features = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}

        self.nfc_facilities: pd.Dataframe = None
        self.meta_nfc_facilities: pd.Dataframe = None
        
        logger.info("Running amplicon data analysis pipeline...")
        self._execute_pipeline()

    def _execute_pipeline(self):
        """Execute the analysis pipeline in sequence."""
        self._load_data()
        self._process_tables()
        self._run_analysis()
        
        if self.verbose:
            logger.info("AmpliconData analysis finished.")

    def _load_data(self):
        data_loader = _DataLoader(
            config=self.config, 
            mode=self.mode, 
            existing_subsets=self.existing_subsets,
            project_dir=self.project_dir, 
            verbose=self.verbose
        )
        self.meta = data_loader.meta
        self.table = data_loader.table
        self.nfc_facilities = data_loader.nfc_facilities
        self.meta_nfc_facilities = data_loader.meta_nfc_facilities

    def _process_tables(self):
        processor = _TableProcessor(
            config=self.config,
            mode=self.mode,
            meta=self.meta,
            table=self.table,
            project_dir=self.project_dir,
            output_dir=Path(self.project_dir.final),
            verbose=self.verbose
        )
        self.tables = processor.tables

    def _run_analysis(self):
        analyzer = _AnalysisManager(
            config=self.config,
            tables=self.tables,
            meta=self.meta,
            output_dir=Path(self.project_dir.final),
            verbose=self.verbose,
            nfc_facilities=self.nfc_facilities,
        )
        
        # Collect results
        self.stats = analyzer.stats
        self.alpha_diversity = analyzer.alpha_diversity
        self.ordination = analyzer.beta_diversity
        self.models = analyzer.models
        self.top_features = analyzer.top_features


class _AnalysisManager(_ProcessingMixin):
    def __init__(
        self,
        config: Dict,
        tables: Dict[str, Dict[str, Table]],
        meta: pd.DataFrame,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
        nfc_facilities: Optional[pd.DataFrame] = None
    ) -> None:
        self.config = config
        self.tables = tables
        self.meta = meta
        self.output_dir = output_dir
        self.verbose = verbose

        self.group_columns = self.config.get('group_columns', [])            

        self.group_column = self.config.get(
            'group_column', constants.DEFAULT_GROUP_COLUMN
        )
        self.group_column_values = self.config.get(
            'group_column_values', constants.DEFAULT_GROUP_COLUMN_VALUES
        )
        
        self._faprotax_cache = {}

        self.maps: Dict = {}
        self.stats: Dict[str, Any] = {}  # Nested dict: group -> 
        self.top_features: Dict[str, Dict[Any, List]] = {}  # Nested dict: group -> condition -> features
        self.alpha_diversity: Dict = {} # Nested dict: group ->
        self.beta_diversity: Dict = {} 
        self.models: Dict[str, Any] = {}

        self.nfc_facilities: pd.Dataframe = None
        self.meta_nfc_facilities: pd.Dataframe = None

        self.run()
        
    def run(self) -> None:
        self._generate_sample_maps()
        self._run_statistical_tests()
        self._identify_top_features()  
        if self.config.get('faprotax', {}).get('enabled', False):
            self._annotate_top_features()
        self._top_features_plots()
        self._run_alpha_diversity()
        self._run_beta_diversity()
        self._run_ml_feature_selection()

    # SAMPLE MAPS
    def _generate_sample_maps(self):
        if self.config.get("maps", {}).get("enabled", False):
            plotter = Maps(
                self.config, 
                self.meta, 
                Path(self.output_dir) / 'sample_maps',
                self.verbose
            )
            maps = plotter.generate_sample_maps(
                nfc_facility_data=self.nfc_facilities
            )
            plotter.maps = maps

    # ALPHA DIVERSITY
    def _run_alpha_diversity(self) -> None:
        alpha = AlphaDiversity(
            config=self.config,
            meta=self.meta,
            tables=self.tables
        )
        alpha.run(
            output_dir=self.output_dir
        )
        self.alpha_diversity = alpha.results

    # BETA DIVERSITY
    def _run_beta_diversity(self) -> None:
        beta = Ordination(
            config=self.config,
            meta=self.meta,
            tables=self.tables,
            verbose=self.verbose
        )
        beta.run(
            output_dir=self.output_dir
        )
        self.beta_diversity = beta.results

    # ML FEATURE SELECTION
    def _run_ml_feature_selection(self) -> None:
        ml = FeatureSelection(
            config=self.config,
            meta=self.meta,
            tables=self.tables,
            verbose=self.verbose
        )
        ml.run(
            output_dir=self.output_dir
        )
        self.models = ml.models

    # STATISTICS
    def _run_statistical_tests(self) -> None:
        """Run statistical tests for primary and special cases"""
        # Primary group
        for group_column in self.group_columns:
            group_column_values = check_if_two_vals(group_column, self.meta)
            if group_column_values:            
                self.stats[group_column['name']] = run_statistical_tests_for_group(
                    config=self.config,  
                    tables=self.tables,
                    meta=self.meta,
                    group_column=group_column['name'],
                    group_column_values=group_column['values'] if 'values' in group_column else [True, False],
                    output_dir=self.output_dir,
                    verbose=self.verbose
                )
                    
        # Special case: NFC facility matching
        if self.config.get('nfc_facilities', {}).get('enabled', False) and 'facility_match' in self.meta.columns:
            self.stats['facility_match'] = run_statistical_tests_for_group(
                config=self.config,
                tables=self.tables,
                meta=self.meta.dropna(subset=['facility_match']),
                group_column='facility_match',
                group_column_values=[True, False],  
                output_dir=self.output_dir,
                verbose=self.verbose
            )

    def _identify_top_features(self) -> None:
        """Identify top features for each group condition"""
        # Process primary group
        self._process_group_features(
            group_column=self.group_column,
            group_values=self.group_column_values
        )
        # Special case: NFC facility matching
        if 'facility_match' in self.meta.columns and 'facility_match' in self.stats:
            self._process_group_features(
                group_column='facility_match',
                group_values=[True, False]
            )

    def _process_group_features(self, group_column: str, group_values: List[Any]) -> None:
        """Helper to identify top features for a specific group"""
        if group_column not in self.stats or not self.stats[group_column]:
            logger.warning(
              f"No statistics calculated for group '{group_column}'. Skipping top features."
            )
            return

        # Initialize storage for this group
        self.top_features[group_column] = {}
        # Analyze top features for both conditions in the group
        analyzer = TopFeaturesAnalyzer(self.config, self.verbose) 
        features_cond1, features_cond2 = analyzer.analyze(
            self.stats[group_column], 
            group_column
        )
        
        # Store results
        self.top_features[group_column][group_values[0]] = features_cond1
        self.top_features[group_column][group_values[1]] = features_cond2
        
        logger.info(
            f"Top features for {group_column}: "
            f"{group_values[0]} ({len(features_cond1)}), "
            f"{group_values[1]} ({len(features_cond2)})"
        )

    # TOP FEATURES PLOTS
    def _top_features_plots(
        self
    ):
        if self.config.get('violin_plots', {}).get('enabled', False) or self.config.get('feature_maps', {}).get('enabled', False):
            plots = top_features_plots(
                output_dir=self.output_dir,
                config=self.config,
                top_features=self.top_features,
                tables=self.tables,
                meta=self.meta,
                nfc_facilities=self.nfc_facilities,
                verbose=self.verbose
            )
            self.top_features = plots

    # FUNCTIONAL ANNOTATION
    def _get_cached_faprotax(
        self, 
        taxon: str
    ) -> List[str]:
        fdb = get_faprotax_parsed() if self.config.get('faprotax', {}).get('enabled', False) else None
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon, fdb, include_references=False
            )
        return self._faprotax_cache[taxon]

    def _annotate_top_features(self) -> None:
        all_taxa = {
            f["feature"] for f in self.top_features[self.group_column][self.group_column_values[0]] 
            + self.top_features[self.group_column][self.group_column_values[1]]
        }
        taxa_list = list(all_taxa)  # Convert to list for stable ordering
    
        # Initialize results array
        results = [None] * len(taxa_list)
        
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self._get_cached_faprotax, taxon): idx
                for idx, taxon in enumerate(taxa_list)
            }
            with get_progress_bar() as progress:
                task = progress.add_task(
                    description=_format_task_desc("Annotating top features"), 
                    total=len(taxa_list)
                )
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    progress.update(task, advance=1)
        
        # Create final taxon map
        taxon_map = dict(zip(taxa_list, results))

        # Annotate features across all groups and conditions
        for group_dict in self.top_features.values():
            for condition, features in group_dict.items():
                for feature in features:
                    feature["faprotax_functions"] = taxon_map.get(feature["feature"], [])
