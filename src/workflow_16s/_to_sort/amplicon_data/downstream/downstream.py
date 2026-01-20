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

# Third-Party Imports
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from skbio.stats.ordination import OrdinationResults
from statsmodels.stats.multitest import multipletests
from biom.table import Table

# Visualization imports (moved to separate section)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import umap

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.statistical_analyses import (
    run_statistical_tests_for_group, TopFeaturesAnalyzer
)
from workflow_16s.amplicon_data.top_features import top_features_plots
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.amplicon_data.downstream.alpha import AlphaDiversity
from workflow_16s.amplicon_data.downstream.beta import Ordination
from workflow_16s.amplicon_data.downstream.feature_selection import FeatureSelection
from workflow_16s.amplicon_data.downstream.input import DownstreamDataLoader as InputData
from workflow_16s.amplicon_data.downstream.maps import Maps
from workflow_16s.amplicon_data.downstream.tables import PrepData
from workflow_16s.amplicon_data.downstream.stats import run_statistical_analysis_with_loading

# ================================= CONFIGURATION & CONSTANTS ========================= #

logger = logging.getLogger("workflow_16s")
umap_lock = threading.Lock()  # Global lock for UMAP operations

# Default analysis parameters
DEFAULT_TOP_N_FEATURES = 20
DEFAULT_NETWORK_CORRELATION_THRESHOLD = 0.3
DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.5

# ================================= UTILITY CLASSES ================================== #

class AnalysisConfig:
    """Configuration container for downstream analysis parameters."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def is_enabled(self, module: str) -> bool:
        """Check if a specific analysis module is enabled."""
        return self.config.get(module, {}).get('enabled', False)
    
    def get_parameter(self, module: str, parameter: str, default: Any = None) -> Any:
        """Get a specific parameter for an analysis module."""
        return self.config.get(module, {}).get(parameter, default)

class ResultsContainer:
    """Container for organizing analysis results."""
    
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.tables: Dict[str, Any] = {}
        self.maps: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}
        self.ordination: Dict[str, Any] = {}
        self.top_features: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.analysis_statistics: Dict[str, Any] = {}

# ================================= FUNCTIONAL ANNOTATION ========================== #

class FunctionalAnnotation:
    """Handles FAPROTAX functional annotations for features."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db = None
        self._faprotax_cache: Dict[str, Any] = {}
        
        if self.config.get("faprotax", {}).get('enabled', False):
            self.db = get_faprotax_parsed()

    def _get_cached_faprotax(self, taxon: str) -> List[str]:
        """Get FAPROTAX functions for a taxon with caching."""
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon=taxon, 
                faprotax_db=self.db, 
                include_references=False
            )
        return self._faprotax_cache[taxon]
    
    def annotate_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Annotate a list of features with functional information."""
        if not self.db:
            logger.warning("FAPROTAX database not loaded")
            return {feature: [] for feature in features}
            
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_feature = {
                executor.submit(self._get_cached_faprotax, feature): feature 
                for feature in features
            }
            
            with get_progress_bar() as progress:
                task_desc = "Annotating features with functional information"
                task_desc_fmt = _format_task_desc(task_desc)
                task = progress.add_task(description=task_desc_fmt, total=len(features))
                
                for future in as_completed(future_to_feature):
                    feature = future_to_feature[future]
                    try:
                        results[feature] = future.result()
                    except Exception as e:
                        logger.error(f"Error annotating feature {feature}: {e}")
                        results[feature] = []
                    progress.update(task, advance=1)
        
        return results

# ================================= MAIN DOWNSTREAM CLASS ========================== #

class Downstream:
    """Main class for orchestrating 16S amplicon data analysis pipeline."""
    
    MODE_CONFIG = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False,
        # Result loading parameters
        load_existing_results: bool = True,
        max_result_age_hours: Optional[float] = None,
        force_recalculate_stats: List[str] = None,
        invalidate_results_patterns: List[str] = None
    ):
        # Initialize core attributes
        self.config = AnalysisConfig(config)
        self.verbose = verbose
        self.project_dir = project_dir
        self.output_dir = self.project_dir.final
        self.existing_subsets = existing_subsets

        # Result loading configuration
        self.load_existing_results = load_existing_results
        self.max_result_age_hours = max_result_age_hours
        self.force_recalculate_stats = force_recalculate_stats or []
        self.invalidate_results_patterns = invalidate_results_patterns or []

        # Initialize analysis settings
        self._setup_analysis_mode()
        self.group_columns = config.get("group_columns", [])
        
        # Initialize result containers
        self.results = ResultsContainer()
        
        # Initialize analysis components
        self.functional_annotation = FunctionalAnnotation(config)
        
        # Execute pipeline
        self._execute_pipeline()
    
    def _setup_analysis_mode(self) -> None:
        """Setup analysis mode based on configuration."""
        default_mode = self.config.config.get("target_subfragment_mode", 
                                            constants.DEFAULT_MODE)
        self.mode = 'genus' if default_mode == 'any' else 'asv'
        
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {list(self.MODE_CONFIG.keys())}")
    
    def _execute_pipeline(self) -> None:
        """Execute the complete analysis pipeline."""
        logger.info("Starting downstream analysis pipeline...")
        
        try:
            # Data loading and preparation
            self._load_and_prepare_data()
            
            # Run analyses based on configuration
            self._run_enabled_analyses()
            
            # Generate summary
            self._log_analysis_summary()
            
            logger.info("Downstream analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def _load_and_prepare_data(self) -> None:
        """Load and prepare data for analysis."""
        logger.info("Loading and preparing data...")
        
        # Load data
        data_loader = InputData(
            self.config.config, self.mode, self.project_dir, self.existing_subsets
        )
        self.results.metadata = data_loader.metadata
        self.results.tables = data_loader.tables
        self.nfc_facilities = data_loader.nfc_facilities
        
        # Prepare data for analysis
        data_prep = PrepData(
            self.config.config, 
            self.results.tables, 
            self.results.metadata, 
            self.mode, 
            self.project_dir
        )
        self.results.metadata = data_prep.metadata
        self.results.tables = data_prep.tables

    def _run_enabled_analyses(self) -> None:
        """Run all enabled analysis modules."""
        analysis_modules = [
            ('maps', self._run_sample_maps),
            ('stats', self._run_statistical_analysis),
            ('alpha_diversity', self._run_alpha_diversity),
            ('ordination', self._run_beta_diversity),
            ('ml', self._run_ml_feature_selection),
            ('top_features', self._run_top_features_analysis),
        ]
        
        for module_name, analysis_method in analysis_modules:
            if self.config.is_enabled(module_name):
                logger.info(f"Running {module_name} analysis...")
                try:
                    analysis_method()
                except Exception as e:
                    logger.error(f"Error in {module_name} analysis: {e}")
                    if self.verbose:
                        raise
            else:
                logger.info(f"{module_name} analysis disabled in configuration")

    def _run_sample_maps(self) -> None:
        """Generate sample maps if enabled."""
        maps = Maps(
            self.config.config, 
            self.results.metadata["raw"]["genus"], 
            Path(self.output_dir) / 'sample_maps', 
            self.verbose
        )
        maps.generate_sample_maps(nfc_facility_data=self.nfc_facilities)
        self.results.maps = maps.maps

    def _run_statistical_analysis(self) -> None:
        """Run statistical analysis with result loading."""
        logger.info("Statistical analysis configuration:")
        logger.info(f"  - Load existing results: {self.load_existing_results}")
        logger.info(f"  - Max file age: {self.max_result_age_hours} hours" 
                   if self.max_result_age_hours else "  - No age limit")
        logger.info(f"  - Force recalculate patterns: {self.force_recalculate_stats}")

        with run_statistical_analysis_with_loading(
            config=self.config.config,
            tables=self.results.tables,
            metadata=self.results.metadata,
            mode=self.mode,
            group_columns=self.group_columns,
            project_dir=self.project_dir,
            load_existing=self.load_existing_results,
            max_file_age_hours=self.max_result_age_hours,
            force_recalculate=self.force_recalculate_stats
        ) as stats:
            # Validate configuration
            self._validate_statistical_configuration(stats)
            
            # Get and log analysis information
            self._log_statistical_analysis_info(stats)
            
            # Store results
            self.stats_obj = stats
            self.results.stats = self._compile_statistical_results(stats)

    def _validate_statistical_configuration(self, stats) -> None:
        """Validate statistical analysis configuration."""
        issues = stats.validate_configuration()
        
        if issues['errors']:
            logger.error("Configuration errors:")
            for error in issues['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Statistical analysis configuration validation failed")
        
        if issues['warnings']:
            logger.warning("Configuration warnings:")
            for warning in issues['warnings']:
                logger.warning(f"  - {warning}")

    def _log_statistical_analysis_info(self, stats) -> None:
        """Log statistical analysis information."""        
        summary = stats.get_summary_statistics()
        logger.info(f"Statistical Analysis Summary:")
        logger.info(f"  - Total tests run: {summary['total_tests_run']}")
        logger.info(f"  - Group columns analyzed: {len(summary['group_columns_analyzed'])}")
        
        # Log loading performance
        load_stats = summary.get('performance_metrics', {}).get('load_statistics', {})
        if load_stats:
            self._log_loading_performance(load_stats)

    def _log_loading_performance(self, load_stats: Dict) -> None:
        """Log result loading performance statistics."""
        total_tasks = load_stats.get('total_tasks', 0)
        loaded_tasks = load_stats.get('loaded_from_files', 0)
        calculated_tasks = load_stats.get('calculated_fresh', 0)
        
        if total_tasks > 0:
            load_percentage = (loaded_tasks / total_tasks) * 100
            logger.info(f"  - Results loaded from files: {loaded_tasks}/{total_tasks} ({load_percentage:.1f}%)")
            logger.info(f"  - Results calculated fresh: {calculated_tasks}/{total_tasks} ({100-load_percentage:.1f}%)")

    def _compile_statistical_results(self, stats) -> Dict[str, Any]:
        """Compile statistical analysis results."""
        summary = stats.get_summary_statistics()
        
        top_features = stats.get_top_features_across_tests()
        
        recommendations = stats.get_analysis_recommendations()

        run_comp_anal = self.config.config['stats']['comprehensive_analysis'].get('enabled', True)
        if run_comp_anal:
            comprehensive_analysis = stats.run_comprehensive_analysis()
        
        results = {
            'test_results': stats.results,
            'top_features': top_features,
            'summary': summary,
            'recommendations': recommendations,
            'comprehensive_analysis': comprehensive_analysis,
            'load_statistics': stats.get_load_report()
        }
        return results

    def _run_alpha_diversity(self) -> None:
        """Run Alpha Diversity Analysis."""
        alpha = AlphaDiversity(self.config.config, self.results.metadata, self.results.tables)
        alpha.run(output_dir=self.output_dir)
        
        self.results.alpha_diversity = alpha.results

    def _run_beta_diversity(self) -> None:
        """Run Beta Diversity (Ordination) Analysis."""
        results = {}
        for group_column in self.group_columns:
            beta = Ordination(
                self.config.config, 
                self.results.metadata, 
                self.results.tables, 
                group_column['name'], 
                self.verbose
            )
            beta.run(output_dir=self.output_dir)
            results[group_column['name']] = beta.results
            
        self.results.ordination = results

    def _run_ml_feature_selection(self) -> None:
        """Run Machine Learning Feature Selection (with CatBoost)."""
        results = {}
        for group_column in self.group_columns:
            if group_column.get('type') == 'bool':
                fs = FeatureSelection(
                    self.config.config, 
                    self.results.metadata, 
                    self.results.tables, 
                    group_column['name'], 
                    self.verbose
                )
                fs.run(output_dir=self.output_dir)
                results[group_column['name']] = fs.models
                
        self.results.models = results

    def _run_top_features_analysis(self) -> None:
        """Run top features analysis."""
        self.results.top_features = {
            "stats": {},
            "models": {}
        }
        
        # Process top features from statistical analysis
        if self.config.is_enabled('stats') and self.results.stats:
            for group_column in self.group_columns:
                self._process_statistical_top_features(group_column)

            logger.info("Plotting plots for top features")
            top_features_with_plots = top_features_plots(
                output_dir=self.output_dir,
                config=self.config,
                top_features=self.results.top_features.stats,
                tables=self.results.tables,
                meta=self.results.metadata,
                nfc_facilities=self.results.nfc_facilities,
                verbose=self.verbose
            )
            logger.info(top_features_with_plots)
            #self.results.top_features = top_features_with_plots
            
        # Process top features from ML feature selection
        if self.config.is_enabled('ml') and self.results.models:
            for group_column in self.group_columns:
                self._process_ml_top_features(group_column)
                
            logger.info("Plotting plots for top features")
            top_features_with_plots = top_features_plots(
                output_dir=self.output_dir,
                config=self.config,
                top_features=self.results.top_features.models,
                tables=self.results.tables,
                meta=self.results.metadata,
                nfc_facilities=self.results.nfc_facilities,
                verbose=self.verbose
            )
            logger.info(top_features_with_plots)
            #self.results.top_features = top_features_with_plots

    def _process_statistical_top_features(self, group_column: Dict) -> None:
        """Process top features from statistical analysis."""
        n_features = self.config.get_parameter('top_features', 'n', DEFAULT_TOP_N_FEATURES)
        
        if not self._validate_group_column_for_top_features(group_column):
            return
            
        if not self.results.stats['test_results'].get(group_column['name']):
            logger.warning(f"No statistics calculated for group '{group_column['name']}'")
            return
            
        self.results.top_features["stats"][group_column['name']] = {}
        
        # Extract and rank features
        all_features = self._extract_statistical_features(group_column)
        if not all_features:
            return
            
        # Split by effect direction and rank
        positive_features = [f for f in all_features if f["effect"] > 0]
        negative_features = [f for f in all_features if f["effect"] < 0]
        
        positive_features.sort(key=lambda d: (-d["effect"], d["p_value"]))
        negative_features.sort(key=lambda d: (d["effect"], d["p_value"]))

        positive_features_n = positive_features[:n_features]
        negative_features_n = negative_features[:n_features]

        positive_features_df = pd.DataFrame(positive_features_n) if positive_features_n else pd.DataFrame()
        negative_features_df = pd.DataFrame(negative_features_n) if negative_features_n else pd.DataFrame()
        
        # Store results
        values = group_column.get('values', [True, False])
        self.results.top_features["stats"][group_column['name']][values[0]] = positive_features_df
        self.results.top_features["stats"][group_column['name']][values[1]] = negative_features_df
        
        logger.info(f"Top features for {group_column['name']}: "
                    f"{values[0]} ({len(positive_features)}), {values[1]} ({len(negative_features)})")

    def _validate_group_column_for_top_features(self, group_column: Dict) -> bool:
        """Validate group column for top features analysis."""
        if not group_column.get('values'):
            if group_column.get('type') == 'bool':
                group_column['values'] = [True, False]
            else:
                logger.warning(f"Group column values not found for {group_column.get('name')}")
                return False
        
        if len(group_column['values']) != 2:
            logger.warning(f"Group column must have exactly 2 values, got {len(group_column['values'])}")
            return False
            
        return True

    def _extract_statistical_features(self, group_column: Dict) -> List[Dict]:
        """Extract significant features from statistical tests."""
        all_features = []

        effect_direction_terminology_config = {
            "verbose": {"pos": "positive", "neg": "negative"},
            "symbols": {"pos": "+", "neg": "-"},
            "arrows": {"pos": "🢁", "neg": "🢃"}
        }
        effect_direction_selector = "verbose"
        pos = effect_direction_terminology_config[effect_direction_selector]["pos"]
        neg = effect_direction_terminology_config[effect_direction_selector]["neg"]
        
        with self.stats_obj as stats:
            test_results = self.results.stats['test_results'][group_column['name']]
            
            for table_type, levels in test_results.items():
                for level, tests in levels.items():
                    for test_name, df in tests.items():
                        if df is None or not isinstance(df, pd.DataFrame) or "p_value" not in df.columns:
                            continue
                            
                        # Get significant features
                        sig_df = df[df["p_value"] < DEFAULT_P_VALUE_THRESHOLD].copy()
                        if sig_df.empty:
                            continue
                            
                        # Calculate effect sizes
                        sig_df["effect"] = sig_df.apply(
                            lambda row: stats.get_effect_size(test_name, row), axis=1
                        )
                        sig_df = sig_df.dropna(subset=["effect"])

                        
                        # Add features to list
                        for _, row in sig_df.iterrows():
                            all_features.append({
                                "Feature": row["feature"],
                                "Column": group_column['name'],
                                "Table Type": table_type,
                                "Level": level,
                                "Method Type": "statistical_test",
                                "Method": test_name,
                                "Effect Size": row["effect"],
                                "P-value": row["p_value"],
                                "Effect Direction": pos if row["effect"] > 0 else neg,
                            })
        
        return all_features

    def _process_ml_top_features(self, group_column: Dict) -> None:
        """Process top features from ML models."""
        n_features = self.config.get_parameter('top_features', 'n', DEFAULT_TOP_N_FEATURES)
        
        if not self.results.models.get(group_column['name']):
            logger.warning(f"No ML models for group '{group_column['name']}'")
            return
        
        features_summary = []
        models_data = self.results.models[group_column['name']]
        
        for table_type, levels in models_data.items():
            for level, methods in levels.items():
                for method, result in methods.items():
                    if not self._validate_ml_result(result, group_column['name'], table_type, level, method):
                        continue
                    
                    # Extract feature importance
                    feat_imp = result.get("feature_importances", {})
                    top_features = result.get("top_features", [])
                    
                    for i, feat in enumerate(top_features[:n_features], 1):
                        importance = feat_imp.get(feat, 0)
                        features_summary.append({
                            "Feature": feat,
                            "Column": group_column['name'],
                            "Table Type": table_type,
                            "Level": level,
                            "Method Type": "feature_selection",
                            "Method": method,
                            "Rank": i,
                            "Importance": f"{importance:.4f}" if isinstance(importance, (int, float)) else "N/A"
                        })
        
        features_df = pd.DataFrame(features_summary) if features_summary else pd.DataFrame()
        self.results.top_features["models"][group_column['name']] = features_df

    def _validate_ml_result(self, result: Any, group_name: str, table_type: str, 
                            level: str, method: str) -> bool:
        """Validate ML model result structure."""
        if not result or not isinstance(result, dict):
            logger.warning(f"Invalid result for {group_name}/{table_type}/{level}/{method}")
            return False
            
        if "top_features" not in result:
            logger.error(f"Missing 'top_features' in {group_name}/{table_type}/{level}/{method}")
            return False
            
        return True

    def _log_analysis_summary(self) -> None:
        """Log comprehensive analysis summary."""
        logger.info("=" * 60)
        logger.info("DOWNSTREAM ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        # Log enabled/disabled modules
        self._log_module_status()
        
        # Log data dimensions
        self._log_data_dimensions()
        
        # Log result loading statistics
        if hasattr(self, 'results') and self.results.stats:
            self._log_result_loading_stats()
        
        logger.info("=" * 60)

    def _log_module_status(self) -> None:
        """Log which analysis modules were enabled/disabled."""
        modules = [
            ('stats', 'Statistical Analysis'),
            ('alpha_diversity', 'Alpha Diversity'),
            ('ordination', 'Beta Diversity/Ordination'),
            ('ml', 'Machine Learning Feature Selection'),
            ('maps', 'Sample Maps'),
            ('faprotax', 'Functional Annotation')
        ]
        
        enabled = [name for key, name in modules if self.config.is_enabled(key)]
        disabled = [name for key, name in modules if not self.config.is_enabled(key)]
        
        logger.info(f"Enabled modules: {', '.join(enabled)}")
        if disabled:
            logger.info(f"Disabled modules: {', '.join(disabled)}")

    def _log_data_dimensions(self) -> None:
        """Log data dimensions summary."""
        if not (self.results.tables and self.results.metadata):
            return
            
        logger.info("Data Summary:")
        for table_type in self.results.tables:
            for level in self.results.tables[table_type]:
                table = self.results.tables[table_type][level]
                metadata = self.results.metadata[table_type][level]
                logger.info(f"  - {table_type}/{level}: {table.shape[1]} samples, {table.shape[0]} features")

    def _log_result_loading_stats(self) -> None:
        """Log result loading performance statistics."""
        if 'load_statistics' not in self.results.stats:
            return
            
        load_stats = self.results.stats['load_statistics']
        summary_info = self.results.stats.get('summary', {})
        
        if summary_info:
            logger.info("Statistical Analysis Performance:")
            logger.info(f"  - Total tasks: {summary_info.get('total_tasks', 'N/A')}")
            logger.info(f"  - Loaded from cache: {summary_info.get('loaded_from_files', 'N/A')}")
            logger.info(f"  - Calculated fresh: {summary_info.get('calculated_fresh', 'N/A')}")

    # ========================== RESULT MANAGEMENT METHODS ========================= #

    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive analysis report."""
        return {
            'config': self.config.config,
            'mode': self.mode,
            'group_columns': self.group_columns,
            'load_settings': {
                'load_existing_results': self.load_existing_results,
                'max_result_age_hours': self.max_result_age_hours,
                'force_recalculate_stats': self.force_recalculate_stats,
                'invalidate_results_patterns': self.invalidate_results_patterns
            },
            'results_summary': self._generate_results_summary()
        }

    def _generate_results_summary(self) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        summary = {}
        
        result_modules = [
            ('stats', self.results.stats),
            ('alpha_diversity', self.results.alpha_diversity),
            ('ordination', self.results.ordination),
            ('models', self.results.models)
        ]
        
        for module_name, module_results in result_modules:
            if module_results:
                summary[module_name] = {'enabled': True}
                if module_name == 'stats' and 'load_statistics' in module_results:
                    summary[module_name]['load_statistics'] = module_results['load_statistics']
        
        return summary

    def invalidate_and_rerun_stats(self, patterns: List[str]) -> None:
        """Invalidate specific results and rerun statistical analysis."""
        logger.info(f"Invalidating and recalculating statistical results for patterns: {patterns}")
        
        # Invalidate existing results
        stats_dir = self.project_dir.final / 'stats'
        total_deleted = 0
        for pattern in patterns:
            deleted_count = self._delete_matching_results(stats_dir, pattern)
            total_deleted += deleted_count
        
        logger.info(f"Invalidated {total_deleted} result files")
        
        # Rerun statistical analysis
        if self.config.is_enabled('stats'):
            original_patterns = self.force_recalculate_stats.copy()
            self.force_recalculate_stats.extend(patterns)
            
            try:
                self._run_statistical_analysis()
                logger.info("Statistical analysis completed successfully")
            finally:
                self.force_recalculate_stats = original_patterns

    def _delete_matching_results(self, stats_dir: Path, pattern: str) -> int:
        """Delete result files matching a specific pattern."""
        if not stats_dir.exists():
            return 0
            
        deleted_count = 0
        
        for group_dir in stats_dir.iterdir():
            if not group_dir.is_dir():
                continue
                
            for table_dir in group_dir.iterdir():
                if not table_dir.is_dir():
                    continue
                    
                for level_dir in table_dir.iterdir():
                    if not level_dir.is_dir():
                        continue
                        
                    for result_file in level_dir.glob('*.tsv'):
                        full_path = f"{group_dir.name}_{table_dir.name}_{level_dir.name}_{result_file.stem}"
                        
                        if self._pattern_matches(pattern, full_path, result_file, table_dir, level_dir, group_dir):
                            result_file.unlink()
                            deleted_count += 1
                            
                            # Also delete correlation matrices for network analysis
                            if result_file.stem == 'network_analysis':
                                corr_file = level_dir / f"{result_file.stem}_correlation_matrix.tsv"
                                if corr_file.exists():
                                    corr_file.unlink()
                                    deleted_count += 1
        
        return deleted_count

    def _pattern_matches(self, pattern: str, full_path: str, result_file: Path, 
                        table_dir: Path, level_dir: Path, group_dir: Path) -> bool:
        """Check if a file matches the deletion pattern."""
        return (pattern in full_path or 
                pattern == result_file.stem or
                pattern == f"{table_dir.name}_{level_dir.name}" or
                pattern == group_dir.name or
                pattern == table_dir.name)
