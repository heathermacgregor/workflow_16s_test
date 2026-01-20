# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s.constants import MODE, GROUP_COLUMNS
from workflow_16s.downstream.load_data import load_data, load_existing_data
from workflow_16s.downstream.prep_data import prep_data
from workflow_16s.downstream.stats_analysis import run_statistical_analysis
from workflow_16s.downstream.beta_diversity import run_beta_diversity

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================================================================== #

# TODO: Refactor these modules
from workflow_16s.amplicon_data.downstream.alpha import AlphaDiversity
from workflow_16s.amplicon_data.downstream.feature_selection import FeatureSelection
from workflow_16s.amplicon_data.downstream.maps import Maps

def run_sample_maps(
    config: Dict,
    metadata: pd.DataFrame,
    output_dir: Union[str, Path],
    nfc_facilities: pd.DataFrame
):
    maps = Maps(
        config=config, 
        metadata=metadata,
        output_dir=output_dir
    )
    maps.generate_sample_maps(nfc_facility_data=nfc_facilities)
    return maps.maps
    

def run_alpha_diversity_analysis(
    config: Dict,
    metadata: Dict,
    tables: Dict,
    output_dir: Union[str, Path]
):
    alpha = AlphaDiversity(config, metadata, tables)
    alpha.run(output_dir=output_dir)
    return alpha.results


def run_feature_selection(
    config: Dict,
    metadata: Dict,
    tables: Dict,
    group_columns: List,
    output_dir: Union[str, Path]
):
    results = {}
    for group_column in group_columns:
        if group_column.get('type') == 'bool':
            name = group_column['name']
            fs = FeatureSelection(config, metadata, tables, name)
            fs.run(output_dir=output_dir)
            results[name] = fs.models
    return results
    
# ==================================================================================== #

class Config:
    """Configuration container for downstream analysis parameters."""
    def __init__(self, config: Dict):
        self.config = config
        
    def is_enabled(self, module: str) -> bool:
        """Check if a specific analysis module is enabled."""
        return self.config.get(module, {}).get('enabled', False)
    
    def get_parameter(self, module: str, parameter: str, default: Any = None) -> Any:
        """Get a specific parameter for an analysis module."""
        return self.config.get(module, {}).get(parameter, default)


class Results:
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
      
# ==================================================================================== #

class DownstreamAnalyzer:
    """Main class for orchestrating 16S amplicon data analysis pipeline."""
    ModeConfig = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None
    ):
        # Initialize core attributes
        self.config = Config(config)
        self.verbose = self.config.config.get("verbose", False)
        self.project_dir = project_dir
        self.output_dir = self.project_dir.final
        self.existing_subsets = existing_subsets

        # Result loading configuration
        self.load_existing_results = self.config.config.get("load_existing_results", True)
        self.max_result_age_hours = self.config.config.get("max_result_age_hours", None)
        self.force_recalculate_stats = self.config.config.get("force_recalculate_stats", None) or []
        self.invalidate_results_patterns = self.config.config.get("invalidate_results_patterns", None) or []

        # Initialize analysis settings
        self._setup_mode()
        self.group_columns = self.config.config.get("group_columns", GROUP_COLUMNS)
        
        # Initialize result containers
        self.results = Results()
        
        # Initialize analysis components
        #self.functional_annotation = FunctionalAnnotation(config)
    
    def _setup_mode(self) -> None:
        """Setup analysis mode based on configuration."""
        default_mode = self.config.config.get("target_subfragment_mode", MODE)
        self.mode = 'genus' if default_mode == 'any' else 'asv'
        
        if self.mode not in self.ModeConfig:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {list(self.ModeConfig.keys())}")
    
    def run(self) -> None:
        """Execute the complete analysis pipeline."""
        logger.info("Starting downstream analysis pipeline...")
        try:
            # Data loading and preparation
            self._load_and_prep_data()
            # Run analyses based on configuration
            self._run_modules()
            # Generate summary
            self._log_analysis_summary()
            logger.info("Downstream analysis pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def _load_and_prep_data(self):
        # Try loading existing data if configured
        if self.config.config.get("features", {}).get("load_existing", False):
            try:
                self._load_existing_data()
                logger.info("Successfully loaded existing data.")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}. Loading new data...")
        
        # Load and prep new data 
        self._load_data()
        self._prep_data()      
            
    def _load_existing_data(self):
        data = load_existing_data(config=self.config.config, project_dir=self.project_dir)
        self.results.metadata = data.metadata
        self.results.tables = data.tables
        
    def _load_data(self) -> None:
        logger.info("Loading data...")
        data = load_data(
            config=self.config.config, project_dir=self.project_dir, 
            existing_subsets=self.existing_subsets
        )
        self.results.metadata = data.metadata
        self.results.tables = data.tables
        self.nfc_facilities = data.nfc_facilities
      
    def _prep_data(self) -> None:
        logger.info("Prepping data...")
        data = prep_data(
            config=self.config.config, metadata=self.results.metadata, 
            tables=self.results.tables, project_dir=self.project_dir
        )
        self.results.metadata = data.metadata
        self.results.tables = data.tables

    def _run_modules(self) -> None:
        """Run all enabled analysis modules."""
        analysis_modules = [
            ('maps', self._run_sample_maps), 
            ('stats', self._run_statistical_analysis),
            ('alpha_diversity', self._run_alpha_diversity), 
            ('ordination', self._run_beta_diversity),
            ('ml', self._run_feature_selection), 
            ('top_features', self._run_top_features_analysis)
        ]
        
        for module_name, module_func in analysis_modules:
            if self.config.is_enabled(module_name):
                logger.info(f"Running {module_name} analysis...")
                try:
                    module_func()
                except Exception as e:
                    logger.error(f"Error in {module_name} analysis: {e}")
                    if self.verbose:
                        raise
            else:
                logger.info(f"Skipping '{module_name}' analysis: disabled in configuration")
    
    def _run_sample_maps(self) -> None:
        """Generate sample maps if enabled."""
        self.results.maps = run_sample_maps(
            config=self.config.config, metadata=self.results.metadata["raw"]["genus"], 
            output_dir=Path(self.output_dir) / 'sample_maps', 
            nfc_facilities=self.nfc_facilities if self.nfc_facilities else pd.DataFrame()
        )
        
    def _run_statistical_analysis(self):
        self.results.stats = run_statistical_analysis(
            config=self.config.config, metadata=self.results.metadata,
            tables=self.results.tables, project_dir=self.project_dir,
            use_process_pool=True
        ).results
    
    def _run_alpha_diversity(self) -> None:
        """Run Alpha Diversity Analysis."""
        self.results.alpha_diversity = run_alpha_diversity_analysis(
            config=self.config.config, metadata=self.results.metadata, 
            tables=self.results.tables, output_dir=self.output_dir
        )        

    def _run_beta_diversity(self) -> None:
        """Run Beta Diversity (Ordination) Analysis."""
        self.results.ordination = run_beta_diversity(
            config=self.config.config, metadata=self.results.metadata,
            tables=self.results.tables, project_dir=self.project_dir,
            group_columns=self.group_columns
        )
    
    def _run_feature_selection(self) -> None:
        """Run Machine Learning Feature Selection (with CatBoost)."""
        self.results.models = run_feature_selection(
            config=self.config.config, metadata=self.results.metadata,
            tables=self.results.tables, group_columns=self.group_columns, 
            output_dir=self.output_dir
        )

    def _run_top_features_analysis(self) -> None:
        """Run top features analysis."""
        self.results.top_features = {"stats": {}, "models": {}}

    def _log_analysis_summary(self) -> None:
        """Log comprehensive analysis summary."""
        logger.info("=" * 60)
        logger.info("DOWNSTREAM ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        self._log_module_status()
        self._log_data_dimensions()
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

# ==================================================================================== #

def run_downstream(
    config: Dict, 
    project_dir: Any, 
    existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None
) -> DownstreamAnalyzer:
    analyzer = DownstreamAnalyzer(
        config=config, project_dir=project_dir, existing_subsets=existing_subsets
    )
    analyzer.run()
    return analyzer

# TODO: more
'''
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
'''
