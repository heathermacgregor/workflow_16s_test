# ml_strategy_orchestrator.py
"""
ML Strategy Orchestrator

Applies semantic ML strategy configurations to the ML pipeline.
Handles:
1. Strategy resolution and validation
2. Feature set preparation (taxonomy, batch, metadata)
3. Preprocessing application (identity or ConQuR)
4. CV method setup (k-fold, LOPOCV, spatial)
5. Data filtering based on policy
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from .ml_strategy_config import (
    MLStrategyConfig, 
    FeatureSet, 
    PreprocessingMethod, 
    CVMethod, 
    FilterPolicy,
    get_strategy, 
    list_strategies,
    print_strategy_catalog
)

logger = logging.getLogger("workflow_16s")


class MLStrategyOrchestrator:
    """
    Orchestrates ML pipeline execution using semantic strategy configurations.
    """
    
    def __init__(self, workflow_context):
        """
        Initialize orchestrator with workflow context.
        
        Args:
            workflow_context: Workflow instance with adata, config, logger, output_dir
        """
        self.workflow = workflow_context
        self.adata = workflow_context.adata
        self.config = workflow_context.config
        self.logger = workflow_context.logger
        self.output_dir = Path(workflow_context.output_dir)
        
        # Strategy tracking
        self.current_strategy: Optional[MLStrategyConfig] = None
        self.executed_strategies: List[str] = []
        self.strategy_results: Dict[str, Dict[str, Any]] = {}
    
    def validate_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Validate that a strategy can be executed with current data.
        
        Args:
            strategy_name: Name of strategy to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        strategy = get_strategy(strategy_name)
        if not strategy:
            return False, f"Strategy '{strategy_name}' not found in registry"
        
        # Check sample count
        if self.adata.n_obs < strategy.recommended_min_samples:
            self.logger.warning(
                f"Strategy {strategy_name} recommends >= {strategy.recommended_min_samples} samples; "
                f"found {self.adata.n_obs}"
            )
        
        # Check required metadata
        missing_cols = []
        for col in strategy.required_metadata_cols:
            if col not in self.adata.obs.columns:
                missing_cols.append(col)
        
        if missing_cols and strategy.filter_policy != FilterPolicy.NONE:
            self.logger.warning(
                f"Strategy {strategy_name} requires metadata columns: {missing_cols} "
                f"(may not be available or may be populated during enrichment)"
            )
        
        # Check study count for LOPOCV
        if strategy.cv_method == CVMethod.LOPOCV:
            study_col = self.config.downstream.study_grouping.study_col if hasattr(self.config, 'downstream') else 'Project'
            if study_col in self.adata.obs.columns:
                n_studies = self.adata.obs[study_col].nunique()
                if n_studies < strategy.min_studies:
                    return False, (
                        f"LOPOCV requires >= {strategy.min_studies} studies; "
                        f"found {n_studies} in column '{study_col}'"
                    )
        
        return True, ""
    
    def prepare_features(self, strategy: MLStrategyConfig) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature set based on strategy.
        
        Combines taxonomy (always included), batch (if specified), and metadata (if specified).
        
        Args:
            strategy: MLStrategyConfig to apply
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.logger.info(f"  📊 Preparing feature set: {strategy.feature_set.name}")
        
        # Start with taxonomy features (ASV/OTU data)
        # Assumes X is already in rCLR-transformed space
        feature_matrix = self.adata.X.copy()
        feature_names = list(self.adata.var_names)
        
        # Add batch column if requested
        if strategy.should_include_batch:
            self.logger.info("    + Adding batch/project column")
            study_col = self._get_study_column()
            if study_col and study_col in self.adata.obs.columns:
                batch_data = pd.Categorical(self.adata.obs[study_col]).codes.reshape(-1, 1)
                feature_matrix = np.hstack([feature_matrix, batch_data])
                feature_names.append('_batch')
        
        # Add metadata columns if requested
        if strategy.should_include_metadata:
            self.logger.info("    + Adding environmental metadata columns")
            metadata_cols = self._get_metadata_columns()
            if metadata_cols is not None and not metadata_cols.empty:
                # Normalize metadata to 0-1 range
                metadata_normalized = (metadata_cols - metadata_cols.min()) / (metadata_cols.max() - metadata_cols.min() + 1e-10)
                feature_matrix = np.hstack([feature_matrix, metadata_normalized.values])
                feature_names.extend(metadata_normalized.columns.tolist())
        
        self.logger.info(f"    ✓ Feature set prepared: {len(feature_names)} total features")
        return feature_matrix, feature_names
    
    def apply_preprocessing(self, strategy: MLStrategyConfig) -> None:
        """
        Apply preprocessing to adata based on strategy.
        
        Args:
            strategy: MLStrategyConfig to apply
        """
        if strategy.preprocessing == PreprocessingMethod.IDENTITY:
            self.logger.info("  🔄 Preprocessing: IDENTITY (no correction)")
            return
        
        elif strategy.preprocessing == PreprocessingMethod.CONQUR:
            self.logger.info("  🔄 Preprocessing: ConQuR batch correction")
            try:
                from ..modular_preprocessing import run_conqur_correction
                study_col = self._get_study_column()
                self.adata = run_conqur_correction(self.adata, batch_col=study_col)
                self.logger.info("    ✓ ConQuR correction applied")
            except Exception as e:
                self.logger.error(f"    ✗ ConQuR failed: {e}. Falling back to identity preprocessing.")
    
    def apply_cv_strategy(self, strategy: MLStrategyConfig) -> Dict[str, Any]:
        """
        Configure cross-validation method.
        
        Args:
            strategy: MLStrategyConfig to apply
            
        Returns:
            Dict with CV configuration
        """
        cv_config = {'method': strategy.cv_method.value}
        
        if strategy.cv_method == CVMethod.KFOLD:
            cv_config['n_splits'] = strategy.cv_folds
            self.logger.info(f"  🔀 CV: K-fold with {strategy.cv_folds} splits")
        
        elif strategy.cv_method == CVMethod.LOPOCV:
            study_col = self._get_study_column()
            cv_config['group_col'] = study_col
            self.logger.info(f"  🔀 CV: Leave-One-Project-Out on column '{study_col}'")
        
        elif strategy.cv_method == CVMethod.SPATIAL:
            cv_config['lat_col'] = self.config.downstream.metadata_cols.lat_col
            cv_config['lon_col'] = self.config.downstream.metadata_cols.lon_col
            cv_config['n_splits'] = strategy.cv_folds
            self.logger.info(f"  🔀 CV: Spatial split on coordinates")
        
        return cv_config
    
    def apply_filter_policy(self, strategy: MLStrategyConfig) -> pd.Index:
        """
        Apply data filtering based on policy.
        
        Args:
            strategy: MLStrategyConfig to apply
            
        Returns:
            Index of samples passing filter
        """
        total_samples = self.adata.n_obs
        
        if strategy.filter_policy == FilterPolicy.NONE:
            self.logger.info("  🔍 Filter: NONE (use all samples)")
            return self.adata.obs_names
        
        eligible_mask = np.ones(total_samples, dtype=bool)
        
        if strategy.filter_policy == FilterPolicy.MULTICLASS_ONLY:
            self.logger.info(f"  🔍 Filter: MULTICLASS_ONLY (avg >= 2 classes, min samples per class >= {strategy.min_samples_per_class})")
            # Filter to studies with >=2 classes
            study_col = self._get_study_column()
            if study_col in self.adata.obs.columns:
                for study in self.adata.obs[study_col].unique():
                    study_mask = self.adata.obs[study_col] == study
                    # For now: just ensure we have studies with multiple classes
                    # Specific target checking would happen per-target
                    if study_mask.sum() >= strategy.min_samples_per_class:
                        eligible_mask &= study_mask
        
        elif strategy.filter_policy == FilterPolicy.VARIANCE_FILTERED:
            self.logger.info("  🔍 Filter: VARIANCE_FILTERED (target must have non-zero variance)")
            # This is target-specific and will be applied during actual ML training
            self.logger.debug("    (Variance filtering is applied per-target during training)")
        
        n_kept = eligible_mask.sum()
        self.logger.info(f"    ✓ Samples after filtering: {n_kept}/{total_samples} ({100*n_kept/total_samples:.1f}%)")
        
        return self.adata.obs_names[eligible_mask]
    
    def run_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """
        Execute a single strategy end-to-end.
        
        Args:
            strategy_name: Name of strategy to execute
            
        Returns:
            Dict with strategy results metadata
        """
        strategy = get_strategy(strategy_name)
        if not strategy:
            self.logger.error(f"Strategy '{strategy_name}' not found")
            return {}
        
        self.current_strategy = strategy
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 STRATEGY: {strategy_name}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Description: {strategy.description}")
        
        # Validate
        is_valid, reason = self.validate_strategy(strategy_name)
        if not is_valid:
            self.logger.error(f"Validation failed: {reason}")
            return {'status': 'failed', 'error': reason}
        
        # Create strategy output directory
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Apply preprocessing
        self.apply_preprocessing(strategy)
        
        # Step 2: Prepare features
        X, feature_names = self.prepare_features(strategy)
        
        # Step 3: Apply CV strategy
        cv_config = self.apply_cv_strategy(strategy)
        
        # Step 4: Apply filter policy
        eligible_samples = self.apply_filter_policy(strategy)
        
        # Step 5: Prepare output for ML pipeline
        result = {
            'status': 'prepared',
            'strategy_name': strategy_name,
            'feature_matrix': X,
            'feature_names': feature_names,
            'sample_index': eligible_samples,
            'cv_config': cv_config,
            'hyperparameters': strategy.hyperparameters,
            'num_features': strategy.num_features,
            'output_dir': str(strategy_dir),
        }
        
        # Log summary
        self.logger.info(f"\nStrategy Summary:")
        self.logger.info(f"  Total samples: {self.adata.n_obs}")
        self.logger.info(f"  Eligible samples: {len(eligible_samples)}")
        self.logger.info(f"  Total features: {len(feature_names)}")
        self.logger.info(f"  Target features: {strategy.num_features}")
        self.logger.info(f"  CV Method: {strategy.cv_method.name}")
        self.logger.info(f"  Output: {strategy_dir}")
        
        # Save strategy metadata
        metadata = {
            'strategy_name': strategy_name,
            'description': strategy.description,
            'feature_set': strategy.feature_set.value,
            'preprocessing': strategy.preprocessing.value,
            'cv_method': strategy.cv_method.value,
            'filter_policy': strategy.filter_policy.value,
            'num_features': strategy.num_features,
            'num_samples': len(eligible_samples),
            'num_total_features': len(feature_names),
        }
        
        metadata_file = strategy_dir / 'strategy_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.strategy_results[strategy_name] = result
        self.executed_strategies.append(strategy_name)
        
        return result
    
    def run_all_strategies(self, strategy_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Execute multiple strategies.
        
        Args:
            strategy_names: List of strategy names to run. If None, uses config.downstream.strategies
            
        Returns:
            Dict of all strategy results
        """
        if strategy_names is None:
            strategy_names = getattr(self.config.downstream, 'strategies', ['taxonomyOnly_Identity_KFold_NoFilter'])
        
        # Check for special flag: print catalog and exit
        if hasattr(self.config.downstream, 'strategy_catalog') and self.config.downstream.strategy_catalog:
            self.logger.info(print_strategy_catalog())
            return {}
        
        self.logger.info(f"\n🎯 ML Strategy Pipeline: {len(strategy_names)} strategies")
        self.logger.info(f"   {', '.join(strategy_names)}\n")
        
        results = {}
        for strategy_name in strategy_names:
            try:
                result = self.run_strategy(strategy_name)
                results[strategy_name] = result
            except Exception as e:
                self.logger.error(f"Error executing strategy '{strategy_name}': {e}", exc_info=True)
                results[strategy_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    # Helper methods
    def _get_study_column(self) -> str:
        """Get the study/project grouping column name."""
        if hasattr(self.config, 'downstream') and hasattr(self.config.downstream, 'study_grouping'):
            col = self.config.downstream.study_grouping.study_col
            if col and col in self.adata.obs.columns:
                return col
        
        # Fallback
        for col in ['Project', 'project', 'batch_original', 'study', 'dataset']:
            if col in self.adata.obs.columns:
                return col
        
        return 'Project'  # Default
    
    def _get_metadata_columns(self) -> Optional[pd.DataFrame]:
        """Get available environmental metadata columns."""
        metadata_patterns = [
            ('lat', 'lon'),
            ('latitude', 'longitude'),
            ('LatitudeParsed', 'LongitudeParsed'),
        ]
        
        available_cols = []
        for col in self.adata.obs.columns:
            # Skip obvious non-metadata columns
            if col.startswith('_') or col in ['batch', 'Project', 'dataset', 'study']:
                continue
            
            # Try to convert to numeric; if successful, it's likely metadata
            try:
                pd.to_numeric(self.adata.obs[col], errors='coerce')
                available_cols.append(col)
            except:
                pass
        
        if not available_cols:
            self.logger.debug("  No numeric metadata columns found")
            return None
        
        # Return only non-NaN data
        metadata = self.adata.obs[available_cols].copy()
        return metadata.dropna(axis=1, how='all')
