"""Phase 8: Visualization Suite - Integration Implementation

This module shows exactly where and how to integrate the visualization 
pipeline into feature_selection.py and analysis.py.

INTEGRATION CHECKLIST:
  ✓ Step 1: Update config schema with visualization parameters
  ✓ Step 2: Add imports to feature_selection.py
  ✓ Step 3: Call VisualizationPipeline after train/test split
  ✓ Step 4: Call visualization stages after each ML step
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger('workflow_16s')


# ============================================================================
# STEP 1: CONFIG SCHEMA INTEGRATION
# ============================================================================

"""
Add to config_schema.py:

@dataclass
class VisualizationConfig:
    '''Configuration for comprehensive visualization pipeline.'''
    enabled: bool = False
    create_png: bool = False  # Requires Kaleido
    output_dir: Optional[str] = None  # Auto-default to {project}/visualizations
    
    # Visualization stages to enable
    sample_visualizations: bool = True
    optuna_visualizations: bool = True
    model_performance_visualizations: bool = True
    feature_analysis_visualizations: bool = True
    
    # Feature plot parameters
    max_top_features: int = 20  # Feature importance top N
    max_corr_features: int = 30  # Correlation heatmap top N
    cluster_correlations: bool = True
    sample_size_scatter: Optional[int] = 1000  # Downsample scatter plots
    
    # Geographic visualizations
    include_geographic_maps: bool = True
    lat_column: str = 'latitude'
    lon_column: str = 'longitude'

# Then add to MLConfig dataclass:
visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
"""


# ============================================================================
# STEP 2: INTEGRATION INTO feature_selection.py
# ============================================================================

"""
Location: workflow_16s/src/workflow_16s/downstream/feature_selection.py

Add these imports at the top:

from workflow_16s.downstream.visualization.integration_guide import VisualizationPipeline
from workflow_16s.downstream.visualization.feature_plots import (
    plot_feature_correlation_heatmap,
    plot_feature_importance_bars,
)

Then in the run_feature_selection() function, after train/test split is created:

def run_feature_selection(
    adata: 'AnnData',
    config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    '''Run ML pipeline with integrated visualization.'''
    
    logger.info("🚀 Starting ML feature selection...")
    
    # [EXISTING CODE: Data validation, eligibility filtering, etc.]
    
    # ─────────────────────────────────────────────────────────────────────
    # VISUALIZATION: Initialize pipeline (AFTER subsetting, BEFORE split)
    # ─────────────────────────────────────────────────────────────────────
    
    viz_config = config.get('ml', {}).get('visualization', {})
    
    if viz_config.get('enabled', False):
        
        # Determine output directory
        viz_output_dir = Path(viz_config.get('output_dir') or 
                              config['paths']['project'] / 'visualizations')
        
        viz_pipeline = VisualizationPipeline(
            output_dir=viz_output_dir,
            create_png=viz_config.get('create_png', False)
        )
        
        logger.info(f"📊 Visualization output: {viz_output_dir}")
    else:
        viz_pipeline = None
        logger.info("⏭️  Visualizations disabled")
    
    # [EXISTING CODE: Feature selection loop over strategies]
    
    for strategy_name in strategies:
        
        logger.info(f"🔧 Processing strategy: {strategy_name}")
        
        # [EXISTING CODE: Feature composition, train/test split]
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # ─────────────────────────────────────────────────────────────────
        # VISUALIZATION: Sample geography & metadata (AFTER SPLIT)
        # ─────────────────────────────────────────────────────────────────
        
        if viz_pipeline and viz_config.get('sample_visualizations', True):
            try:
                logger.info("🗺️  Generating sample visualizations...")
                
                lat_col = viz_config.get('lat_column', 'latitude')
                lon_col = viz_config.get('lon_column', 'longitude')
                
                if lat_col in adata.obs.columns and lon_col in adata.obs.columns:
                    viz_pipeline.visualize_samples(
                        metadata=adata.obs.copy(),
                        lat_col=lat_col,
                        lon_col=lon_col,
                        train_indices=train_indices,
                        test_indices=test_indices,
                    )
                else:
                    logger.warning(f"Geographic columns not found: {lat_col}, {lon_col}")
            
            except Exception as e:
                logger.warning(f"⚠️  Sample visualization failed: {e}")
        
        # [EXISTING CODE: Model training with Optuna]
        
        # ─────────────────────────────────────────────────────────────────
        # VISUALIZATION: Optuna optimization progress (AFTER STUDY COMPLETE)
        # ─────────────────────────────────────────────────────────────────
        
        if viz_pipeline and viz_config.get('optuna_visualizations', True):
            try:
                logger.info("📈 Visualizing Optuna optimization...")
                
                # Assuming study is from Optuna
                trials_df = study.trials_dataframe()
                
                if not trials_df.empty:
                    viz_pipeline.visualize_optuna_optimization(trials_df)
                else:
                    logger.warning("No completed trials to visualize")
            
            except Exception as e:
                logger.warning(f"⚠️  Optuna visualization failed: {e}")
        
        # [EXISTING CODE: Model training & predictions]
        
        # Get y_train, y_test, y_pred, y_proba from training results
        
        # ─────────────────────────────────────────────────────────────────
        # VISUALIZATION: Model performance (AFTER TRAINING & PREDICTION)
        # ─────────────────────────────────────────────────────────────────
        
        if viz_pipeline and viz_config.get('model_performance_visualizations', True):
            try:
                logger.info("🎯 Visualizing model performance...")
                
                # Get metrics dict from training results
                metrics = {
                    'accuracy': results.get('accuracy', 0),
                    'f1_score': results.get('f1', 0),
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0),
                    'auc': results.get('auc', 0),
                }
                
                # Determine if classification or regression
                is_categorical = len(np.unique(y_test)) < 10
                
                if is_categorical:
                    viz_pipeline.visualize_model_performance(
                        y_true=y_test,
                        y_pred=y_pred,
                        y_score=y_proba if y_proba is not None else None,
                        class_names=None,  # Auto-label
                        metrics=metrics,
                    )
                else:
                    logger.info("Regression target detected - skipping classification-specific plots")
            
            except Exception as e:
                logger.warning(f"⚠️  Model performance visualization failed: {e}")
        
        # [EXISTING CODE: Feature selection results]
        
        # Get X_test, feature_importance from results
        
        # ─────────────────────────────────────────────────────────────────
        # VISUALIZATION: Feature analysis (AFTER FEATURE SELECTION)
        # ─────────────────────────────────────────────────────────────────
        
        if viz_pipeline and viz_config.get('feature_analysis_visualizations', True):
            try:
                logger.info("🔬 Visualizing feature analysis...")
                
                # Create feature importance Series
                feature_importance = pd.Series(
                    results.get('feature_importances', []),
                    index=X_test.columns
                ).sort_values(ascending=False)
                
                viz_pipeline.visualize_features(
                    X_features=X_test,
                    y_target=y_test,
                    feature_importance=feature_importance if not feature_importance.empty else None,
                    target_is_categorical=is_categorical,
                )
            
            except Exception as e:
                logger.warning(f"⚠️  Feature visualization failed: {e}")
        
        # [EXISTING CODE: Store results]
        
        results[f'{strategy_name}_viz_outputs'] = viz_pipeline.plots_generated if viz_pipeline else []
    
    # Final summary
    if viz_pipeline:
        logger.info(viz_pipeline.generate_summary())
    
    return results
"""


# ============================================================================
# STEP 3: CONFIGURATION YAML UPDATE
# ============================================================================

"""
In microbeatlas_test_config.yaml, update the ml.plots section:

ml:
  plots:
    enabled: True
    interpretability:
      enabled: True
      shap_plots: True
      feature_importance_plots: True
      sample_maps: True
      confusion_matrices: True
      roc_curves: True
    
    # Comprehensive visualization pipeline (NEW)
    visualization:
      enabled: True
      create_png: False  # Set to True if Kaleido installed
      output_dir: null   # Auto: {project}/visualizations
      
      # What to visualize
      sample_visualizations: True
      optuna_visualizations: True
      model_performance_visualizations: True
      feature_analysis_visualizations: True
      
      # Feature plot parameters
      max_top_features: 20
      max_corr_features: 30
      cluster_correlations: True
      sample_size_scatter: 1000
      
      # Geographic
      include_geographic_maps: True
      lat_column: 'lat'
      lon_column: 'lon'
"""


# ============================================================================
# STEP 4: CODE SNIPPET - Quick Integration
# ============================================================================

def setup_visualization_pipeline(config: Dict[str, Any]) -> Optional['VisualizationPipeline']:
    """
    Factory function for creating visualization pipeline from config.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    
    Returns
    -------
    VisualizationPipeline or None
        Initialized pipeline if enabled, else None.
    """
    viz_config = config.get('ml', {}).get('visualization', {})
    
    if not viz_config.get('enabled', False):
        logger.info("⏭️  Visualizations disabled in config")
        return None
    
    # Determine output directory
    output_dir = viz_config.get('output_dir')
    if output_dir is None:
        output_dir = Path(config['paths']['project']) / 'visualizations'
    else:
        output_dir = Path(output_dir)
    
    # Create pipeline
    from workflow_16s.downstream.visualization.integration_guide import VisualizationPipeline
    
    pipeline = VisualizationPipeline(
        output_dir=output_dir,
        create_png=viz_config.get('create_png', False)
    )
    
    logger.info(f"📊 Visualization pipeline initialized: {output_dir}")
    
    return pipeline


def visualize_ml_results(
    viz_pipeline: Optional['VisualizationPipeline'],
    adata: 'AnnData',
    config: Dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    feature_importance: Optional[pd.Series] = None,
    metrics: Optional[Dict[str, float]] = None,
    trials_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Run all visualization stages.
    
    Parameters
    ----------
    viz_pipeline : VisualizationPipeline or None
        Initialized pipeline (or None to skip).
    adata : AnnData
        Annotated data object.
    config : Dict[str, Any]
        Configuration.
    train_indices : np.ndarray
        Training sample indices.
    test_indices : np.ndarray
        Test sample indices.
    y_train, y_test : np.ndarray
        Training/test labels.
    y_pred : np.ndarray
        Model predictions.
    y_proba : np.ndarray, optional
        Model probabilities/scores.
    X_train, X_test : pd.DataFrame, optional
        Feature matrices.
    feature_importance : pd.Series, optional
        Feature importance scores.
    metrics : Dict[str, float], optional
        Model metrics (accuracy, F1, etc.).
    trials_df : pd.DataFrame, optional
        Optuna trials dataframe.
    """
    if viz_pipeline is None:
        return
    
    viz_config = config.get('ml', {}).get('visualization', {})
    
    # Stage 1: Sample visualizations
    if viz_config.get('sample_visualizations', True):
        try:
            metadata = adata.obs.copy()
            lat_col = viz_config.get('lat_column', 'latitude')
            lon_col = viz_config.get('lon_column', 'longitude')
            
            if lat_col in metadata.columns and lon_col in metadata.columns:
                viz_pipeline.visualize_samples(
                    metadata=metadata,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
                logger.info("✅ Sample visualizations complete")
            else:
                logger.warning(f"Geographic columns not found: {lat_col}, {lon_col}")
        except Exception as e:
            logger.warning(f"⚠️  Sample visualization failed: {e}")
    
    # Stage 2: Optuna optimization
    if viz_config.get('optuna_visualizations', True) and trials_df is not None:
        try:
            viz_pipeline.visualize_optuna_optimization(trials_df)
            logger.info("✅ Optuna visualization complete")
        except Exception as e:
            logger.warning(f"⚠️  Optuna visualization failed: {e}")
    
    # Stage 3: Model performance
    if viz_config.get('model_performance_visualizations', True):
        try:
            is_categorical = len(np.unique(y_test)) < 10
            
            if is_categorical:
                viz_pipeline.visualize_model_performance(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_score=y_proba,
                    metrics=metrics or {}
                )
                logger.info("✅ Model performance visualization complete")
            else:
                logger.info("Regression target - skipping classification plots")
        except Exception as e:
            logger.warning(f"⚠️  Model performance visualization failed: {e}")
    
    # Stage 4: Feature analysis
    if viz_config.get('feature_analysis_visualizations', True) and X_test is not None:
        try:
            is_categorical = len(np.unique(y_test)) < 10
            
            viz_pipeline.visualize_features(
                X_features=X_test,
                y_target=pd.Series(y_test),
                feature_importance=feature_importance,
                target_is_categorical=is_categorical,
            )
            logger.info("✅ Feature analysis visualization complete")
        except Exception as e:
            logger.warning(f"⚠️  Feature visualization failed: {e}")
    
    # Print summary
    logger.info(viz_pipeline.generate_summary())


# ============================================================================
# STEP 5: UPDATE YAML CONFIGURATION
# ============================================================================

yaml_update = """
# Add this section to ml.plots in microbeatlas_test_config.yaml:

ml:
  plots:
    enabled: True
    interpretability:
      enabled: True
      shap_plots: True
      feature_importance_plots: True
      sample_maps: True
      confusion_matrices: True
      roc_curves: True
    
    # NEW: Comprehensive visualization pipeline (Phase 8)
    visualization:
      enabled: True                      # Master toggle for all visualizations
      create_png: False                  # PNG export (requires Kaleido)
      output_dir: null                   # Auto-default to {project}/visualizations
      
      # Visualization stages
      sample_visualizations: True        # Geographic & metadata maps
      optuna_visualizations: True        # Optimization trial progress
      model_performance_visualizations: True  # Confusion matrix, ROC, metrics
      feature_analysis_visualizations: True   # Correlations, importance, distributions
      
      # Feature plot configuration
      max_top_features: 20               # Feature importance top N
      max_corr_features: 30              # Correlation matrix size
      cluster_correlations: True         # Apply hierarchical clustering
      sample_size_scatter: 1000          # Downsample large scatter plots
      
      # Geographic visualization
      include_geographic_maps: True      # Create train/test split maps
      lat_column: 'lat'                  # Latitude column name
      lon_column: 'lon'                  # Longitude column name
"""
