"""Integration guide for comprehensive visualization suite.

This document shows how to integrate sample_maps.py, ml_metrics.py, and 
feature_plots.py into the ML pipeline for complete observability.

Usage Pattern:
1. After train/test split -> sample_maps.py (geographic + metadata viz)
2. During Optuna optimization -> ml_metrics.py (trial progress)
3. After model selection -> ml_metrics.py (confusion matrix, ROC)
4. After feature selection -> feature_plots.py (correlations, importance)
5. Final analysis -> feature_plots.py (feature vs target relationships)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all viz modules
from workflow_16s.downstream.visualization.sample_maps import (
    create_categorical_sample_map,
    create_numerical_sample_map,
    create_train_test_split_map,
    create_metadata_sample_maps,
    create_multi_column_overview,
)
from workflow_16s.downstream.visualization.ml_metrics import (
    plot_optuna_trial_progress,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_vs_target_categorical,
    plot_feature_vs_target_continuous,
    plot_feature_importance_bars,
    plot_model_metrics_summary,
)
from workflow_16s.downstream.visualization.feature_plots import (
    plot_feature_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_pairs_scatter,
    plot_cumulative_variance,
    plot_top_features_boxplot,
)

logger = logging.getLogger('workflow_16s')


class VisualizationPipeline:
    """Orchestrate all visualization components in sequence."""
    
    def __init__(self, output_dir: Path, create_png: bool = False):
        """
        Initialize visualization pipeline.
        
        Parameters
        ----------
        output_dir : Path
            Root output directory for all plots.
        create_png : bool
            Whether to export PNG versions (requires Kaleido).
        """
        self.output_dir = Path(output_dir)
        self.create_png = create_png
        self.plots_generated = []
        
        # Create subdirectories for each stage
        self.sample_viz_dir = self.output_dir / "01_sample_visualizations"
        self.optuna_viz_dir = self.output_dir / "02_optimization_results"
        self.model_viz_dir = self.output_dir / "03_model_performance"
        self.feature_viz_dir = self.output_dir / "04_feature_analysis"
        
        for d in [self.sample_viz_dir, self.optuna_viz_dir, self.model_viz_dir, self.feature_viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STAGE 1: Sample & Metadata Visualization
    # =========================================================================
    
    def visualize_samples(
        self,
        metadata: pd.DataFrame,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        """
        Generate all sample-level visualizations.
        
        Parameters
        ----------
        metadata : pd.DataFrame
            Sample metadata.
        lat_col : str, optional
            Latitude column name.
        lon_col : str, optional
            Longitude column name.
        train_indices : np.ndarray, optional
            Indices of training samples.
        test_indices : np.ndarray, optional
            Indices of test samples.
        
        Returns
        -------
        Dict[str, Path]
            Mapping of visualization names to output paths.
        """
        outputs = {}
        
        logger.info("🗺️  Generating sample visualizations...")
        
        try:
            # 1. Categorical metadata maps
            if lat_col and lon_col and lat_col in metadata.columns and lon_col in metadata.columns:
                for col in metadata.select_dtypes(include=['object', 'category']).columns:
                    try:
                        output_path = self.sample_viz_dir / f"sample_map_categorical_{col}.html"
                        create_categorical_sample_map(
                            metadata, lat_col, lon_col, col, output_path
                        )
                        outputs[f"categorical_{col}"] = output_path
                    except Exception as e:
                        logger.warning(f"Could not visualize {col}: {e}")
            
            # 2. Numerical metadata maps
            if lat_col and lon_col:
                for col in metadata.select_dtypes(include=[np.number]).columns:
                    try:
                        output_path = self.sample_viz_dir / f"sample_map_numerical_{col}.html"
                        create_numerical_sample_map(
                            metadata, lat_col, lon_col, col, 
                            colorscale='Viridis',
                            output_path=output_path
                        )
                        outputs[f"numerical_{col}"] = output_path
                    except Exception as e:
                        logger.warning(f"Could not visualize {col}: {e}")
            
            # 3. Train/test split map
            if train_indices is not None and test_indices is not None and lat_col and lon_col:
                try:
                    output_path = self.sample_viz_dir / "sample_map_train_test_split.html"
                    create_train_test_split_map(
                        metadata, train_indices, test_indices, lat_col, lon_col,
                        output_path=output_path
                    )
                    outputs['train_test_split'] = output_path
                    logger.info(f"✅ Train/test split map: {output_path}")
                except Exception as e:
                    logger.warning(f"Could not visualize train/test split: {e}")
            
            # 4. Multi-column overview
            try:
                cat_cols = metadata.select_dtypes(include=['object', 'category']).columns.tolist()[:8]
                if cat_cols:
                    output_path = self.sample_viz_dir / "sample_map_multi_column_overview.html"
                    create_multi_column_overview(
                        metadata, cat_cols, max_cols=4,
                        output_path=output_path
                    )
                    outputs['multi_column_overview'] = output_path
            except Exception as e:
                logger.warning(f"Could not create multi-column overview: {e}")
        
        except Exception as e:
            logger.error(f"Error in sample visualization: {e}")
        
        self.plots_generated.extend(list(outputs.values()))
        logger.info(f"✅ Generated {len(outputs)} sample visualizations")
        return outputs
    
    # =========================================================================
    # STAGE 2: Optuna Optimization Results
    # =========================================================================
    
    def visualize_optuna_optimization(
        self,
        trials_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        """
        Visualize Optuna trial progress.
        
        Parameters
        ----------
        trials_df : pd.DataFrame
            DataFrame from study.trials_dataframe().
        
        Returns
        -------
        Dict[str, Path]
            Mapping of visualization names to output paths.
        """
        outputs = {}
        
        logger.info("📊 Visualizing Optuna optimization...")
        
        try:
            output_path = self.optuna_viz_dir / "optuna_trial_progress.html"
            fig = plot_optuna_trial_progress(trials_df, output_path=output_path)
            if fig:
                outputs['trial_progress'] = output_path
                logger.info(f"✅ Optuna trial progress: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize trial progress: {e}")
        
        self.plots_generated.extend(list(outputs.values()))
        return outputs
    
    # =========================================================================
    # STAGE 3: Model Performance Metrics
    # =========================================================================
    
    def visualize_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Path]:
        """
        Visualize model performance metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        y_score : np.ndarray, optional
            Predicted scores/probabilities.
        class_names : list, optional
            Class label names.
        metrics : Dict[str, float], optional
            Dictionary of metric names and values.
        
        Returns
        -------
        Dict[str, Path]
            Mapping of visualization names to output paths.
        """
        outputs = {}
        
        logger.info("🎯 Visualizing model performance...")
        
        try:
            # Confusion matrix
            output_path = self.model_viz_dir / "confusion_matrix.html"
            fig = plot_confusion_matrix(y_true, y_pred, class_names, output_path=output_path)
            if fig:
                outputs['confusion_matrix'] = output_path
                logger.info(f"✅ Confusion matrix: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize confusion matrix: {e}")
        
        try:
            # ROC curve for binary classification
            if y_score is not None and len(np.unique(y_true)) == 2:
                output_path = self.model_viz_dir / "roc_curve.html"
                fig = plot_roc_curve(y_true, y_score, output_path=output_path)
                if fig:
                    outputs['roc_curve'] = output_path
                    logger.info(f"✅ ROC curve: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize ROC curve: {e}")
        
        try:
            # Metrics summary
            if metrics:
                output_path = self.model_viz_dir / "model_metrics_summary.html"
                fig = plot_model_metrics_summary(metrics, output_path=output_path)
                if fig:
                    outputs['metrics_summary'] = output_path
                    logger.info(f"✅ Metrics summary: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize metrics: {e}")
        
        self.plots_generated.extend(list(outputs.values()))
        return outputs
    
    # =========================================================================
    # STAGE 4: Feature Analysis
    # =========================================================================
    
    def visualize_features(
        self,
        X_features: pd.DataFrame,
        y_target: Optional[pd.Series] = None,
        feature_importance: Optional[pd.Series] = None,
        target_is_categorical: bool = True,
    ) -> Dict[str, Path]:
        """
        Visualize feature relationships and importance.
        
        Parameters
        ----------
        X_features : pd.DataFrame
            Feature matrix.
        y_target : pd.Series, optional
            Target variable for feature vs target plots.
        feature_importance : pd.Series, optional
            Feature importance scores.
        target_is_categorical : bool
            Whether target is categorical (affects plot types).
        
        Returns
        -------
        Dict[str, Path]
            Mapping of visualization names to output paths.
        """
        outputs = {}
        
        logger.info("🔬 Generating feature analysis visualizations...")
        
        try:
            # Feature correlation heatmap
            output_path = self.feature_viz_dir / "feature_correlation_heatmap.html"
            fig = plot_feature_correlation_heatmap(X_features, top_n_features=30, 
                                                   output_path=output_path, cluster=True)
            if fig:
                outputs['correlation_heatmap'] = output_path
                logger.info(f"✅ Feature correlation: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize correlations: {e}")
        
        try:
            # Feature distributions
            output_path = self.feature_viz_dir / "feature_distributions.html"
            fig = plot_feature_distributions(X_features, top_n_features=8,
                                             output_path=output_path)
            if fig:
                outputs['distributions'] = output_path
                logger.info(f"✅ Feature distributions: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize distributions: {e}")
        
        try:
            # Feature pairs scatter
            output_path = self.feature_viz_dir / "feature_pairs_scatter.html"
            fig = plot_feature_pairs_scatter(X_features, max_pairs=6,
                                             output_path=output_path, sample_size=1000)
            if fig:
                outputs['feature_pairs'] = output_path
                logger.info(f"✅ Feature pairs: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize feature pairs: {e}")
        
        try:
            # Cumulative variance
            output_path = self.feature_viz_dir / "cumulative_variance.html"
            fig = plot_cumulative_variance(X_features, max_features=50,
                                           output_path=output_path)
            if fig:
                outputs['cumulative_variance'] = output_path
                logger.info(f"✅ Cumulative variance: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize variance: {e}")
        
        try:
            # Feature importance
            if feature_importance is not None and len(feature_importance) > 0:
                output_path = self.feature_viz_dir / "feature_importance.html"
                fig = plot_feature_importance_bars(feature_importance, top_n=20,
                                                   output_path=output_path)
                if fig:
                    outputs['importance_bars'] = output_path
                    logger.info(f"✅ Feature importance: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize importance: {e}")
        
        try:
            # Feature vs target
            if y_target is not None:
                if target_is_categorical:
                    output_path = self.feature_viz_dir / "feature_vs_target_violin.html"
                    fig = plot_feature_vs_target_categorical(X_features, y_target,
                                                             max_features=6,
                                                             output_path=output_path)
                    if fig:
                        outputs['feature_vs_target_violin'] = output_path
                        logger.info(f"✅ Feature vs target (violin): {output_path}")
                else:
                    output_path = self.feature_viz_dir / "feature_vs_target_scatter.html"
                    fig = plot_feature_vs_target_continuous(X_features, y_target,
                                                            max_features=6,
                                                            output_path=output_path)
                    if fig:
                        outputs['feature_vs_target_scatter'] = output_path
                        logger.info(f"✅ Feature vs target (scatter): {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize feature vs target: {e}")
        
        try:
            # Top features boxplot
            output_path = self.feature_viz_dir / "top_features_boxplot.html"
            fig = plot_top_features_boxplot(X_features, top_n_features=10,
                                            output_path=output_path)
            if fig:
                outputs['feature_boxplot'] = output_path
                logger.info(f"✅ Feature boxplots: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize boxplots: {e}")
        
        # Clean up matplotlib figures to avoid "too many open files" error
        plt.close('all')
        
        self.plots_generated.extend(list(outputs.values()))
        logger.info(f"✅ Generated {len(outputs)} feature visualizations")
        return outputs
    
    # =========================================================================
    # Summary & Reporting
    # =========================================================================
    
    def generate_summary(self) -> str:
        """
        Generate summary of all visualizations generated.
        
        Returns
        -------
        str
            Summary report.
        """
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║         VISUALIZATION PIPELINE SUMMARY                         ║
╚════════════════════════════════════════════════════════════════╝

📊 Total plots generated: {len(self.plots_generated)}

📁 Output directories:
   • Sample visualizations: {self.sample_viz_dir.relative_to(self.output_dir.parent)}
   • Optimization results: {self.optuna_viz_dir.relative_to(self.output_dir.parent)}
   • Model performance:    {self.model_viz_dir.relative_to(self.output_dir.parent)}
   • Feature analysis:     {self.feature_viz_dir.relative_to(self.output_dir.parent)}

📈 All plots saved as interactive HTML + optional PNG
"""
        return summary


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Demonstrate full visualization pipeline integration."""
    
    # Create pipeline
    viz_pipeline = VisualizationPipeline(
        output_dir=Path("output/visualizations"),
        create_png=False  # Set to True if Kaleido is available
    )
    
    # Dummy data
    metadata = pd.DataFrame({
        'sample_id': [f'S{i}' for i in range(100)],
        'latitude': np.random.uniform(-90, 90, 100),
        'longitude': np.random.uniform(-180, 180, 100),
        'project': np.random.choice(['ProjectA', 'ProjectB', 'ProjectC'], 100),
        'elevation': np.random.uniform(0, 3000, 100),
    })
    
    X_features = pd.DataFrame(
        np.random.randn(100, 50),
        columns=[f'Feature_{i}' for i in range(50)]
    )
    
    y_target = pd.Series(np.random.choice([0, 1], 100), name='target')
    
    trials_df = pd.DataFrame({
        'number': range(50),
        'value': np.random.randn(50).cumsum() + 0.5,
        'state': ['COMPLETE'] * 50,
    })
    
    # Generate all visualizations
    viz_pipeline.visualize_samples(
        metadata,
        lat_col='latitude',
        lon_col='longitude',
        train_indices=np.arange(0, 70),
        test_indices=np.arange(70, 100),
    )
    
    viz_pipeline.visualize_optuna_optimization(trials_df)
    
    y_pred = np.random.choice([0, 1], 100)
    y_score = np.random.rand(100)
    metrics = {'accuracy': 0.85, 'f1': 0.82, 'auc': 0.90}
    
    viz_pipeline.visualize_model_performance(
        y_target.values, y_pred, y_score, metrics=metrics
    )
    
    feature_importance = pd.Series(
        np.random.exponential(1, 50),
        index=[f'Feature_{i}' for i in range(50)]
    )
    
    viz_pipeline.visualize_features(
        X_features, y_target, feature_importance,
        target_is_categorical=True
    )
    
    # Print summary
    print(viz_pipeline.generate_summary())


if __name__ == '__main__':
    example_usage()
