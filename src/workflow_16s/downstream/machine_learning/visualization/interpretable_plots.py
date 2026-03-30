"""
Interpretable ML Visualization Module

Creates publication-ready plots for model interpretability:
- SHAP Summary Plots (feature contribution overview)
- Feature Importance Plots (permutation and tree-based)
- Sample Maps (geographic distribution of train/test sets)
- Confusion Matrices & ROC Curves (model evaluation)

All plots use Plotly for interactivity and publication-ready styling.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import shap
import matplotlib.pyplot as plt
import io

logger = logging.getLogger("workflow_16s")


class InterpretablePlots:
    """
    Generate interpretable ML visualization plots.
    """
    
    def __init__(self, output_dir: Path, logger: logging.Logger = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("workflow_16s")
    
    # ========================================================================
    # SHAP Summary Plots
    # ========================================================================
    
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        X_data: pd.DataFrame,
        title: str = "SHAP Summary Plot",
        plot_type: str = "bar",
        output_name: str = "shap_summary"
    ) -> Path:
        """
        Create SHAP summary plot (bar or beeswarm style).
        
        Args:
            shap_values: SHAP values array (samples × features)
            feature_names: List of feature names
            X_data: Feature data (used for beeswarm plots)
            title: Plot title
            plot_type: "bar" (mean |SHAP|) or "scatter" (individual SHAP values)
            output_name: Name for output file
            
        Returns:
            Path to saved plot
        """
        try:
            if plot_type == "bar":
                return self._shap_bar_plot(
                    shap_values, feature_names, title, output_name
                )
            elif plot_type == "scatter":
                return self._shap_scatter_plot(
                    shap_values, feature_names, X_data, title, output_name
                )
        except Exception as e:
            self.logger.error(f"Error creating SHAP summary: {e}")
        
        return None
    
    def _shap_bar_plot(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        title: str,
        output_name: str
    ) -> Path:
        """Create bar plot of mean |SHAP| values."""
        # Calculate mean absolute SHAP values
        if shap_values.ndim == 3:
            # Multi-class: (samples, features, classes)
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # Binary/regression: (samples, features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort by importance
        idx = np.argsort(mean_abs_shap)[::-1][:20]  # Top 20
        
        fig = go.Figure(
            data=[go.Bar(
                x=mean_abs_shap[idx],
                y=[feature_names[i] for i in idx],
                orientation='h',
                marker=dict(
                    color=mean_abs_shap[idx],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Mean |SHAP|")
                )
            )]
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Mean |SHAP| Value",
            yaxis_title="Feature",
            height=600,
            width=1000,
            template="plotly_white",
            font=dict(size=12),
            showlegend=False
        )
        
        output_path = self.output_dir / f"{output_name}.html"
        fig.write_html(str(output_path))
        self.logger.info(f"Saved SHAP bar plot: {output_path}")
        
        return output_path
    
    def _shap_scatter_plot(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        X_data: pd.DataFrame,
        title: str,
        output_name: str
    ) -> Path:
        """Create scatter plot of SHAP values vs feature values."""
        # Get top features by importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][0]  # Top feature
        
        # Create scatter plot
        shap_vals = shap_values[:, top_idx]
        feature_vals = X_data.iloc[:, top_idx].values
        
        fig = go.Figure(
            data=[go.Scatter(
                x=feature_vals,
                y=shap_vals,
                mode='markers',
                marker=dict(
                    size=8,
                    color=shap_vals,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="SHAP Value"),
                    line=dict(width=1, color='white')
                ),
                text=[f"SHAP: {sv:.3f}" for sv in shap_vals],
                hoverinfo='text'
            )]
        )
        
        fig.update_layout(
            title=f"{title}<br><sub>Top Feature: {feature_names[top_idx]}</sub>",
            xaxis_title=f"{feature_names[top_idx]} (Feature Value)",
            yaxis_title="SHAP Value",
            height=500,
            width=800,
            template="plotly_white"
        )
        
        output_path = self.output_dir / f"{output_name}.html"
        fig.write_html(str(output_path))
        self.logger.info(f"Saved SHAP scatter plot: {output_path}")
        
        return output_path
    
    # ========================================================================
    # Feature Importance Plots
    # ========================================================================
    
    def plot_feature_importance(
        self,
        importances: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 20,
        output_name: str = "feature_importance"
    ) -> Path:
        """
        Create feature importance bar plot.
        
        Args:
            importances: Dict of {feature_name: importance_score}
            title: Plot title
            top_n: Number of top features to show
            output_name: Name for output file
            
        Returns:
            Path to saved plot
        """
        try:
            # Sort and get top N
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            names, scores = zip(*top_features)
            
            fig = go.Figure(
                data=[go.Bar(
                    x=list(scores),
                    y=list(names),
                    orientation='h',
                    marker=dict(
                        color=list(scores),
                        colorscale='Greens',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    )
                )]
            )
            
            fig.update_layout(
                title=f"{title} (Top {top_n} Features)",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=600,
                width=1000,
                template="plotly_white",
                font=dict(size=12),
                showlegend=False
            )
            
            output_path = self.output_dir / f"{output_name}.html"
            fig.write_html(str(output_path))
            self.logger.info(f"Saved feature importance plot: {output_path}")
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {e}")
            return None
    
    # ========================================================================
    # Sample Maps (Train/Test Geographic Distribution)
    # ========================================================================
    
    def plot_sample_map(
        self,
        metadata: pd.DataFrame,
        train_samples: List[str],
        test_samples: List[str],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        sample_col: str = "SampleID",
        title: str = "Train/Test Set Distribution",
        output_name: str = "sample_map"
    ) -> Path:
        """
        Create geographic scatter map showing train vs test samples.
        
        Args:
            metadata: DataFrame with sample coordinates
            train_samples: List of training sample IDs
            test_samples: List of test sample IDs
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            sample_col: Name of sample ID column
            title: Plot title
            output_name: Name for output file
            
        Returns:
            Path to saved plot
        """
        try:
            # Filter valid coordinates
            valid = metadata.dropna(subset=[lat_col, lon_col]).copy()
            
            # Add train/test label
            valid['Set'] = valid[sample_col].apply(
                lambda x: 'Training' if x in train_samples else (
                    'Testing' if x in test_samples else 'Other'
                )
            )
            
            # Filter to train/test only
            valid = valid[valid['Set'].isin(['Training', 'Testing'])]
            
            if valid.empty:
                self.logger.warning("No valid coordinates for sample map")
                return None
            
            # Create map
            fig = px.scatter_geo(
                valid,
                lat=lat_col,
                lon=lon_col,
                color='Set',
                hover_name=sample_col,
                hover_data={lat_col: ':.2f', lon_col: ':.2f'},
                title=title,
                color_discrete_map={
                    'Training': '#1f77b4',
                    'Testing': '#ff7f0e'
                },
                projection='natural earth',
                scope='world',
                size_max=10
            )
            
            fig.update_layout(
                height=600,
                width=1200,
                font=dict(size=12),
                title_x=0.5
            )
            
            output_path = self.output_dir / f"{output_name}.html"
            fig.write_html(str(output_path))
            self.logger.info(f"Saved sample map: {output_path}")
            
            # Also create a simple scatter plot for non-geographic view
            fig_scatter = go.Figure()
            
            for set_name, color in [('Training', '#1f77b4'), ('Testing', '#ff7f0e')]:
                subset = valid[valid['Set'] == set_name]
                fig_scatter.add_trace(go.Scatter(
                    x=subset[lon_col],
                    y=subset[lat_col],
                    mode='markers',
                    name=set_name,
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    text=subset[sample_col],
                    hoverinfo='text'
                ))
            
            fig_scatter.update_layout(
                title=f"{title} (Scatter View)",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=600,
                width=1000,
                template="plotly_white",
                font=dict(size=12)
            )
            
            output_path_scatter = self.output_dir / f"{output_name}_scatter.html"
            fig_scatter.write_html(str(output_path_scatter))
            self.logger.info(f"Saved sample scatter plot: {output_path_scatter}")
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating sample map: {e}")
            return None
    
    # ========================================================================
    # Confusion Matrix Plot
    # ========================================================================
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        output_name: str = "confusion_matrix"
    ) -> Path:
        """
        Create confusionmatrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
            output_name: Name for output file
            
        Returns:
            Path to saved plot
        """
        try:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            if labels is None:
                labels = [str(i) for i in range(len(cm))]
            
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title="Count")
                )
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=600,
                width=700,
                template="plotly_white",
                font=dict(size=12)
            )
            
            output_path = self.output_dir / f"{output_name}.html"
            fig.write_html(str(output_path))
            self.logger.info(f"Saved confusion matrix plot: {output_path}")
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix: {e}")
            return None
    
    # ========================================================================
    # ROC Curve Plot
    # ========================================================================
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
        output_name: str = "roc_curve"
    ) -> Path:
        """
        Create ROC curve plot.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities (class 1)
            title: Plot title
            output_name: Name for output file
            
        Returns:
            Path to saved plot
        """
        try:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC={roc_auc:.3f})',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Diagonal (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=600,
                width=700,
                template="plotly_white",
                font=dict(size=12),
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            output_path = self.output_dir / f"{output_name}.html"
            fig.write_html(str(output_path))
            self.logger.info(f"Saved ROC curve: {output_path}")
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error creating ROC curve: {e}")
            return None


def create_interpretable_plots(
    output_dir: Path,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    X_data: Optional[pd.DataFrame] = None,
    importances: Optional[Dict[str, float]] = None,
    metadata: Optional[pd.DataFrame] = None,
    train_samples: Optional[List[str]] = None,
    test_samples: Optional[List[str]] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_pred_proba: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Path]:
    """
    Convenience function to create all requested interpretable plots.
    
    Returns:
        Dict of {plot_name: output_path}
    """
    plots = InterpretablePlots(output_dir, logger=logger)
    results = {}
    
    # SHAP plots
    if shap_values is not None and feature_names is not None:
        try:
            results['shap_bar'] = plots.plot_shap_summary(
                shap_values, feature_names, X_data,
                plot_type='bar', output_name='shap_summary_bar'
            )
        except Exception as e:
            logger.warning(f"Could not create SHAP bar plot: {e}")
    
    # Feature importance
    if importances is not None:
        try:
            results['importance'] = plots.plot_feature_importance(
                importances, output_name='top_features'
            )
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")
    
    # Sample maps
    if metadata is not None and train_samples is not None and test_samples is not None:
        try:
            results['sample_map'] = plots.plot_sample_map(
                metadata, train_samples, test_samples,
                output_name='sample_distribution'
            )
        except Exception as e:
            logger.warning(f"Could not create sample map: {e}")
    
    # Confusion matrix
    if y_true is not None and y_pred is not None:
        try:
            results['confusion_matrix'] = plots.plot_confusion_matrix(
                y_true, y_pred, output_name='confusion_matrix'
            )
        except Exception as e:
            logger.warning(f"Could not create confusion matrix: {e}")
    
    # ROC curve
    if y_true is not None and y_pred_proba is not None:
        try:
            results['roc_curve'] = plots.plot_roc_curve(
                y_true, y_pred_proba, output_name='roc_curve'
            )
        except Exception as e:
            logger.warning(f"Could not create ROC curve: {e}")
    
    return results
