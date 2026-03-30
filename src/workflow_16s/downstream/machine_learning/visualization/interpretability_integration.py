"""
Integration module for interpretable ML visualizations.
Called after model training to generate SHAP plots, feature importance, and sample maps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from workflow_16s.downstream.machine_learning.visualization.interpretable_plots import (
    InterpretablePlots,
    create_interpretable_plots
)

logger = logging.getLogger("workflow_16s")


def generate_model_interpretability_plots(
    model_dir: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metadata: pd.DataFrame,
    model,
    task_type: str = "Classification",
    target_name: str = "Target",
    train_samples: Optional[List[str]] = None,
    test_samples: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Generate interpretable ML visualization plots for a trained model.
    
    Args:
        model_dir: Output directory for plots
        X_train: Training feature data
        X_test: Test feature data
        y_train: Training target
        y_test: Test target
        metadata: Sample metadata (with lat/lon for maps)
        model: Trained CatBoost/sklearn model
        task_type: "Classification" or "Regression"
        target_name: Name of target variable
        train_samples: List of training sample IDs
        test_samples: List of test sample IDs
        config: ML configuration dict (with plots.interpretability settings)
        
    Returns:
        Dict of {plot_type: output_path}
    """
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse config
    if config is None:
        config = {}
    plots_cfg = config.get('plots', {})
    interp_cfg = plots_cfg.get('interpretability', {})
    
    enable_shap = interp_cfg.get('shap_plots', True)
    enable_importance = interp_cfg.get('feature_importance_plots', True)
    enable_maps = interp_cfg.get('sample_maps', True)
    enable_confusion = interp_cfg.get('confusion_matrices', True)
    enable_roc = interp_cfg.get('roc_curves', True)
    
    results = {}
    
    # 1. Create predictions
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        y_pred = model.predict(X_test)
        
        logger.info(f"Generated predictions for {len(X_test)} test samples")
    except Exception as e:
        logger.warning(f"Could not generate predictions: {e}")
        y_pred_proba = None
        y_pred = None
    
    # 2. Get feature importances
    try:
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(X_train.columns, model.feature_importances_))
        else:
            importances = None
            
        if importances:
            logger.info(f"Extracted feature importances for {len(importances)} features")
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        importances = None
    
    # 3. Get SHAP values (if possible)
    try:
        if enable_shap:
            import shap
            
            if hasattr(model, 'get_feature_importance'):
                # CatBoost SHAP
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values).mean(axis=0)
                
                logger.info(f"Computed SHAP values: shape {shap_values.shape}")
            else:
                shap_values = None
                
    except Exception as e:
        logger.debug(f"Could not compute SHAP values: {e}")
        shap_values = None
    
    # 4. Prepare sample lists
    if train_samples is None:
        train_samples = X_train.index.tolist()
    if test_samples is None:
        test_samples = X_test.index.tolist()
    
    # 5. Generate plots
    plot_results = create_interpretable_plots(
        output_dir=model_dir,
        shap_values=shap_values if enable_shap else None,
        feature_names=list(X_train.columns) if enable_shap else None,
        X_data=X_test if enable_shap else None,
        importances=importances if enable_importance else None,
        metadata=metadata if enable_maps else None,
        train_samples=train_samples if enable_maps else None,
        test_samples=test_samples if enable_maps else None,
        y_true=y_test.values if enable_confusion or enable_roc else None,
        y_pred=y_pred if enable_confusion else None,
        y_pred_proba=y_pred_proba if enable_roc else None,
        logger=logger
    )
    
    results.update(plot_results)
    
    # 6. Summary report
    summary_report = f"""
    ========================================
    ML MODEL INTERPRETABILITY PLOTS SUMMARY
    ========================================
    
    Target: {target_name}
    Task Type: {task_type}
    
    Model Performance:
    - Training samples: {len(X_train)}
    - Test samples: {len(X_test)}
    - Features: {X_train.shape[1]}
    
    Plots Generated:
    """
    
    if plot_results.get('shap_bar'):
        summary_report += f"\n  ✓ SHAP Summary Plot"
    if plot_results.get('importance'):
        summary_report += f"\n  ✓ Feature Importance Plot"
    if plot_results.get('sample_map'):
        summary_report += f"\n  ✓ Sample Geographic Map"
    if plot_results.get('confusion_matrix'):
        summary_report += f"\n  ✓ Confusion Matrix"
    if plot_results.get('roc_curve'):
        summary_report += f"\n  ✓ ROC Curve"
    
    summary_report += f"\n\nOutput Directory: {model_dir}\n"
    
    logger.info(summary_report)
    
    # Save summary
    summary_path = model_dir / "INTERPRETABILITY_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    results['summary'] = summary_path
    
    return results


def integrate_plots_with_ml_pipeline(ml_output_dir: Path, adata, ml_config=None):
    """
    Convenience function to integrate interpretable plots with existing ML output.
    Scans ML output directory for trained models and generates interpretability plots.
    
    Args:
        ml_output_dir: Output directory from ML pipeline
        adata: AnnData object with metadata
        ml_config: ML configuration
    """
    logger.info("Generating interpretable ML visualizations...")
    
    # Find model directories
    model_dirs = list(ml_output_dir.glob("**/models"))
    
    if not model_dirs:
        logger.warning(f"No model directories found in {ml_output_dir}")
        return {}
    
    all_results = {}
    
    for model_dir in model_dirs:
        try:
            # Load model results if available
            results_file = model_dir / "results.pkl"
            if results_file.exists():
                import pickle
                with open(results_file, 'rb') as f:
                    model_results = pickle.load(f)
                
                # Try to generate plots
                if "model" in model_results and "X_train" in model_results:
                    plots = generate_model_interpretability_plots(
                        model_dir=model_dir.parent / "plots",
                        X_train=model_results["X_train"],
                        X_test=model_results["X_test"],
                        y_train=model_results["y_train"],
                        y_test=model_results["y_test"],
                        metadata=adata.obs,
                        model=model_results["model"],
                        task_type=model_results.get("task_type", "Classification"),
                        target_name=model_results.get("target", "Unknown"),
                        config=ml_config
                    )
                    
                    all_results[str(model_dir)] = plots
                    logger.info(f"✓ Generated plots for {model_dir.parent.name}")
                    
        except Exception as e:
            logger.debug(f"Could not generate plots for {model_dir}: {e}")
            continue
    
    return all_results
