# src/workflow_16s/downstream/machine_learning/meta_analysis.py

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

from workflow_16s.utils.logger import get_logger


def perform_meta_analysis(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    batches: np.ndarray, 
    output_dir: Path, 
    n_top_features: int = 20
) -> Dict[str, Any]:
    """
    Performs a Study-by-Study Consensus Analysis to identify 'Universal Biomarkers'.
    
    RATIONALE:
    Global models can be dominated by a single large study. Meta-analysis treats 
    each study as an independent experiment, identifying features that are 
    predictive across different sequencing runs, labs, and cohorts.
    """
    logger = get_logger("workflow_16s")
    # Normalize batch identifiers
    batches = np.array([str(x) if pd.notna(x) else "Unknown" for x in batches])
    unique_batches = np.unique(batches)
    
    logger.info(f"🌎 Starting Meta-Analysis across {len(unique_batches)} independent studies...")
    
    # 1. Detect Task Type
    is_numeric = pd.api.types.is_numeric_dtype(y_series)
    is_regression = is_numeric and (y_series.nunique() > 10)
    task_type = "Regression" if is_regression else "Classification"
    
    # Results Accumulators
    study_feature_map = {} 
    feature_counts = defaultdict(int)
    feature_importance_sum = defaultdict(float)
    valid_studies = 0
    
    # 2. Iterate through individual studies
    for batch in unique_batches:
        mask = (batches == batch)
        X_batch = X_df.loc[mask]
        y_batch = y_series.loc[mask]
        
        # Guardrail: Minimal data for a valid sub-model
        if len(X_batch) < 15: 
            continue
        if (is_regression and y_batch.std() == 0) or (not is_regression and y_batch.nunique() < 2):
            continue
            
        valid_studies += 1
        
        try:
            # Internal Split for Early Stopping (Prevents memorizing batch-specific noise)
            X_tr, X_val, y_tr, y_val = train_test_split(X_batch, y_batch, test_size=0.2, random_state=42)

            params = {
                'iterations': 1000,
                'depth': 4,                    # Shallower depth for smaller study cohorts
                'learning_rate': 0.05,
                'early_stopping_rounds': 50,
                'verbose': False,
                'allow_writing_files': False,
                'thread_count': 4
            }

            model = CatBoostRegressor(**params) if is_regression else CatBoostClassifier(**params)
            
            # Use auto_class_weights if classification is imbalanced
            if not is_regression and y_tr.value_counts().min() < 10:
                model.set_params(auto_class_weights='Balanced')

            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            
            # Extract top features for this study
            importances = model.get_feature_importance()
            top_idx = np.argsort(importances)[::-1][:n_top_features]
            
            batch_top_feats = X_df.columns[top_idx].tolist()
            study_feature_map[batch] = batch_top_feats
            
            for idx in top_idx:
                fname = X_df.columns[idx]
                feature_counts[fname] += 1
                feature_importance_sum[fname] += importances[idx]
                
        except Exception as e:
            logger.warning(f"Meta-model failed for study {batch}: {e}")
    
    logger.info(f"ℹ️ Consensus established from {valid_studies} eligible studies.")
            
    # 3. Aggregate Global Consensus
    meta_results = []
    for feat, count in feature_counts.items():
        meta_results.append({
            'Feature': feat,
            'Study_Frequency': count,
            'Frequency_Pct': count / valid_studies if valid_studies > 0 else 0,
            'Avg_Importance': feature_importance_sum[feat] / valid_studies if valid_studies > 0 else 0
        })
    
    df_meta = pd.DataFrame(meta_results).sort_values(
        ['Study_Frequency', 'Avg_Importance'], 
        ascending=False
    ).head(n_top_features)

    # 4. Persistence and Visualization
    out_path = output_dir / "meta_analysis"
    out_path.mkdir(exist_ok=True, parents=True)
    df_meta.to_csv(out_path / "consensus_biomarkers.csv", index=False)
    
    if not df_meta.empty and valid_studies > 1:
        generate_study_overlap_matrix(study_feature_map, n=10, n_matrix=30, output_dir=out_path)
    
    return {
        'method': 'meta_analysis',
        'valid_studies': valid_studies,
        'consensus_df': df_meta,
        'feature_importance_map': dict(zip(df_meta.Feature, df_meta.Frequency_Pct))
    }

def apply_meta_consensus_weighting(
    global_importance_df: pd.DataFrame, 
    meta_results: Dict[str, Any],
    min_freq_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Adjusts global feature importance based on cross-study consensus.
    
    FORMULA:
        Adjusted_Score = Global_Importance * (Meta_Frequency_Pct + Constant)
    
    This penalizes 'fluke' biomarkers that have high importance in the 
    global model but fail to appear consistently across independent studies.
    """
    logger = get_logger("workflow_16s")
    logger.info("⚖️  Applying Meta-Consensus weighting to global biomarkers...")
    
    # 1. Map Meta-Frequency to the Global DF
    meta_map = meta_results.get('feature_importance_map', {})
    
    # We use a small constant (0.1) so that features not in the meta-top-20 
    # aren't immediately zeroed out, but are heavily penalized.
    global_importance_df['Meta_Frequency'] = global_importance_df['feature'].map(meta_map).fillna(0.05)
    
    # 2. Calculate Robustness-Adjusted Score
    global_importance_df['Adjusted_Importance'] = (
        global_importance_df['importance'] * (global_importance_df['Meta_Frequency'] + 0.1)
    )
    
    # 3. Flag for high-confidence (Universal) status
    global_importance_df['Is_Universal'] = global_importance_df['Meta_Frequency'] >= 0.5
    
    # Sort by the new adjusted score
    df_weighted = global_importance_df.sort_values('Adjusted_Importance', ascending=False)
    
    logger.info(f"✅ Weighting complete. Found {df_weighted['Is_Universal'].sum()} Universal Biomarkers.")
    return df_weighted

from pathlib import Path
def generate_study_overlap_matrix(
    study_feature_map: Dict[str, List[str]], 
    output_dir: Path,
    n: int = 10,
    n_matrix: int = 30
) -> None:
    """Generates an interactive heatmap showing shared biomarkers across top studies."""
    # Select top n studies by count for display
    top_studies = sorted(
        study_feature_map.keys(), 
        key=lambda x: len(study_feature_map[x]), 
        reverse=True
    )[:n]
    all_feats = sorted(list(set([f for s in top_studies for f in study_feature_map[s]])))
    
    if not all_feats: return

    # Binary Presence/Absence matrix
    matrix = pd.DataFrame(0, index=all_feats, columns=top_studies)
    for study in top_studies:
        for feat in study_feature_map[study]:
            matrix.loc[feat, study] = 1

    matrix['Total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('Total', ascending=False).head(n_matrix) # Display top n_matrix most shared

    fig = px.imshow(
        matrix.drop(columns='Total'),
        labels=dict(x="Independent Studies", y="Top Microbial Features", color="Presence"),
        x=top_studies,
        y=[f.split('__')[-1].replace('_', ' ') for f in matrix.index],
        color_continuous_scale=[[0, 'white'], [1, '#4B0082']],
        title="Meta-Analysis Consensus: Biomarker Overlap Across Boundaries"
    )
    
    fig.update_layout(template='plotly_white', coloraxis_showscale=False, height=800)
    fig.write_html(str(output_dir / "study_consensus_matrix.html"))
    logger = get_logger("workflow_16s")
    logger.info(f"✅ Meta-consensus visualization saved to {output_dir}")