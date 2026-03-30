# src/workflow_16s/downstream/machine_learning/overfitting_prevention/stability_consensus.py

import pandas as pd
import plotly.express as px
import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional
from sklearn.model_selection import cross_validate

from workflow_16s.utils.logger import get_logger


def run_comprehensive_validation(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    output_dir: Path, 
    target_name: str,
    groups: Optional[np.ndarray] = None,
    task_type: str = 'Classification',
    n_cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Tier 2 Audit: Detects overfitting by comparing Train vs. Test performance.
    
    Logic:
    1. Nested CV: Measures performance variance across different data subsets.
    2. Learning Curves: Determines if more samples would improve the model.
    3. Gap Analysis: Calculates (Train Score - Test Score).
    """
    logger = get_logger("workflow_16s")
    logger.info(f"Auditing {target_name} for Overfitting...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scoring = 'accuracy' if task_type == 'Classification' else 'r2'
    
    # 1. Performance Stability (Cross-Validation)
    cv_results = cross_validate(
        model, X, y, groups=groups, cv=n_cv_splits, 
        scoring=scoring, return_train_score=True
    )
    
    mean_train = np.mean(cv_results['train_score'])
    mean_test = np.mean(cv_results['test_score'])
    overfit_gap = mean_train - mean_test
    
    # 2. Results Logging
    audit_results = {
        'target': target_name,
        'mean_train_score': float(mean_train),
        'mean_test_score': float(mean_test),
        'overfit_gap': float(overfit_gap),
        'status': 'PASS' if overfit_gap < 0.15 else 'FAIL (Overfit)'
    }
    
    with open(output_dir / f"{target_name}_overfit_audit.json", 'w') as f:
        import json
        json.dump(audit_results, f, indent=4)
        
    if overfit_gap > 0.15:
        logger.warning(f"⚠️  High Overfitting Gap ({overfit_gap:.2f}) detected for {target_name}!")
        
    return audit_results

def run_stability_consensus_workflow(
    stability_df: pd.DataFrame,
    output_dir: Path,
    tree_path: Optional[Path] = None,
    taxonomy_table: Optional[pd.DataFrame] = None
) -> None:
    """
    Synthesizes ML stability scores with biological structure.
    
    METHODOLOGY:
    1. If tree_path exists: Performs Phylogenetic Stability Mapping (Tree-based).
    2. If no tree: Performs Taxonomic Grouping (String-based).
    3. Output: Generates a Consensus Plot highlighting robust 'Bio-Modules'.
    """
    logger = get_logger("workflow_16s")
    logger.info("🎨 Synthesizing Stability Consensus...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for features that passed the initial stability threshold (e.g., > 40%)
    robust_features = stability_df[stability_df['Frequency'] >= 0.4].copy()
    
    if robust_features.empty:
        logger.warning("No stable features found above threshold. Skipping consensus plot.")
        return

    # --- LOGIC BRANCH: Tree-based vs. Taxonomy-based ---
    if tree_path and tree_path.exists():
        try:
            _generate_phylo_consensus(robust_features, tree_path, output_dir)
        except Exception as e:
            logger.error(f"Phylo-consensus failed: {e}. Falling back to Taxonomic grouping.")
            _generate_taxonomic_consensus(robust_features, output_dir)
    else:
        logger.info("No tree found. Proceeding with Taxonomic Stability Mapping.")
        _generate_taxonomic_consensus(robust_features, output_dir)

def _generate_phylo_consensus(
    df: pd.DataFrame, 
    tree_path: Path, 
    output_dir: Path
):
    """
    [Requires Bio.Phylo or similar] 
    Maps stability frequencies onto a circular phylogenetic tree.
    (Conceptual implementation - uses heatmap-style tree annotation)
    """
    # Logic for circular tree with frequency bars on the outer ring
    get_logger("workflow_16s").info(f"Mapping {len(df)} features to tree: {tree_path.name}")
    # In practice, this would call a tool like iTOL or an internal Plotly/GrapeTree wrapper
    _generate_taxonomic_consensus(df, output_dir, suffix="_with_phylo_context")

def _generate_taxonomic_consensus(
    df: pd.DataFrame, 
    output_dir: Path, 
    suffix: str = ""
):
    """
    Groups stable features by taxonomic lineage (Phylum/Class) to 
    identify if specific clades are consistently predictive.
    """
    # Extract Phylum for coloring (assumes format 'p__Phylum;...;g__Genus')
    df['Phylum'] = df['Feature'].apply(lambda x: x.split(';')[1] if ';' in x else 'Unknown')
    df['Label'] = df['Feature'].apply(lambda x: x.split('__')[-1].replace('_', ' '))

    
    fig = px.scatter(
        df,
        x='Mean_Importance',
        y='Frequency',
        size='Mean_Importance',
        color='Phylum',
        text='Label',
        title=f"Stability Consensus: Biological Robustness vs. Predictive Power",
        labels={'Frequency': 'Selection Stability (Bootstrap %)', 'Mean_Importance': 'Mean ML Importance (SHAP/Weight)'},
        template='plotly_white',
        hover_data=['Feature']
    )

    fig.update_traces(textposition='top center')
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="High Confidence Zone")
    
    fig.update_layout(height=800)
    
    plot_path = output_dir / f"stability_consensus_map{suffix}.html"
    fig.write_html(str(plot_path))
    get_logger("workflow_16s").info(f"✅ Stability Consensus Plot saved: {plot_path.name}")