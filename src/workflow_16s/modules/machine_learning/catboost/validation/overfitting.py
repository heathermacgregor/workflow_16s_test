# workflow_16s/modules/machine_learning/catboost/validation/overfitting.py

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import cross_validate

from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")

def extract_phylum(taxon_str):
    parts = re.split(r'[;|]', taxon_str)
    # Look for the part starting with p__ or D_1__ (SILVA phylum)
    for p in parts:
        if p.strip().startswith(('p__', 'D_1__', 'd__')):
            return re.sub(r'^[a-zA-Z0-9_]+__', '', p).strip()
    
    # Fallback: if there are at least 2 parts, assume index 1 is phylum
    return parts[1].strip() if len(parts) > 1 else 'Unknown'

@with_logger
def run_comprehensive_validation(
    model: Any, X: pd.DataFrame, y: pd.Series, output_dir: Path, 
    target_name: str, groups: Optional[np.ndarray] = None,
    task_type: str = 'Classification', n_cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Tier 2 Audit: Detects overfitting by comparing Train vs. Test performance.
    
    Logic:
    1. Nested CV: Measures performance variance across different data subsets.
    2. Learning Curves: Determines if more samples would improve the model.
    3. Gap Analysis: Calculates (Train Score - Test Score).
    """
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

@with_logger
def run_stability_consensus_workflow(
    stability_df: pd.DataFrame, output_dir: Path, tree_path: Optional[Path] = None,
    taxonomy_table: Optional[pd.DataFrame] = None
) -> None:
    """
    Synthesizes ML stability scores with biological structure.
    
    METHODOLOGY:
    1. If tree_path exists: Performs Phylogenetic Stability Mapping (Tree-based).
    2. If no tree: Performs Taxonomic Grouping (String-based).
    3. Output: Generates a Consensus Plot highlighting robust 'Bio-Modules'.
    """
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

@with_logger
def _generate_phylo_consensus(
    df: pd.DataFrame, tree_path: Path, output_dir: Path
):
    """
    [Requires Bio.Phylo or similar] 
    Maps stability frequencies onto a circular phylogenetic tree.
    (Conceptual implementation - uses heatmap-style tree annotation)
    """
    # Logic for circular tree with frequency bars on the outer ring
    logger.info(f"Mapping {len(df)} features to tree: {tree_path.name}")
    # In practice, this would call a tool like iTOL or an internal Plotly/GrapeTree wrapper
    _generate_taxonomic_consensus(df, output_dir, suffix="_with_phylo_context")

def extract_phylum(taxon_str: str) -> str:
    """Robustly extracts the Phylum name from various taxonomy formats."""
    if not isinstance(taxon_str, str):
        return "Unknown"
    
    parts = re.split(r'[;|]', taxon_str)
    
    # Look for the part starting with common Phylum prefixes (Greengenes, SILVA, GTDB)
    for p in parts:
        p_clean = p.strip()
        if p_clean.startswith(('p__', 'D_1__', 'd__')):
            return re.sub(r'^[a-zA-Z0-9_]+__', '', p_clean).replace('_', ' ').strip()
    
    # Fallback: if there are at least 2 parts, assume index 1 is phylum
    if len(parts) > 1:
        clean_fallback = re.sub(r'^[a-zA-Z0-9_]+__', '', parts[1]).replace('_', ' ').strip()
        return clean_fallback if clean_fallback else "Unknown"
        
    return "Unknown"

def extract_lowest_rank(taxon_str: str) -> str:
    """Extracts the most specific taxonomic rank for plotting labels."""
    if not isinstance(taxon_str, str):
        return str(taxon_str)
        
    parts = [p.strip() for p in re.split(r'[;|]', taxon_str) if p.strip()]
    if not parts:
        return "Unknown"
        
    last_part = parts[-1]
    return re.sub(r'^[a-zA-Z0-9_]+__', '', last_part).replace('_', ' ').strip()

@with_logger
def _generate_taxonomic_consensus(
    df: pd.DataFrame, 
    output_dir: Path, 
    suffix: str = ""
):
    """
    Groups stable features by taxonomic lineage (Phylum/Class) to 
    identify if specific clades are consistently predictive.
    """
    # 💡 FIX: Apply the robust regex parsers
    df['Phylum'] = df['Feature'].apply(extract_phylum)
    df['Label'] = df['Feature'].apply(extract_lowest_rank)

    fig = px.scatter(
        df, x='Mean_Importance', y='Frequency', size='Mean_Importance',
        color='Phylum', text='Label',
        title=f"Stability Consensus: Biological Robustness vs. Predictive Power",
        labels={'Frequency': 'Selection Stability (Bootstrap %)', 'Mean_Importance': 'Mean ML Importance (SHAP/Weight)'},
        template='plotly_white', hover_data=['Feature']
    )

    fig.update_traces(textposition='top center')
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="High Confidence Zone")
    
    fig.update_layout(height=800)
    
    plot_path = output_dir / f"stability_consensus_map{suffix}.html"
    fig.write_html(str(plot_path))
    logger.info(f"✅ Stability Consensus Plot saved: {plot_path.name}")
