# feature_selection/reporting.py

import logging
import pandas as pd
import numpy as np
import shap
import traceback
import json
from typing import Tuple, Union, Optional, Any, Dict, List
from pathlib import Path
from scipy.stats import spearmanr

# Type Aliases
PathLike = Union[str, Path]

logger = logging.getLogger('workflow_16s')

# --- SECTION 1: FUNCTIONAL DATABASE LOADERS ---

def load_faprotax(file_path: PathLike) -> Dict[str, List[str]]:
    """
    Parses a FAPROTAX database file into a Taxon -> [Functions] mapping.
    
    Format expected: 
    function: <name>
    taxa:
    <taxon_1>
    <taxon_2>
    """
    faprotax_map = {}
    current_func = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('function:'):
                    current_func = line.replace('function:', '').strip()
                elif line.startswith('taxa:'):
                    continue
                elif current_func:
                    taxon = line.strip()
                    if taxon not in faprotax_map:
                        faprotax_map[taxon] = []
                    faprotax_map[taxon].append(current_func)
        
        logger.info(f"✓ FAPROTAX loaded: {len(faprotax_map)} taxa mapped to functions.")
        return faprotax_map
    except Exception as e:
        logger.error(f"Failed to load FAPROTAX: {e}")
        return {}

# TODO: Implement BugBase Loader
# BugBase usually provides a per-sample trait table. 
# Integration would involve correlating taxa to predicted phenotypes like 'Oxygen_Tolerance'.
def _load_bugbase(file_path: PathLike):
    pass

# TODO: Implement MACADAM Loader
# MACADAM focuses on metabolic pathways (Pathway Completion Limits).
# Integration would involve mapping taxids/names to specific Pathway IDs.
def _load_macadam(file_path: PathLike):
    pass


# --- SECTION 2: ENRICHMENT ENGINE ---

def _get_functional_annotation(
    taxon_string: str, 
    faprotax_db: Dict[str, List[str]]
) -> str:
    """
    Matches a 16S taxon string to functional databases.
    Handles prefixes (g__, s__) and searches up the taxonomic lineage.
    """
    # 1. Clean the string: "k__Bacteria; p__Firmicutes; g__Bacillus" -> ["Bacteria", "Firmicutes", "Bacillus"]
    parts = [p.split('__')[-1].replace('_', ' ').strip() for p in taxon_string.split(';')]
    
    # 2. Search FAPROTAX (search from specific to general)
    matched_functions = []
    for taxon_name in reversed(parts):
        if taxon_name in faprotax_db:
            matched_functions.extend(faprotax_db[taxon_name])
            
    if matched_functions:
        # Deduplicate and return
        return "; ".join(sorted(list(set(matched_functions))))

    # Placeholder: Insert BugBase/MACADAM matching logic here
    
    return "No functional data available"


# --- SECTION 3: REPORTING ---

def generate_shap_report(
    model: Any, 
    X: pd.DataFrame, 
    K: int = 20,
    faprotax_path: Optional[PathLike] = None
) -> Tuple[str, pd.DataFrame]:
    """
    Generates detailed text and data reports with FAPROTAX functional enrichment.
    """
    # Load database if path provided
    faprotax_db = load_faprotax(faprotax_path) if faprotax_path else {}

    try: 
        expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(X) 
        
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else np.mean([np.abs(s) for s in sv], axis=0)
        
        if sv.ndim == 3: 
            sv = sv.mean(axis=2) 
        
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:K]
        top_f = list(X.columns[idx])
        top_m = mean_abs[idx]

        report_data = []
        lines = [f"Impact Report (Top {K} Taxa):", "="*120]
        header = f"{'Taxon':<30} | {'Impact':<8} | {'Correlation':<12} | {'FAPROTAX Functions'}"
        lines.append(header)
        lines.append("-" * 120)
        
        for f, impact_val in zip(top_f, top_m):
            vals = X[f].values
            f_idx = X.columns.get_loc(f)
            shap_vector = sv[:, f_idx]
            
            # Correlation
            with np.errstate(all='ignore'):
                corr_result = spearmanr(vals, shap_vector, nan_policy='omit')
                # Use index access for compatibility with all scipy versions
                rho = corr_result[0]
                # Ensure rho is a float for comparison
                try:
                    rho = float(rho) # type: ignore
                except (TypeError, ValueError):
                    rho = 0.0
            
            rho = rho if not np.isnan(rho) else 0.0
            direction = "Positive (+)" if rho > 0 else "Negative (-)"
            
            # Functional Annotation
            if faprotax_db:
                func_note = _get_functional_annotation(f, faprotax_db)
                clean_name = f.split('__')[-1].replace('_', ' ')
                
                line = (f"{clean_name[:30]:<30} | {impact_val:.4f} | "
                        f"{direction:<12} | {func_note}")
                lines.append(line)
            else:
                clean_name = f.split('__')[-1].replace('_', ' ')
                
                line = (f"{clean_name[:30]:<30} | {impact_val:.4f} | "
                        f"{direction:<12} | No functional database provided")
                lines.append(line)
                func_note = "No functional database provided"
            
            report_data.append({
                'feature': f,
                'mean_abs_shap': impact_val,
                'spearman_rho': rho,
                'functional_functions': func_note
            })
            
        return "\n".join(lines), pd.DataFrame(report_data)
    
    except Exception as e: 
        logger.error(f"SHAP Report Failure: {e}")
        return f"Error: {str(e)}", pd.DataFrame()
    

def save_feature_importances(df: pd.DataFrame, output_dir: Path, filename: str = "top_features.csv"):
    """
    Standardizes the saving of importance scores.
    Ensures the 'feature' and 'importance' columns are correctly named for the 
    downstream Visualization and Meta-Analysis modules.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    
    # Standardize column names to lowercase to avoid import errors in visualizations
    df.columns = [c.lower() for c in df.columns]
    
    if 'feature' not in df.columns or 'importance' not in df.columns:
        logger.warning(f"Unexpected columns in importance DF: {df.columns}. Attempting to fix...")
        # Fallback logic if columns are named differently
        if len(df.columns) >= 2:
            df.columns = ['feature', 'importance'] + list(df.columns[2:])

    df.to_csv(out_path, index=False)
    logger.info(f"💾 Feature importances saved to {out_path}")