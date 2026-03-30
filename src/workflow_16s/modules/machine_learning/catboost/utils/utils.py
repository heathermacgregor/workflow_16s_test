# workflow_16s/modules/machine_learning/catboost/utils/utils.py

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef

from workflow_16s.constants import EXPECTED_VAR_COLUMNS
from workflow_16s.utils.logger import with_logger

def sanitize_catboost_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean parameters to prevent 'n_estimators' vs 'iterations' conflicts."""
    clean = params.copy()
    aliases = {'n_estimators', 'num_boost_round', 'num_trees', 'iterations'}
    keys_present = [k for k in clean.keys() if k in aliases]
    if len(keys_present) > 1:
        primary = 'iterations' if 'iterations' in keys_present else keys_present[0]
        for k in keys_present:
            if k != primary: clean.pop(k, None)
    return clean

def get_model_class(task_type: str, algorithm: str) -> type:
    """Factory to return the correct model class based on task and algorithm."""
    if algorithm.lower() == 'catboost':
        return CatBoostRegressor if task_type.lower() == 'regression' else CatBoostClassifier
    return RandomForestRegressor if task_type.lower() == 'regression' else RandomForestClassifier

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes feature names for the validation tier.
    Removes brackets and special characters that break CatBoost and LaTeX reporting.
    """
    df = df.copy()
    # Replace common illegal characters
    df.columns = [
        str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_').replace('.', '_')
        for col in df.columns
    ]
    return df

def resolve_feature_names(adata_agg: Any, level: str) -> List[str]:
    """
    Standardized taxonomy resolver for the Validation Tier.
    Extracts Genus/Species names from AnnData or raw semicolon strings.
    """
    if 'taxon_name' in adata_agg.var.columns:
        return adata_agg.var['taxon_name'].tolist()
    
    # Fallback: clean the index names
    return [str(name).split(';')[-1] for name in adata_agg.var_names]

def format_audit_results(train_score: float, test_score: float, target: str) -> Dict[str, Any]:
    """Calculates the scientific 'Generalization Gap' for forensic records."""
    gap = train_score - test_score
    return {
        "target": target,
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "overfitting_gap": round(gap, 4),
        "status": "PASS" if gap < 0.15 else "FAIL"
    }

def resolve_feature_names(
    adata_agg: Any, 
    level: str
) -> List[str]:
    """
    Extracts high-quality taxonomic names from AnnData var names.
    Handles 'Genus', 'ASV', or semicolon-delimited strings.
    """
    if 'taxon_name' in adata_agg.var.columns:
        return adata_agg.var['taxon_name'].tolist()
    
    # Fallback: cleaning long semicolon strings
    raw_names = adata_agg.var_names.tolist()
    clean_names = []
    for name in raw_names:
        parts = [p.strip() for p in name.split(';') if p.strip()]
        if len(parts) > 0:
            clean_names.append(parts[-1])
        else:
            clean_names.append(name)
    return clean_names

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes characters that break tree-based ML models (CatBoost, LightGBM)
    and LaTeX. Cleans brackets, dots, commas, spaces, and leading numbers.
    Safely handles resulting duplicate column names.
    """
    df = df.copy()
    
    def _clean_string(col_name: str) -> str:
        name = str(col_name)
        # 1. Replace spaces with underscores
        name = name.replace(' ', '_')
        
        # 2. Remove problematic punctuation: [, ], <, >, ., and ,
        name = re.sub(r'[\[\]<>\.,]', '', name)
        
        # 3. Strip leading digits and any leading underscores that remain
        name = re.sub(r'^[\d_]+', '', name)
        
        # 4. Fallback if the string becomes completely empty
        return name if name else "unnamed_feature"

    # Apply the cleaning function
    cleaned_cols = [_clean_string(col) for col in df.columns]
    
    # 5. Deduplicate names (e.g. if '1_Bacteroides' and '2_Bacteroides' both become 'Bacteroides')
    seen = {}
    final_cols = []
    for col in cleaned_cols:
        if col not in seen:
            seen[col] = 0
            final_cols.append(col)
        else:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
            
    df.columns = final_cols
    return df

def align_data_robust(
    X: pd.DataFrame, 
    obs: pd.DataFrame, 
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Ensures X and y indices match perfectly and drops NaNs in target."""
    common_idx = X.index.intersection(obs.index)
    X_subset = X.loc[common_idx]
    obs_subset = obs.loc[common_idx]
    
    # Drop samples where target is NaN
    mask = obs_subset[target_col].notna()
    X_final = X_subset[mask]
    y_final = obs_subset.loc[mask, target_col]
    meta_final = obs_subset[mask]
    
    return X_final, y_final, meta_final

@with_logger
def verify_model_outputs(
    output_dir: Union[str, Path],
    expected_files: Optional[List[str]] = None
) -> bool:
    """
    Verifies that expected machine learning artifacts were successfully generated,
    are not empty, and are correctly formatted (for JSONs).
    
    Args:
        output_dir: Path to the directory containing model outputs.
        expected_files: List of filenames to check. Defaults to standard ML outputs.
        
    Returns:
        bool: True if all expected files exist and are valid, False otherwise.
    """
    out_path = Path(output_dir)
    
    # 1. Directory Check
    if not out_path.exists() or not out_path.is_dir():
        logger.error(f"❌ Output directory does not exist or is not a directory: {out_path}")
        return False
        
    # 2. Define Expected Artifacts
    if expected_files is None:
        # Defaults based on standard CatBoost / sklearn microbiome pipelines
        expected_files = [
            "metrics.json", 
            "feature_importance.csv", 
            "predictions.csv",
            "model.cbm" # Replace with .pkl if using standard sklearn models
        ]
        
    logger.info(f"🔍 Verifying {len(expected_files)} expected model outputs in {out_path}...")
    
    missing_files = []
    empty_or_invalid_files = []
    
    # 3. File Verification Loop
    for file_name in expected_files:
        file_path = out_path / file_name
        
        # Check Existence
        if not file_path.exists():
            missing_files.append(file_name)
            continue
            
        # Check Size (guard against silent 0-byte file creation)
        if file_path.stat().st_size == 0:
            empty_or_invalid_files.append(f"{file_name} (Empty file)")
            continue
            
        # Check JSON Parsing (guard against corrupted metrics writes)
        if file_name.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError:
                empty_or_invalid_files.append(f"{file_name} (Invalid JSON format)")

    # 4. Report Results
    is_valid = True
    if missing_files:
        logger.error(f"❌ Missing expected output files: {', '.join(missing_files)}")
        is_valid = False
        
    if empty_or_invalid_files:
        logger.error(f"❌ Files exist but are empty or corrupted: {', '.join(empty_or_invalid_files)}")
        is_valid = False
        
    if is_valid:
        logger.info("✅ All model outputs verified successfully.")
        
    return is_valid

def optimize_threshold(
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Microbiome datasets are often imbalanced. Instead of using 0.5, 
    this finds the probability threshold that maximizes the 
    Matthews Correlation Coefficient (MCC).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_mcc = -1.0
    best_thresh = 0.5
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            
    return float(best_thresh), float(best_mcc)

@with_logger
def validate_batch_variance(
    X_df: pd.DataFrame, 
    batch_series: pd.Series, 
    min_samples: int = 5,
    max_zero_var_frac: float = 0.90
) -> Dict[str, Any]:
    """
    Audits batches to ensure they are suitable for centering and ML.
    
    Checks:
    1. Sample Count: Batches with very few samples provide unreliable means.
    2. Sparsity / Zero Variance: Calculates the exact percentage of features 
       that have near-zero variance. If a batch is mostly zero-variance 
       features, it is flagged as biologically suspect or corrupted.
       
    Args:
        X_df: Feature matrix (e.g., CLR-transformed data).
        batch_series: Series containing batch assignments.
        min_samples: Minimum number of samples required to keep a batch.
        max_zero_var_frac: Maximum allowed proportion of zero-variance features (0.0 to 1.0).
    """
    n_features = len(X_df.columns)
    logger.info(f" 📊 Auditing batch variance across {n_features} features...")
    
    report = {
        'passed_batches': [],
        'failed_batches': [],
        'warnings': []
    }
    
    unique_batches = batch_series.unique()
    
    for batch in unique_batches:
        mask = (batch_series == batch)
        X_batch = X_df.loc[mask]
        n_samples = len(X_batch)
        
        # 1. Size Check
        if n_samples < min_samples:
            reason = f"Insufficient N ({n_samples} < {min_samples})"
            report['failed_batches'].append({'batch': batch, 'reason': reason})
            logger.warning(f"Batch '{batch}' failed audit: {reason}")
            continue
            
        # 2. Proportion of Zero-Variance Features Check
        # Calculate variance for each feature individually
        batch_vars = X_batch.var()
        
        # Count how many features have effectively zero variance
        zero_var_count = (batch_vars < 1e-6).sum()
        zero_var_frac = zero_var_count / n_features
        
        if zero_var_frac > max_zero_var_frac:
            reason = f"Extreme sparsity: {zero_var_frac:.1%} of features have zero variance."
            report['failed_batches'].append({'batch': batch, 'reason': reason})
            logger.warning(f"Batch '{batch}' failed audit: {reason}")
            continue
            
        # Optional: Log a warning if it passes but is still highly sparse
        if zero_var_frac > 0.50:
            logger.info(f" ⚠️ Notice: Batch '{batch}' has {zero_var_frac:.1%} zero-variance features, but passed.")
            
        report['passed_batches'].append(batch)

    logger.info(f"Batch Audit Complete: {len(report['passed_batches'])} passed, "
                f"{len(report['failed_batches'])} failed.")
    return report

def apply_batch_centered_clr(
    X_clr: pd.DataFrame, 
    batch_series: pd.Series
) -> pd.DataFrame:
    """
    Applies Batch-Centering with an integrated variance guardrail.
    """
    # TODO: Modify this function to also prune the metadata/obs to only include samples from valid batches, ensuring perfect alignment for downstream ML steps.
    logger = get_logger("workflow_16s")
    # Audit batches first to ensure we only center on valid ones
    audit = validate_batch_variance(X_clr, batch_series)
    valid_batches = audit['passed_batches']
    
    # Subset to valid batches and apply centering only to those
    mask = batch_series.isin(valid_batches)
    X_sub = X_clr.loc[mask]
    batches = batch_series.loc[mask]
    
    if len(X_sub) < len(X_clr):
        n = len(X_clr) - len(X_sub)
        logger.info(f" ✂️ Pruning {n} samples from invalid batches.")

    # Apply centering: x_new = x - mean(batch)
    batch_means = X_sub.groupby(batches).transform('mean')
    X_centered = X_sub - batch_means
    
    return X_centered