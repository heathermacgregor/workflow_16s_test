import logging
logger = logging.getLogger(__name__)
# workflow_16s/modules/machine_learning/catboost/feature_selection/validation.py

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, zscore
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from workflow_16s.utils.logger import with_logger


@with_logger
def check_compositionality_transformation(X: pd.DataFrame) -> None:
    """
    Heuristic check to warn users if the input data does not appear to be 
    Centered Log-Ratio (CLR) transformed.
    """
    X_numeric = X.select_dtypes(include=['number'])
    
    if X_numeric.empty:
        return

    # Check 1: Are there any negative values?
    # Raw counts and relative abundances will be strictly >= 0.
    if (X_numeric >= 0).all().all():
        logger.warning(
            "⚠️ COMPOSITIONALITY ALERT: The feature matrix contains strictly non-negative values. "
            "16S data should typically be CLR-transformed upstream to avoid spurious correlations "
            "and severe feature selection bias. Please verify your preprocessing pipeline."
        )
        return # Skip the row-sum check if it already failed the negativity check

    # Check 2: Do the rows sum to approximately zero?
    # CLR vectors must sum to 0. We use a generous tolerance for float precision.
    row_sums = X_numeric.sum(axis=1)
    if not np.allclose(row_sums, 0, atol=1e-3):
        logger.warning(
            "⚠️ COMPOSITIONALITY ALERT: The row sums of your numeric features do not approximate 0. "
            "While negative values are present, the data does not strictly match a CLR distribution. "
            "Ensure standardizers or other scalers haven't warped the compositional space."
        )


@with_logger
def fix_compositionality_if_needed(X: pd.DataFrame, tolerance: float = 1e-3, auto_fix: bool = False) -> pd.DataFrame:
    """
    Detects and optionally fixes compositionality issues in CLR-transformed data.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix (samples × features)
    tolerance : float
        Tolerance for row sum check (default: 1e-3, matches check_compositionality_transformation)
    auto_fix : bool
        If True, automatically re-applies CLR transformation if row sums deviat too much.
        If False, only logs warnings but returns data unchanged.
    
    Returns:
    --------
    pd.DataFrame
        Original data (if compositionality OK or auto_fix=False) or re-CLR-transformed data.
    """
    X_numeric = X.select_dtypes(include=['number'])
    
    if X_numeric.empty:
        return X
    
    # Check row sums
    row_sums = X_numeric.sum(axis=1)
    is_composable = np.allclose(row_sums, 0, atol=tolerance)
    
    if is_composable:
        logger.info("✅ Data compositionality check passed (row sums ≈ 0)")
        return X
    
    # Compositionality problem detected
    logger.warning(f"⚠️ COMPOSITIONALITY ISSUE: Row sums deviate from 0 (max deviation: {row_sums.abs().max():.6f})")
    
    if not auto_fix:
        logger.info("   → auto_fix=False: Returning data unchanged. Set auto_fix=True to auto-correct.")
        return X
    
    # AUTO-FIX: Re-apply CLR transformation
    logger.info("🔄 AUTO-FIX: Re-applying Centered Log-Ratio (CLR) transformation...")
    
    try:
        # 1. Convert to composition (relative abundance)
        # Handle both dense and sparse matrices
        if isinstance(X_numeric.values, np.ndarray):
            counts = X_numeric.values
        else:
            counts = np.asarray(X_numeric.values)
        
        # 2. Add pseudocount to avoid log(0)
        counts_pseudo = counts + 1.0
        
        # 3. Compute geometric mean per sample
        # log(geometric_mean) = mean(log(x))
        log_counts = np.log(counts_pseudo)
        log_geom_mean = log_counts.mean(axis=1, keepdims=True)  # Shape: (n_samples, 1)
        
        # 4. CLR = log(x_i) - log(geometric_mean)
        X_clr = log_counts - log_geom_mean
        
        # 5. Verify row sums are now ~0
        row_sums_fixed = X_clr.sum(axis=1)
        max_deviation = row_sums_fixed.abs().max()
        
        if np.allclose(row_sums_fixed, 0, atol=tolerance):
            logger.info(f"✅ CLR re-applied successfully! Row sum deviation: {max_deviation:.2e}")
            
            # Return as DataFrame with original index and column names
            X_fixed = pd.DataFrame(
                X_clr,
                index=X_numeric.index,
                columns=X_numeric.columns
            )
            
            # Preserve non-numeric columns from original X
            for col in X.columns:
                if col not in X_fixed.columns:
                    X_fixed[col] = X[col]
            
            return X_fixed
        else:
            logger.error(f"❌ CLR re-application failed: Row sums still deviate (max: {max_deviation:.6f})")
            logger.warning("   → Returning original data unchanged")
            return X
            
    except Exception as e:
        logger.error(f"❌ Error during CLR re-application: {e}")
        logger.warning("   → Returning original data unchanged")
        return X

@with_logger
def check_for_data_leakage(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.95
) -> bool:
    """
    Checks if any single feature is too highly correlated with the target,
    suggesting potential data leakage or a technical artifact.
    """
    for col in X.columns:
        correlation = X[col].corr(y)
        if abs(correlation) > threshold:
            logger.warning(f"⚠️ LEAKAGE ALERT: Feature '{col}' has {correlation:.2f} correlation with target.")
    return True

@with_logger
def filter_data(
    X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame, group_col: str, 
    test_size: float = 0.3, random_state: int = 42, 
    min_samples_per_class: int = 2, task_type: str = 'Classification',
    cv_groups: Optional[np.ndarray] = None, output_dir: Optional[Union[str, Path]] = None,
    filter_outliers: bool = True, outlier_z_threshold: float = 3.0,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filters data ensuring Group-Level consistency, class balance, and sample quality.
    
    This gatekeeper removes outliers (low depth/extreme diversity) before splitting 
    to ensure the model learns from high-quality biological signals.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features).
    y : pd.Series
        Target vector.
    metadata : pd.DataFrame
        Sample metadata containing grouping information.
    group_col : str
        Column in metadata defining sample groups.
    test_size : float   
        Proportion of data to reserve for testing.
    random_state : int
        Random seed for reproducibility.
    min_samples_per_class : int
        Minimum samples required per class (classification only).
    task_type : str
        'Classification' or 'Regression'.
    cv_groups : Optional[np.ndarray]
        Group labels for GroupShuffleSplit.
    output_dir : Optional[Union[str, Path]]
        Directory to save diagnostic plots.
    filter_outliers : bool
        Whether to filter outlier samples.
    outlier_z_threshold : float
        Z-score threshold to define outliers.
    **kwargs : Additional parameters.

    Returns
    -------
    Tuple containing:
        X_train : pd.DataFrame
            Training feature matrix.
        X_test : pd.DataFrame
            Testing feature matrix.
        y_train : pd.Series
            Training target vector.
        y_test : pd.Series
            Testing target vector.
        g_train : Optional[np.ndarray]
            Training group labels (if cv_groups provided).
        g_test : Optional[np.ndarray]
            Testing group labels (if cv_groups provided).
    """
    out_path = Path(output_dir) if output_dir else None
    
    # 1. Sync Metadata
    if not X.index.isin(metadata.index).all():
        metadata = metadata.loc[X.index]
    
    y_labels = metadata.loc[X.index, group_col]
    
    # INTERNAL HELPER: Safe mask application for groups
    def safe_mask_groups(groups, mask_obj):
        if groups is None: return None
        # Convert mask to raw boolean array (handling both Series and ndarray)
        m_arr = mask_obj.values if hasattr(mask_obj, 'values') else mask_obj
        m_arr = np.asarray(m_arr, dtype=bool)
        # Handle groups whether it is a Series or ndarray
        g_raw = groups.values if hasattr(groups, 'values') else groups
        return g_raw[m_arr]

    # 2. Outlier Detection (Pre-Filtering)
    if filter_outliers:
        X, y, y_labels, cv_groups = _handle_outliers(
            X, y, y_labels, cv_groups, outlier_z_threshold, out_path
        )

    # 3. Task-Specific Filtering
    stratify_target = None 
    if task_type == 'Classification':
        counts = y_labels.value_counts()
        keep_idx = counts[counts >= min_samples_per_class].index
        
        if len(keep_idx) < len(counts):
            logger.info(f"Filtering {len(counts) - len(keep_idx)} rare classes.")
            mask = y_labels.isin(keep_idx)
            X, y, y_labels = X.loc[mask], y.loc[mask], y_labels.loc[mask]
            cv_groups = safe_mask_groups(cv_groups, mask)      
        
        stratify_target = y_labels 
    else: 
        mask = y_labels.notna()
        X, y = X.loc[mask], y.loc[mask]
        cv_groups = safe_mask_groups(cv_groups, mask)

    if X.empty or (task_type == 'Classification' and y_labels.nunique() < 2):
        logger.error("Data filtering resulted in insufficient samples.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None, None

    # 4. Splitting Strategy
    try:
        def _plot_class_balance(y_train, y_test, out_dir):
            """
            Plots class distribution in train and test splits.
            """
            import matplotlib.pyplot as plt
            import pandas as pd
        
            train_counts = y_train.value_counts().sort_index()
            test_counts = y_test.value_counts().sort_index()
            df = pd.DataFrame({'Train': train_counts, 'Test': test_counts}).fillna(0)
        
            df.plot(kind='bar', figsize=(10, 6))
            plt.title("Class Balance: Train vs Test")
            plt.ylabel("Sample Count")
            plt.xlabel("Class")
            plt.tight_layout()
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / "class_balance.png")
            plt.close()
            
        # Ensure groups are a raw array for scikit-learn
        g_arr = cv_groups.values if hasattr(cv_groups, 'values') else cv_groups # type: ignore

        if g_arr is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=g_arr))
            
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            g_tr, g_te = g_arr[train_idx], g_arr[test_idx]
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
            )
            g_tr, g_te = None, None

        if task_type == 'Classification' and out_path:
            _plot_class_balance(y_tr, y_te, out_path)
        
        return X_tr, X_te, y_tr, y_te, g_tr, g_te
        
    except Exception as e:
        logger.warning(f"Split failed ({e}). Falling back to simple random split.")
        # Ensure we return aligned types
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_tr, X_te, y_tr, y_te, None, None

@with_logger
def _handle_outliers(X, y, y_labels, cv_groups, threshold, out_path):
    """
    Detects samples with extreme sequencing depth or alpha diversity.
    Safely prunes samples and their corresponding group labels.
    """
    import numpy as np
    from scipy.stats import zscore
    
    # 1. Calculate Metrics
    lib_size = X.select_dtypes(include=['number']).sum(axis=1)
    # ✅ FIX: Select only numeric columns for depth/richness calculations
    X_numeric = X.select_dtypes(include=['number'])

    # 1. Depth Filtration (Library Size)
    lib_size = X_numeric.sum(axis=1) # Use X_numeric instead of X
    
    # 2. Richness Filtration (Observed Features)
    # This was the crashing line: (X > 0).sum(axis=1)
    observed_otus = (X_numeric > 0).sum(axis=1)
    
    # 2. Identify Outliers via Z-Score
    # lib_z and otu_z are numpy arrays
    lib_z = np.abs(zscore(lib_size))
    otu_z = np.abs(zscore(observed_otus))
    
    # outlier_mask is a numpy boolean array
    outlier_mask = (lib_z > threshold) | (otu_z > threshold)
    
    if outlier_mask.any():
        n_outliers = outlier_mask.sum()
        logger.warning(f"🚨 {n_outliers} outliers detected (Z > {threshold} in depth or diversity).")
        
        if out_path:
            _plot_outliers(lib_size, observed_otus, outlier_mask, out_path)
            
        # 3. Create the Cleaning Mask
        clean_mask = ~outlier_mask # Raw numpy boolean array
        
        # 4. Prune Main Data (Pandas .loc works with numpy boolean arrays)
        X, y, y_labels = X.loc[clean_mask], y.loc[clean_mask], y_labels.loc[clean_mask]
        
        # 5. SAFE SLICING for cv_groups (NumPy vs Pandas)
        if cv_groups is not None:
            # Get raw array if it's a Series, otherwise use as-is
            g_raw = cv_groups.values if hasattr(cv_groups, 'values') else cv_groups
            # Slice using the numpy boolean mask directly (No .values needed)
            cv_groups = g_raw[clean_mask]
            
    return X, y, y_labels, cv_groups

@with_logger
def _handle_outliers(X, y, y_labels, cv_groups, threshold, out_path):
    """
    Detects samples with extreme sequencing depth or alpha diversity
    using robust Median Absolute Deviation (MAD).
    """
    def robust_zscore(x: np.ndarray) -> np.ndarray:
        """Calculates the modified Z-score using MAD."""
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        if mad == 0:
            # If the data is entirely uniform, avoid division by zero
            return np.zeros_like(x, dtype=float) 
        # 0.6745 scales MAD to approximate the standard deviation of a normal distribution
        modified_z = 0.6745 * (x - median) / mad
        return np.abs(modified_z)

    # 1. Calculate Metrics
    X_numeric = X.select_dtypes(include=['number'])

    # Depth Filtration (Library Size)
    # 🚨 Add a pseudo-count of 1 to avoid log10(0) on empty samples
    lib_size = X_numeric.sum(axis=1)
    log_lib_size = np.log10(lib_size + 1) 
    
    # Richness Filtration (Observed Features)
    observed_otus = (X_numeric > 0).sum(axis=1)

    # 2. Identify Outliers via Robust Z-Score (MAD)
    # We apply this to the log-transformed depth, and the raw richness
    lib_z_robust = robust_zscore(log_lib_size.values)
    otu_z_robust = robust_zscore(observed_otus.values)

    # outlier_mask is a numpy boolean array
    outlier_mask = (lib_z_robust > threshold) | (otu_z_robust > threshold)

    if outlier_mask.any():
        n_outliers = outlier_mask.sum()
        logger.warning(f"🚨 {n_outliers} outliers detected (Robust Z > {threshold} in depth or diversity).")
        
        if out_path:
            # Pass the raw lib_size for the plot so the axes are interpretable to the user
            _plot_outliers(lib_size.values, observed_otus.values, outlier_mask, out_path)
            
    # 3. Create the Cleaning Mask
    clean_mask = ~outlier_mask 
    
    # 4. Prune Main Data 
    X_clean = X.loc[clean_mask]
    y_clean = y.loc[clean_mask]
    y_labels_clean = y_labels.loc[clean_mask]
    
    # 5. SAFE SLICING for cv_groups 
    if cv_groups is not None:
        g_raw = cv_groups.values if hasattr(cv_groups, 'values') else cv_groups
        cv_groups_clean = g_raw[clean_mask]
    else:
        cv_groups_clean = None
        
    return X_clean, y_clean, y_labels_clean, cv_groups_clean

def _plot_outliers(lib_size, otus, mask, out_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(lib_size[~mask], otus[~mask], alpha=0.6, label='Kept Samples')
    plt.scatter(lib_size[mask], otus[mask], color='red', marker='x', label='Outliers')
    plt.xlabel("Library Size (Total Reads)")
    plt.ylabel("Observed Features (Richness)")
    plt.title("Outlier Detection: Depth vs. Diversity")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "outlier_diagnostic.png")
    plt.close()
    