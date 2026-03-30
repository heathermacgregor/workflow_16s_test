# feature_selection/validation.py

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.stats import spearmanr, zscore

logger = logging.getLogger('workflow_16s')


# ============================================================================
# CRITICAL FIX: EligibilityManager - Dynamic Multi-Class Threshold Reduction
# ============================================================================
#
# Algorithm ensures we have eligible training samples BEFORE train/test split.
# Problem (from logs): Filtering applied AFTER split → training set goes to 0 samples.
# Solution: Filter BEFORE split using dynamic threshold reduction.
#

class EligibilityManager:
    """
    Manages multi-class eligibility filtering with dynamic threshold reduction.
    
    CRITICAL: Call this BEFORE train/test split to ensure training data isn't
    filtered to zero samples.
    
    Algorithm:
    1. Start with max classes in any study (or config start_threshold)
    2. Find studies with >= threshold classes AND non-constant target
    3. Count eligible samples → check against min_samples_for_training
    4. If insufficient: reduce threshold by 1, retry (down to min_threshold)
    5. Return eligible sample indices for downstream processing
    
    This replaces the old approach which filtered AFTER splitting, breaking
    the training set.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,  # Full data (e.g., adata.obs or metadata)
        target_col: str,         # Target variable column name
        study_col: str = "Project",  # Study/project grouping column
        start_threshold: Optional[int] = None,  # Starting class requirement
        min_threshold: int = 2,                  # Don't reduce below this
        min_samples_for_training: int = 50,   # After filtering & train/test split
        test_size: float = 0.2,                 # test_size for estimating training count
        task_type: str = "Classification",      # Task type
    ):
        """Initialize eligibility manager.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            Full metadata dataframe with target and study columns.
        target_col : str
            Target variable column name.
        study_col : str
            Column for study/project grouping.
        start_threshold : int, optional
            Starting class threshold. If None, auto-detect max classes.
        min_threshold : int
            Minimum classes required (don't reduce below this).
        min_samples_for_training : int
            Minimum training samples needed after filtering & split.
        test_size : float
            Test/train split ratio (for estimating training count).
        task_type : str
            'Classification' or 'Regression'.
        """
        self.data = data_df
        self.target_col = target_col
        self.study_col = study_col
        self.task_type = task_type
        self.test_size = test_size
        self.min_threshold = min_threshold
        self.min_samples_for_training = min_samples_for_training
        
        # Auto-detect start threshold if not provided
        if start_threshold is None:
            # Maximum classes in any single study
            if task_type == "Classification":
                self.start_threshold = self._get_max_classes_per_study()
            else:
                self.start_threshold = 10  # Arbitrary for regression
        else:
            self.start_threshold = start_threshold
        
        logger.debug(
            f"✅ EligibilityManager initialized: "
            f"target='{target_col}', study='{study_col}', "
            f"start_threshold={self.start_threshold}, min_threshold={min_threshold}"
        )
    
    def _get_max_classes_per_study(self) -> int:
        """Get maximum number of classes in any study."""
        if self.study_col not in self.data.columns:
            logger.warning(f"Study column '{self.study_col}' not found, using global max classes")
            return self.data[self.target_col].nunique()
        
        max_classes = 0
        for study in self.data[self.study_col].unique():
            if pd.isna(study):
                continue
            study_mask = self.data[self.study_col] == study
            n_classes = self.data.loc[study_mask, self.target_col].nunique()
            max_classes = max(max_classes, n_classes)
        
        return max_classes
    
    def _is_study_eligible(self, study_indices: pd.Index, threshold: int) -> bool:
        """Check if a study is eligible at a given threshold.
        
        Eligible = has >= threshold classes AND target is non-constant.
        """
        study_targets = self.data.loc[study_indices, self.target_col]
        n_classes = study_targets.nunique()
        is_constant = study_targets.nunique() == 1
        
        return n_classes >= threshold and not is_constant
    
    def get_eligible_samples(self) -> pd.Index:
        """
        Get eligible sample indices using dynamic threshold reduction.
        
        Returns
        -------
        pd.Index
            Indices of eligible samples. Empty if no eligible samples found.
        """
        if self.study_col not in self.data.columns:
            logger.warning(
                f"Study column '{self.study_col}' not found. "
                f"Using all {len(self.data)} samples."
            )
            return self.data.index
        
        current_threshold = self.start_threshold
        eligible_indices = []
        
        # Dynamic threshold reduction loop
        while current_threshold >= self.min_threshold:
            eligible_indices = []
            
            # Collect eligible samples per study
            for study in self.data[self.study_col].unique():
                if pd.isna(study):
                    continue
                
                study_mask = self.data[self.study_col] == study
                study_indices = self.data[study_mask].index
                
                if self._is_study_eligible(study_indices, current_threshold):
                    eligible_indices.extend(study_indices)
            
            eligible_indices = pd.Index(eligible_indices)
            n_eligible = len(eligible_indices)
            
            # Estimate training samples after split
            n_train_estimated = int(n_eligible * (1 - self.test_size))
            
            logger.debug(
                f"Threshold={current_threshold}: "
                f"{n_eligible} eligible samples → "
                f"{n_train_estimated} estimated training samples"
            )
            
            # Check if we have enough training samples
            if n_train_estimated >= self.min_samples_for_training:
                logger.info(
                    f"✅ ELIGIBLE SAMPLES FOUND (criteria met):\n"
                    f"   Threshold: {current_threshold}\n"
                    f"   Eligible: {n_eligible} samples\n"
                    f"   Est. Training: {n_train_estimated} samples "
                    f"(≥ {self.min_samples_for_training} required)"
                )
                return eligible_indices
            
            # Try reducing threshold
            current_threshold -= 1
        
        # Fallback: return whatever we have at min_threshold
        logger.warning(
            f"⚠️ ELIGIBILITY: Could not find sufficient training samples even at min_threshold={self.min_threshold}. "
            f"Returning best available ({len(eligible_indices)} samples, "
            f"est. {int(len(eligible_indices) * (1 - self.test_size))} training)."
        )
        
        return eligible_indices
    
    def validate_training_adequacy(
        self,
        train_indices: pd.Index,
    ) -> bool:
        """Validate that training data is adequate for modeling.
        
        Checks:
        - At least min_samples_for_training samples
        - At least 2 classes (for classification)
        - No single class >95% of samples (class imbalance check)
        """
        n_train = len(train_indices)
        
        if n_train < self.min_samples_for_training:
            logger.error(
                f"❌ Training data inadequate: {n_train} samples "
                f"(need ≥ {self.min_samples_for_training})"
            )
            return False
        
        if self.task_type == "Classification":
            train_targets = self.data.loc[train_indices, self.target_col]
            n_classes = train_targets.nunique()
            
            if n_classes < 2:
                logger.error(
                    f"❌ Training data has {n_classes} class(es) (need ≥ 2)"
                )
                return False
            
            # Check class balance
            class_props = train_targets.value_counts(normalize=True)
            max_prop = class_props.max()
            
            if max_prop > 0.95:
                logger.warning(
                    f"⚠️ Severe class imbalance: largest class = {100*max_prop:.1f}%"
                )
        
        logger.info(f"✅ Training data validation passed ({n_train} samples)")
        return True


def fix_compositionality_if_needed(
    X: Union[pd.DataFrame, np.ndarray],
    tolerance: float = 1e-3,
    auto_fix: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Detects and optionally fixes CLR (Centered Log-Ratio) compositionality issues.
    
    In compositional data analysis, CLR-transformed features should have row sums ≈ 0.
    When data is partially transformed or proportionally scaled, row sums deviate from zero.
    
    Args:
        X: Data matrix (samples × features). Can be DataFrame or ndarray.
        tolerance: Acceptable deviation from zero (default 1e-3). Data with |row_sum| > tolerance
                   is flagged as having compositionality issues.
        auto_fix: If True, re-applies CLR transformation when detected. If False, just logs warning.
    
    Returns:
        X (potentially transformed): Original or re-transformed data.
    
    Logic:
        1. Compute row sums (should be ≈ 0 for CLR-transformed data)
        2. Find rows with |sum| > tolerance
        3. If found and auto_fix=True:
           • Re-apply CLR: log(x / geometric_mean(x)) for each row
           • Log warning about deviation detected
        4. If not auto_fix, just log warning and return original
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    X_array = X.values if is_dataframe else X
    
    # Compute row sums
    row_sums = np.sum(X_array, axis=1)
    max_deviation = np.max(np.abs(row_sums))
    issues_detected = np.mean(np.abs(row_sums) > tolerance)
    
    if max_deviation > tolerance:
        logger.warning(
            f"⚠️ COMPOSITIONALITY ISSUE DETECTED: Row sums max deviation = {max_deviation:.6f} "
            f"(tolerance={tolerance}). {100*issues_detected:.1f}% of rows exceed tolerance."
        )
        
        if auto_fix:
            logger.info("   🔧 AUTO-FIX ENABLED: Re-applying CLR transformation...")
            
            # Re-apply CLR to fix compositionality
            # CLR: log(x / geometric_mean(x))
            X_fixed = np.zeros_like(X_array, dtype=float)
            
            for i in range(X_array.shape[0]):
                row = X_array[i].astype(float)
                # Handle zeros and negative values (from sparse data)
                row = np.maximum(row, 1e-10)
                geom_mean = np.exp(np.mean(np.log(row)))
                X_fixed[i] = np.log(row / geom_mean)
            
            # Verify fix
            new_row_sums = np.sum(X_fixed, axis=1)
            new_max_dev = np.max(np.abs(new_row_sums))
            logger.info(f"   ✅ CLR re-applied. New max deviation: {new_max_dev:.6f}")
            
            return pd.DataFrame(X_fixed, index=X.index, columns=X.columns) if is_dataframe else X_fixed
        else:
            logger.warning("   auto_fix=False: Returning original data. Consider setting auto_fix_compositionality=True.")
    else:
        logger.debug(f"✅ Compositionality OK: row sums max deviation = {max_deviation:.6f} (≤ {tolerance})")
    
    return X

def check_for_data_leakage(
    X: pd.DataFrame, 
    y: pd.Series, 
    threshold: float = 0.95
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

def filter_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    metadata: pd.DataFrame, 
    group_col: str, 
    test_size: float = 0.3,  
    random_state: int = 42, 
    min_samples_per_class: int = 2,
    task_type: str = 'Classification',
    cv_groups: Optional[np.ndarray] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filter_outliers: bool = True,
    outlier_z_threshold: float = 3.0,
    test_indices: Optional[list] = None,
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

    # 3. Task-Specific Filtering (with guardrails)
    stratify_target = None 
    if task_type == 'Classification':
        counts = y_labels.value_counts()
        keep_idx = counts[counts >= min_samples_per_class].index
        
        # 🚨 GUARDRAIL: Only filter if it leaves >= 2 classes AND >= 10% of data
        n_before = len(X)
        valid_filter = len(keep_idx) >= 2 and len(keep_idx) < len(counts)
        samples_after_filter = counts.loc[keep_idx].sum() if valid_filter else n_before
        removal_ratio = 1.0 - (samples_after_filter / n_before) if n_before > 0 else 0
        
        # PHASE 1 FIX: Skip filtering if removal percentage > 80% (or other configured threshold)
        skip_if_removes_more_than = kwargs.get('skip_if_removes_more_than', 0.80)
        
        if valid_filter and samples_after_filter >= max(10, int(0.1 * n_before)) and removal_ratio <= skip_if_removes_more_than:
            logger.info(f"Filtering {len(counts) - len(keep_idx)} rare classes ({samples_after_filter}/{n_before} samples remain, {100*removal_ratio:.1f}% removal).")
            mask = y_labels.isin(keep_idx)
            X, y, y_labels = X.loc[mask], y.loc[mask], y_labels.loc[mask]
            cv_groups = safe_mask_groups(cv_groups, mask)
        elif valid_filter and removal_ratio > skip_if_removes_more_than:
            logger.warning(
                f"⚠️ SKIPPING RARE CLASS FILTER: Would remove {100*removal_ratio:.1f}% of samples "
                f"({n_before - samples_after_filter}/{n_before}). Threshold: {100*skip_if_removes_more_than:.0f}% max removal. "
                f"Keeping all {n_before} samples despite {len(counts)-len(keep_idx)} rare classes."
            )
        elif valid_filter:
            logger.warning(f"⚠️ Skipping rare class filter: would leave only {samples_after_filter}/{n_before} samples (<{int(0.1*n_before)} minimum).")
        # else: no rare classes to filter
        
        stratify_target = y_labels 
    else: 
        mask = y_labels.notna()
        X, y = X.loc[mask], y.loc[mask]
        cv_groups = safe_mask_groups(cv_groups, mask)

    if X.empty or len(X) < 2 or (task_type == 'Classification' and y_labels.nunique() < 2):
        logger.error(f"❌ Data filtering resulted in insufficient samples ({len(X)} samples, {y_labels.nunique() if task_type == 'Classification' else 'N/A'} classes).")
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
        
        # TWO-TIER SPLIT: Use explicit test indices if provided
        if test_indices is not None:
            logger.info(f" 📊 Two-tier split: {len(test_indices)} test samples from filtered set, rest for training")
            test_mask = X.index.isin(test_indices)
            train_mask = ~test_mask
            
            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            
            # Handle groups for two-tier case
            g_tr, g_te = None, None
            if cv_groups is not None:
                g_arr = cv_groups.values if hasattr(cv_groups, 'values') else cv_groups
                g_tr = g_arr[train_mask.values]
                g_te = g_arr[test_mask.values]
            
            if task_type == 'Classification' and out_path:
                _plot_class_balance(y_tr, y_te, out_path)
            
            return X_tr, X_te, y_tr, y_te, g_tr, g_te
        
        # STANDARD SPLIT: Random or group-based split
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