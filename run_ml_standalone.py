import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StandaloneML")

# ==========================================
# CONFIGURATION (Hardcoded)
# ==========================================
# FILE_PATH = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_5/merged_samples.h5ad"
FILE_PATH = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_5/merged_samples.h5ad" # Update this path if needed
TARGET_COL = "facility_match"
TAX_LEVEL = "Genus"
MIN_SAMPLES_PER_CLASS = 10

# ==========================================
# 1. AGGREGATION LOGIC (The Fixed Version)
# ==========================================
def aggregate_to_level(adata, level='Genus'):
    """
    Aggregates counts to a taxonomic level using sparse matrix math.
    INCLUDES THE CRITICAL FIX FOR WHITESPACE STRIPPING.
    """
    logger.info(f"--- Aggregating to {level} ---")
    
    if level not in adata.var.columns:
        logger.error(f"Column '{level}' not found in .var")
        return None

    # 1. CLEANING (The Fix)
    # Convert to string, strip whitespace (handles ' g__X'), unify garbage
    groups = adata.var[level].astype(str).str.strip()
    groups = groups.replace(['nan', 'NaN', 'None', '', '<NA>', 'NoneType'], 'Unassigned')
    groups = groups.fillna('Unassigned')
    
    unique_groups = sorted(groups.unique())
    logger.info(f"Found {len(unique_groups)} unique groups after cleaning.")

    # 2. CREATE GROUPER MATRIX
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    col_indices = [group_dict[g] for g in groups]
    row_indices = range(len(groups))
    
    grouper = sparse.csr_matrix(
        (np.ones(len(groups)), (row_indices, col_indices)),
        shape=(len(groups), len(unique_groups))
    )
    
    # 3. AGGREGATE
    new_X = adata.X @ grouper
    
    # 4. CREATE NEW ANNDATA
    new_var = pd.DataFrame(index=unique_groups)
    adata_new = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)
    
    # 5. FILTER UNASSIGNED
    if 'Unassigned' in adata_new.var_names:
        if len(adata_new.var_names) > 1:
            logger.info("Filtering 'Unassigned' taxa...")
            adata_new = adata_new[:, adata_new.var_names != 'Unassigned'].copy()
        else:
            logger.warning("⚠️ Only 'Unassigned' exists! Keeping it to prevent empty dataset.")

    return adata_new

# ==========================================
# 2. TRANSFORMATION (CLR)
# ==========================================
def clr_transform(adata, pseudocount=1.0):
    logger.info("Applying CLR transform...")
    try:
        adata_clr = adata.copy()
        if sparse.issparse(adata_clr.X):
            adata_clr.X = adata_clr.X.toarray()
        
        mat = adata_clr.X + pseudocount
        gmeans = np.exp(np.log(mat).mean(axis=1, keepdims=True))
        adata_clr.X = np.log(mat / gmeans)
        return adata_clr
    except Exception as e:
        logger.error(f"CLR failed: {e}")
        return adata

# ==========================================
# 3. MACHINE LEARNING
# ==========================================
def run_ml(adata, target_col):
    logger.info(f"--- Running ML for Target: {target_col} ---")
    
    # 1. Prepare Data
    if target_col not in adata.obs.columns:
        logger.error(f"Target '{target_col}' not found in metadata!")
        return

    # Filter Missing Targets
    valid_obs = adata.obs.dropna(subset=[target_col])
    X = adata[valid_obs.index].X
    y = valid_obs[target_col]
    
    # Encode Target
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    classes = le.classes_
    logger.info(f"Target Classes: {classes}")

    # Check Sample Counts
    counts = pd.Series(y_enc).value_counts()
    if counts.min() < MIN_SAMPLES_PER_CLASS:
        logger.warning(f"⚠️ Classes too small for ML: {counts.to_dict()}. Skipping.")
        return

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, stratify=y_enc, random_state=42)
    
    # 3. Model Selection (CatBoost -> Random Forest fallback)
    try:
        from catboost import CatBoostClassifier
        logger.info("🚀 Training CatBoost Classifier...")
        model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0, allow_writing_files=False)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        logger.info("⚠️ CatBoost not found. Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4. Train & Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 5. Evaluate
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=classes))
    
    # 6. Feature Importance (Top 5)
    if hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
        indices = np.argsort(imps)[::-1][:5]
        print("\n🏆 Top 5 Features (Genera):")
        for i in indices:
            print(f"   - {adata.var_names[i]}: {imps[i]:.4f}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not Path(FILE_PATH).exists():
        # Fallback to current dir if hardcoded path is wrong
        FILE_PATH = "merged_samples.h5ad" 
        
    logger.info(f"Loading data from {FILE_PATH}...")
    try:
        adata = sc.read_h5ad(FILE_PATH)
        logger.info(f"Loaded: {adata.shape}")
        
        # 1. Aggregate
        adata_agg = aggregate_to_level(adata, TAX_LEVEL)
        logger.info(f"Aggregated Shape: {adata_agg.shape}")
        
        # 2. Transform
        adata_clr = clr_transform(adata_agg)
        
        # 3. ML
        run_ml(adata_clr, TARGET_COL)
        
        logger.info("🎉 Workflow Finished Successfully.")
        
    except Exception as e:
        logger.critical(f"Script Failed: {e}", exc_info=True)
