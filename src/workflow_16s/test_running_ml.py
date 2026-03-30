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
# CONFIGURATION
# ==========================================
# Main Input File
FILE_PATH = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_5/merged_samples.h5ad"

# Where to save/load the aggregated file
CACHE_DIR = Path("/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_5/cache/")

TARGET_COL = "facility_match"
TAX_LEVEL = "Genus"
MIN_SAMPLES_PER_CLASS = 10

# ==========================================
# 1. AGGREGATION LOGIC (Cached & Robust)
# ==========================================
def get_aggregated_data(adata, level='Genus', cache_dir=None):
    """
    Tries to load aggregated data from cache. If missing, calculates and saves it.
    """
    # 1. Check Cache
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"taxa_agg_{level}.h5ad"
        
        if cache_path.exists():
            logger.info(f"⚡ Loading cached aggregation from: {cache_path}")
            try:
                return sc.read_h5ad(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recalculating.")

    # 2. Calculate if not cached
    logger.info(f"--- Aggregating to {level} (Fresh Calculation) ---")
    
    if level not in adata.var.columns:
        logger.error(f"Column '{level}' not found in .var")
        return None

    # A. CLEANING (Robust Fix)
    # Convert to string, strip whitespace (handles ' g__X'), unify garbage
    groups = adata.var[level].astype(str).str.strip()
    groups = groups.replace(['nan', 'NaN', 'None', '', '<NA>', 'NoneType'], 'Unassigned')
    groups = groups.fillna('Unassigned')
    
    unique_groups = sorted(groups.unique())
    logger.info(f"Found {len(unique_groups)} unique groups after cleaning.")

    # B. CREATE GROUPER MATRIX
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    col_indices = [group_dict[g] for g in groups]
    row_indices = range(len(groups))
    
    grouper = sparse.csr_matrix(
        (np.ones(len(groups)), (row_indices, col_indices)),
        shape=(len(groups), len(unique_groups))
    )
    
    # C. AGGREGATE
    new_X = adata.X @ grouper
    
    # D. CREATE NEW ANNDATA
    new_var = pd.DataFrame(index=unique_groups)
    adata_new = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)
    
    # E. FILTER UNASSIGNED
    if 'Unassigned' in adata_new.var_names:
        if len(adata_new.var_names) > 1:
            logger.info("Filtering 'Unassigned' taxa...")
            adata_new = adata_new[:, adata_new.var_names != 'Unassigned'].copy()
        else:
            logger.warning("⚠️ Only 'Unassigned' exists! Keeping it to prevent empty dataset.")

    # F. SAVE CACHE
    if cache_dir:
        try:
            # Critical: Ensure index is string to prevent HDF5 errors
            adata_new.var_names = adata_new.var_names.astype(str)
            adata_new.obs_names = adata_new.obs_names.astype(str)
            
            logger.info(f"💾 Saving aggregation cache to: {cache_path}")
            adata_new.write(cache_path)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

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
        logger.warning(f"File not found at {FILE_PATH}, trying local 'merged_samples.h5ad'")
        FILE_PATH = "merged_samples.h5ad" 
        
    logger.info(f"Loading data from {FILE_PATH}...")
    try:
        adata = sc.read_h5ad(FILE_PATH)
        
        # --- CRITICAL FIX FOR DUPLICATES ---
        logger.info("🔧 Ensuring unique sample IDs...")
        adata.obs_names_make_unique()
        
        logger.info(f"Loaded Raw Data: {adata.shape}")
        
        # 1. Aggregate (With Caching)
        adata_agg = get_aggregated_data(adata, level=TAX_LEVEL, cache_dir=CACHE_DIR)
        
        if adata_agg is not None:
            logger.info(f"Aggregated Shape: {adata_agg.shape}")
            
            # 2. Transform
            adata_clr = clr_transform(adata_agg)
            
            # 3. ML
            run_ml(adata_clr, TARGET_COL)
            
            logger.info("🎉 Workflow Finished Successfully.")
        
    except Exception as e:
        logger.critical(f"Script Failed: {e}", exc_info=True)