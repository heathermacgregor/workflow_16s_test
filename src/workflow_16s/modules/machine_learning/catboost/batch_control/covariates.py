# workflow_16s/modules/machine_learning/catboost/batch_control/covariates.py

from typing import Any, Dict, List, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")

@with_logger
def prepare_batch_covariates(
    adata: ad.AnnData, batch_columns: List[str], sample_indices: pd.Index,
    model_algorithm: str = 'rf', one_hot_encode: bool = True, 
    max_categories: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transforms raw metadata into a feature matrix. 
    Preserves raw categorical structures if the target algorithm is CatBoost.
    """
    batch_df_list = []
    metadata = {'dropped': [], 'encoded': {}, 'numeric': [], 'categorical': []}
    
    # We'll also track which columns remain categorical for CatBoost
    catboost_cat_features = []
    
    for col in batch_columns:
        actual_col = col
        
        # Hierarchical Fallback
        if col == 'batch_original' and col not in adata.obs.columns and 'study_accession' in adata.obs.columns:
            actual_col = 'study_accession'
            
        if actual_col not in adata.obs.columns:
            metadata['dropped'].append({'column': actual_col, 'reason': 'not_found'})
            continue

        col_data = adata.obs.loc[sample_indices, actual_col].copy()
        
        # Sparsity Check
        if col_data.empty:
            continue
            
        missing_pct = col_data.isna().sum() / len(col_data)
        if missing_pct > 0.5:
            metadata['dropped'].append({'column': actual_col, 'reason': 'too_many_missing', 'missing_pct': missing_pct})
            continue
        
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numeric:
            col_data_clean = col_data.fillna(col_data.median())
            batch_df_list.append(pd.DataFrame({actual_col: col_data_clean}, index=sample_indices))
            metadata['numeric'].append(actual_col)
            
        else:
            n_categories = col_data.nunique()
            if n_categories > max_categories:
                metadata['dropped'].append({'column': actual_col, 'reason': 'too_many_categories', 'n_categories': n_categories})
                continue
            
            # Clean categories
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                if 'Unknown' not in col_data.cat.categories:
                    col_data = col_data.cat.add_categories('Unknown')
                col_data_clean = col_data.fillna('Unknown')
            else:
                col_data_clean = col_data.fillna('Unknown').astype(str)
            
            # ---------------------------------------------------------
            # ALGORITHM-AWARE ENCODING LOGIC
            # ---------------------------------------------------------
            if model_algorithm.lower() == 'catboost':
                # CatBoost wants strings or pandas categoricals directly
                df_cat = pd.DataFrame({actual_col: col_data_clean}, index=sample_indices)
                batch_df_list.append(df_cat)
                catboost_cat_features.append(actual_col)
                metadata['encoded'][actual_col] = {'method': 'native_catboost', 'n_categories': n_categories}
                
            elif one_hot_encode:
                # Standard OHE for Random Forest / XGBoost
                dummies = pd.get_dummies(col_data_clean, prefix=actual_col, drop_first=True)
                batch_df_list.append(dummies)
                metadata['encoded'][actual_col] = {'method': 'one_hot', 'n_categories': n_categories, 'n_features': len(dummies.columns)}
                
            else:
                # Label Encoding fallback
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                encoded = le.fit_transform(col_data_clean.astype(str))
                batch_df_list.append(pd.DataFrame({actual_col: encoded}, index=sample_indices))
                metadata['encoded'][actual_col] = {'method': 'label', 'n_categories': n_categories}
            
            metadata['categorical'].append(actual_col)
    
    if not batch_df_list:
        return pd.DataFrame(index=sample_indices), metadata
    
    batch_df = pd.concat(batch_df_list, axis=1)
    
    # Store the exact column names that CatBoost needs to treat as categorical
    metadata['catboost_cat_features'] = catboost_cat_features
    
    return batch_df, metadata

@with_logger
def calculate_batch_importance(
    batch_df: pd.DataFrame, y: pd.Series, task_type: str = 'classification'
) -> pd.DataFrame:
    """
    Calculates the 'Technical Driver Score' for each metadata feature.
    
    Identifies which technical variables are most predictive of the target, 
    serving as the foundation for the Metadata Risk Profile.
    
    Parameters
    ----------
    batch_df : pd.DataFrame
        DataFrame of batch covariates (features).
    y : pd.Series
        Target variable.    
    task_type : str
        'classification' or 'regression'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores.
    """
    if batch_df.empty:
        logger.warning("Empty batch DataFrame passed to calculate_batch_importance")
        return pd.DataFrame(columns=['feature', 'importance'])

    # Factorial logic for task type
    if task_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Fit using exclusively batch features
    model.fit(batch_df.fillna(0), y)

    importance_df = pd.DataFrame({
        'feature': batch_df.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Log specific warnings for high-risk technical drivers
    top_driver = importance_df.iloc[0] if not importance_df.empty else None
    if top_driver is not None and top_driver['importance'] > 0.3:
        logger.warning(f"MAJOR TECHNICAL DRIVER: '{top_driver['feature']}' explains {top_driver['importance']:.1%} of target variance.")

    return importance_df
