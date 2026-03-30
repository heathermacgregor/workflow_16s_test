# src/workflow_16s/downstream/machine_learning/batch_control/covariates.py

import anndata as ad
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from workflow_16s.utils.logger import get_logger


def prepare_batch_covariates(
    adata: ad.AnnData,
    batch_columns: List[str],
    sample_indices: pd.Index,
    one_hot_encode: bool = True,
    max_categories: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transforms raw metadata into a numeric feature matrix for batch-aware ML.
    
    This function handles fallback logic (e.g., study_accession), imputation, 
    and encoding to ensure metadata can be processed by tree-based models.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix with metadata in .obs.
    batch_columns : List[str]
        List of batch variable column names in adata.obs.
    sample_indices : pd.Index
        Indices of samples to consider for batch covariate preparation.
    one_hot_encode : bool
        Whether to one-hot encode categorical variables.
    max_categories : int
        Maximum number of categories allowed for categorical variables.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        DataFrame of batch covariates and metadata about the processing steps.
    """
    logger = get_logger("workflow_16s")
    batch_df_list = []
    metadata = {'dropped': [], 'encoded': {}, 'numeric': [], 'categorical': []}
    
    for col in batch_columns:
        actual_col = col
        # Hierarchical Fallback
        if col == 'batch_original' and col not in adata.obs.columns and 'study_accession' in adata.obs.columns:
            logger.info("'batch_original' not found, falling back to 'study_accession'.")
            actual_col = 'study_accession'
            
        if actual_col not in adata.obs.columns:
            logger.warning(f"Batch column '{actual_col}' not found, skipping")
            metadata['dropped'].append({'column': actual_col, 'reason': 'not_found'})
            continue

        col_data = adata.obs.loc[sample_indices, actual_col].copy()
        
        # Sparsity Check
        missing_pct = col_data.isna().sum() / len(col_data)
        if missing_pct > 0.5:
            logger.warning(f"Batch column '{actual_col}' has {missing_pct:.1%} missing values, skipping")
            metadata['dropped'].append({'column': actual_col, 'reason': 'too_many_missing', 'missing_pct': missing_pct})
            continue
        
        # Type-specific processing
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numeric:
            col_data_clean = col_data.fillna(col_data.median())
            batch_df_list.append(pd.DataFrame({actual_col: col_data_clean}, index=sample_indices))
            metadata['numeric'].append(actual_col)
        else:
            n_categories = col_data.nunique()
            if n_categories > max_categories:
                logger.warning(f"Batch column '{actual_col}' has {n_categories} categories, skipping")
                metadata['dropped'].append({'column': actual_col, 'reason': 'too_many_categories', 'n_categories': n_categories})
                continue
            
            # Category handling
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                if 'Unknown' not in col_data.cat.categories:
                    col_data = col_data.cat.add_categories('Unknown')
                col_data_clean = col_data.fillna('Unknown')
            else:
                col_data_clean = col_data.fillna('Unknown').astype(str)
            
            if one_hot_encode:
                dummies = pd.get_dummies(col_data_clean, prefix=actual_col, drop_first=True)
                batch_df_list.append(dummies)
                metadata['encoded'][actual_col] = {'method': 'one_hot', 'n_categories': n_categories, 'n_features': len(dummies.columns)}
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                encoded = le.fit_transform(col_data_clean.astype(str))
                batch_df_list.append(pd.DataFrame({actual_col: encoded}, index=sample_indices))
                metadata['encoded'][actual_col] = {'method': 'label', 'n_categories': n_categories}
            
            metadata['categorical'].append(actual_col)
    
    if not batch_df_list:
        logger.warning("No valid batch covariates prepared")
        return pd.DataFrame(index=sample_indices), metadata
    
    batch_df = pd.concat(batch_df_list, axis=1)
    logger.info(f"Prepared {len(batch_df.columns)} batch features from {len(metadata['numeric']) + len(metadata['categorical'])} columns")
    
    return batch_df, metadata

def calculate_batch_importance(
    batch_df: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'classification'
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
    logger = get_logger("workflow_16s")
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
        logger.warning(f"🚨 MAJOR TECHNICAL DRIVER: '{top_driver['feature']}' explains {top_driver['importance']:.1%} of target variance.")

    return importance_df