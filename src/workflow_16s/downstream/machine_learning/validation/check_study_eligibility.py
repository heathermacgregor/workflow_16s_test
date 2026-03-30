# src/workflow_16s/downstream/machine_learning/validation/check_study_eligibility.py

import pandas as pd
import numpy as np

from typing import Any, List, Tuple

from workflow_16s.utils.logger import get_logger


class StudyEligibilityManager:
    """
    Manages the 'Pre-flight' audit of 16S studies to prevent model artifacts.
    
    MODES:
    - 'audit_only': Print report but do not filter data.
    - 'filter': Subset data to passing studies before analysis.
    - 'dual_path': Branch the workflow into 'Raw' and 'Filtered' runs.
    """
    
    def __init__(self, adata, target_col: str, min_n: int = 10):
        """
        Initialize StudyEligibilityManager.
        
        Args:
            adata: AnnData object with study/batch information
            target_col: Column name for the target variable
            min_n: Minimum samples per study (default 10, reduced from 15 to preserve training data)
        """
        self.adata = adata
        self.target_col = target_col
        self.min_n = min_n
        self.eligibility_df = pd.DataFrame()

    def diagnose_studies(self) -> pd.DataFrame:
        """
        Audits studies for sample volume, target variance, and class diversity.
        """
        logger = get_logger("workflow_16s")
        meta = self.adata.obs
        if self.target_col not in meta.columns:
            logger.error(f"❌ Target '{self.target_col}' not found in metadata.")
            return pd.DataFrame()

        # Identify target type for variance checks
        y = meta[self.target_col]
        is_numeric = pd.api.types.is_numeric_dtype(y)
        
        # Try multiple common batch column names
        possible_batch_cols = ['batch_original', 'study_accession', 'Project', 'dataset', 'study', 'batch', 'project_id', 'study_id']
        batch_col = None
        for col in possible_batch_cols:
            if col in meta.columns:
                batch_col = col
                break
        
        if batch_col is None:
            logger.warning(f"⚠️  No batch column found among {possible_batch_cols}. Using synthetic 'cohort' column.")
            batches = pd.Series(['cohort']*len(meta), index=meta.index)
        else:
            batches = meta[batch_col]
        
        results = []
        for batch in batches.unique():
            y_sub = y[batches == batch].dropna()
            n_samples = len(y_sub)
            
            status = "✅ PASS"
            reason = ""
            
            # 1. Volume Check
            if n_samples < self.min_n:
                status = "❌ FAIL"
                reason = f"Small N ({n_samples} < {self.min_n})"
            
            # 2. Variance/Diversity Check
            elif is_numeric:
                # For Regression: Must have a gradient
                std_dev = y_sub.astype(float).std()
                if std_dev == 0 or np.isnan(std_dev):
                    status = "❌ FAIL"
                    reason = "Zero Variance (Constant)"
            else:
                # For Classification: Must have at least two classes
                if y_sub.nunique() < 2:
                    status = "❌ FAIL"
                    reason = "Single Class (No Contrast)"
                
            results.append({
                'Study': batch, 
                'N': n_samples, 
                'Status': status, 
                'Reason': reason
            })
            
        self.eligibility_df = pd.DataFrame(results)
        
        # Format logging output
        logger.info(f"\n{'='*50}\nELIGIBILITY REPORT: {self.target_col}\n{'='*50}"
                    f"\n{self.eligibility_df.to_string(index=False)}")
        return self.eligibility_df

    def get_filtered_adata(self):
        """Returns a subsetted AnnData containing only passing studies."""
        if self.eligibility_df.empty: 
            self.diagnose_studies()
            
        passing = self.eligibility_df[self.eligibility_df['Status'] == "✅ PASS"]['Study'].tolist()
        
        # Try multiple common batch column names
        possible_batch_cols = ['batch_original', 'study_accession', 'Project', 'dataset', 'study', 'batch', 'project_id', 'study_id']
        batch_col = None
        for col in possible_batch_cols:
            if col in self.adata.obs.columns:
                batch_col = col
                break
        
        if batch_col is None:
            logger = get_logger("workflow_16s")
            logger.warning(f"⚠️  No batch column found among {possible_batch_cols}. Using all samples.")
            return self.adata.copy()
        
        return self.adata[self.adata.obs[batch_col].isin(passing)].copy()

    def get_all_adata(self):
        """Returns all data (both passing and failing studies). Used for training."""
        return self.adata.copy()

    def get_two_tier_split(self):
        """
        Returns a tuple of (train_adata, test_adata) for two-tier eligibility filtering.
        
        Train data: All studies (both passing and failing) for maximum training power
        Test data: Only passing studies for unbiased evaluation
        
        Returns:
            Tuple[AnnData, AnnData]: (train_adata with all studies, test_adata with only PASS studies)
        """
        if self.eligibility_df.empty:
            self.diagnose_studies()
        
        train_adata = self.get_all_adata()
        test_adata = self.get_filtered_adata()
        
        logger = get_logger("workflow_16s")
        logger.info(f" 📊 Two-tier split for '{self.target_col}':")
        logger.info(f"    Train (all studies):  {len(train_adata)} samples")
        logger.info(f"    Test (PASS only):    {len(test_adata)} samples")
        
        return train_adata, test_adata

# --- INTEGRATED ORCHESTRATOR LOGIC ---

def run_ml_eligibility_workflow(
    workflow: Any, 
    target_col: str, 
    mode: str = 'both'
) -> List[Tuple[str, Any]]:
    """
    Executes the branching logic for the ML matrix.
    
    Args:
        mode: 'raw' (no filter), 'filtered' (prune studies), or 'both' (parallel runs).
    """
    manager = StudyEligibilityManager(workflow.adata, target_col=target_col)
    manager.diagnose_studies()
    
    analysis_queue = []
    
    if mode in ['raw', 'both']:
        analysis_queue.append(('ML_Raw', workflow.adata))
        
    if mode in ['filtered', 'both']:
        filtered_adata = manager.get_filtered_adata()
        if len(filtered_adata) > 0:
            analysis_queue.append(('ML_Filtered', filtered_adata))
        else:
            logger = get_logger("workflow_16s")
            logger.warning(f"⚠️ No studies passed eligibility for {target_col}. Filtered run skipped.")
        
    return analysis_queue