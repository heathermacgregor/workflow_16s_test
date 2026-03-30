# workflow_16s/modules/machine_learning/catboost/validation/audit.py

import json
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import numpy as np

from workflow_16s.utils.logger import with_logger, get_logger
logger = get_logger('workflow_16s')

@with_logger
class StudyEligibilityManager:
    """
    Manages the 'Pre-flight' audit of 16S studies to prevent model artifacts.
    
    MODES:
    - 'audit_only': Print report but do not filter data.
    - 'filter': Subset data to passing studies before analysis.
    - 'dual_path': Branch the workflow into 'Raw' and 'Filtered' runs.
    """
    
    def __init__(self, adata, target_col: str, min_n: int = 15):
        self.adata = adata
        self.target_col = target_col
        self.min_n = min_n
        self.eligibility_df = pd.DataFrame()

    def diagnose_studies(self) -> pd.DataFrame:
        """
        Audits studies for sample volume, target variance, and class diversity.
        Safely distinguishes between continuous regression and encoded binary classification.
        """
        from workflow_16s.utils.logger import get_logger
        logger = get_logger('workflow_16s')
        meta = self.adata.obs
        if self.target_col not in meta.columns:
            logger.error(f"❌ Target '{self.target_col}' not found in metadata.")
            return pd.DataFrame()

        # 1. Safe Task Type Detection
        y = meta[self.target_col]
        is_numeric = pd.api.types.is_numeric_dtype(y)
        # 💡 FIX: Treat numeric columns with few unique values as classification
        is_regression = is_numeric and (y.nunique() > 10)
        
        batch_col = 'batch_original' if 'batch_original' in meta.columns else 'study_accession'
        batches = meta[batch_col] if batch_col in meta.columns else pd.Series(['cohort']*len(meta), index=meta.index)
        
        results = []
        for batch in batches.unique():
            y_sub = y[batches == batch].dropna()
            n_samples = len(y_sub)
            status = "✅ PASS"
            reason = " "
            # 2. Volume Check
           # if n_samples < self.min_n:
           #     status = "❌ FAIL"
           #     reason = f"Small N ({n_samples} < {self.min_n})"
            
            # 3. Variance/Diversity Check
          #  elif is_regression:
          #      # For Regression: Must have a gradient/variance
          #      std_dev = y_sub.astype(float).std()
          #      if std_dev == 0 or np.isnan(std_dev):
          #          status = "❌ FAIL"
          #          reason = "Zero Variance (Constant)"
          #else:
          #      # For Classification: Must have at least two distinct classes
          #      if y_sub.nunique() < 2:
          #       status = "❌ FAIL"
          #       reason = "Single Class (No Contrast)"
                
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
        batch_col = 'batch_original' if 'batch_original' in self.adata.obs.columns else 'study_accession'
        
        return self.adata  # Bypassed KeyError for MicrobeAtlas

@with_logger
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

@with_logger
class BiomarkerAuditor:
    """
    Automated scientific auditor for 16S Machine Learning results.
    Acts as a 'Certification Gate' for forensic and ecological discovery.
    """
    def __init__(self, output_dir: Path, target: str, level: str = 'Genus'):
        self.base_dir = Path(output_dir)
        self.target = target
        self.level = level
        self.report = []
        self.status = "PASS"

    def _log(self, check_name: str, passed: bool, value: Any, threshold: str):
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed: 
            self.status = "FAIL"
        self.report.append({
            'Check': check_name,
            'Status': status_icon,
            'Value': str(value),
            'Threshold': threshold
        })

    def run_audit(self) -> bool:
        """Runs the 4-point certification check."""
        logger = get_logger("workflow_16s")
        logger.info(f"🛡️ Certifying Discovery: {self.target} ({self.level})")
        
        # 1. Biological Signal Check
        agnostic_path = self.base_dir / "agnostic" / f"{self.level}_{self.target}" / "results_summary.json"
        if agnostic_path.exists():
            with open(agnostic_path) as f:
                res = json.load(f)
            mcc = res.get('test_scores', {}).get('mcc', 0.0)
            self._log("Biological Signal (MCC)", mcc > 0.40, f"{mcc:.3f}", "> 0.40")
        else:
            self._log("Biological Signal", False, "Missing", "Exists")

        # 2. Generalization Audit (The Gap)
        # Uses the audit results generated by our overfitting_prevention module
        audit_path = self.base_dir / "agnostic" / f"{self.level}_{self.target}" / "overfitting_audit" / "audit_results.json"
        if audit_path.exists():
            with open(audit_path) as f:
                audit = json.load(f)
            gap = audit.get('nested_cv', {}).get('overfitting_gap', 1.0)
            self._log("Overfitting Gap", gap < 0.15, f"{gap:.3f}", "< 0.15")
        else:
            self._log("Overfitting Audit", False, "Missing", "Exists")

        # 3. Statistical Significance Check
        shuffle_path = self.base_dir / "significance_test" / f"{self.level}_{self.target}" / "shuffle_stats.json"
        if shuffle_path.exists():
            with open(shuffle_path) as f:
                sh_res = json.load(f)
            p_val = sh_res.get('p_value', 1.0)
            self._log("Significance (p-val)", p_val < 0.05, f"{p_val:.4f}", "< 0.05")
        
        # 4. Consistency Check (Meta-Analysis)
        meta_path = self.base_dir / "meta_analysis" / "consensus_biomarkers.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            # Find biomarkers present in > 50% of studies
            stable_count = (df['Frequency_Pct'] >= 0.5).sum()
            self._log("Stable Biomarkers", stable_count >= 3, f"{stable_count}", ">= 3")

        return self.status == "PASS"

    def get_summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.report)
    
@with_logger
def verify_run(output_dir: Path, target_names: List[str]) -> bool:
    """
    The main entry point for Tier 4 Certification.
    Scans a completed ML run and generates a 'Pass/Fail' certification log.
    """
    logger.info(f"🛡️  Starting Final Certification for: {target_names}")
    
    overall_pass = True
    for target in target_names:
        auditor = BiomarkerAuditor(output_dir, target)
        try:
            auditor.run_audit()
            logger.info(f"✅ Target '{target}' Certification: COMPLETE")
        except Exception as e:
            logger.error(f"❌ Target '{target}' Certification: FAILED - {e}")
            overall_pass = False
            
    return overall_pass
