"""
Results validation utilities for workflow quality checks.

Provides validation checks for analysis results to catch common issues.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger("workflow_16s")


class ResultsValidator:
    """Validate analysis results for common issues."""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
        
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        name: str,
        min_rows: int = 1,
        required_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Validate a results dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate
        name : str
            Name of the dataframe for reporting
        min_rows : int
            Minimum required rows
        required_columns : list, optional
            List of required column names
            
        Returns
        -------
        bool
            True if validation passed
        """
        if df is None:
            self.errors.append(f"{name}: DataFrame is None")
            return False
            
        if df.empty:
            self.warnings.append(f"{name}: DataFrame is empty")
            return False
            
        if len(df) < min_rows:
            self.warnings.append(
                f"{name}: Only {len(df)} rows (expected >={min_rows})"
            )
            
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                self.errors.append(
                    f"{name}: Missing columns: {missing}"
                )
                return False
                
        # Check for all-NaN columns
        nan_cols = df.columns[df.isna().all()].tolist()
        if nan_cols:
            self.warnings.append(
                f"{name}: All-NaN columns: {nan_cols}"
            )
            
        return True
    
    def validate_ml_results(
        self,
        results: Dict[str, Any],
        task_type: str = "classification"
    ) -> bool:
        """
        Validate machine learning results.
        
        Parameters
        ----------
        results : dict
            ML results dictionary
        task_type : str
            'classification' or 'regression'
            
        Returns
        -------
        bool
            True if validation passed
        """
        if not results:
            self.warnings.append("ML results empty")
            return False
            
        # Check for convergence indicators
        if task_type == "classification":
            if 'oob_score' in results:
                oob = results['oob_score']
                if oob < 0.3:
                    self.warnings.append(
                        f"Low OOB score ({oob:.3f}) - model may not be reliable"
                    )
                    
        elif task_type == "regression":
            if 'r2_score' in results:
                r2 = results['r2_score']
                if r2 < 0:
                    self.warnings.append(
                        f"Negative R² ({r2:.3f}) - model performs worse than mean"
                    )
                    
        # Check feature importance
        if 'feature_importance' in results:
            importance = results['feature_importance']
            if isinstance(importance, pd.DataFrame):
                if importance.empty:
                    self.warnings.append("Feature importance is empty")
                elif importance['importance'].max() == importance['importance'].min():
                    self.warnings.append("All features have same importance")
                    
        return True
    
    def validate_diversity_results(
        self,
        alpha_df: Optional[pd.DataFrame] = None,
        beta_df: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Validate diversity analysis results.
        
        Parameters
        ----------
        alpha_df : pd.DataFrame, optional
            Alpha diversity results
        beta_df : pd.DataFrame, optional
            Beta diversity / ordination results
            
        Returns
        -------
        bool
            True if validation passed
        """
        valid = True
        
        if alpha_df is not None:
            valid &= self.validate_dataframe(
                alpha_df,
                "Alpha Diversity",
                min_rows=10
            )
            
            # Check for negative values
            numeric_cols = alpha_df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if (alpha_df[col] < 0).any():
                    self.warnings.append(
                        f"Alpha Diversity: Negative values in {col}"
                    )
                    
        if beta_df is not None:
            valid &= self.validate_dataframe(
                beta_df,
                "Beta Diversity",
                min_rows=10
            )
            
        return valid
    
    def validate_statistical_results(
        self,
        results_df: pd.DataFrame,
        test_name: str
    ) -> bool:
        """
        Validate statistical test results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Statistical test results
        test_name : str
            Name of the test
            
        Returns
        -------
        bool
            True if validation passed
        """
        valid = self.validate_dataframe(
            results_df,
            f"{test_name} Results",
            min_rows=1
        )
        
        if 'p_value' in results_df.columns:
            # Check for invalid p-values
            invalid_pvals = (
                (results_df['p_value'] < 0) | 
                (results_df['p_value'] > 1)
            ).sum()
            
            if invalid_pvals > 0:
                self.errors.append(
                    f"{test_name}: {invalid_pvals} invalid p-values"
                )
                valid = False
                
            # Check if any significant results
            if 'q_value' in results_df.columns:
                sig_count = (results_df['q_value'] < 0.05).sum()
                if sig_count == 0:
                    self.warnings.append(
                        f"{test_name}: No significant results (q < 0.05)"
                    )
                    
        return valid
    
    def get_summary(self) -> str:
        """Generate validation summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("VALIDATION SUMMARY")
        lines.append("=" * 80)
        
        if not self.errors and not self.warnings:
            lines.append("\n✅ All validations passed - no issues detected")
        else:
            if self.errors:
                lines.append(f"\n❌ ERRORS ({len(self.errors)}):")
                for error in self.errors:
                    lines.append(f"  • {error}")
                    
            if self.warnings:
                lines.append(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    lines.append(f"  • {warning}")
                    
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def save_report(self, output_path: Path):
        """Save validation report to file."""
        report = self.get_summary()
        output_path.write_text(report)
        logger.info(f"Validation report saved to: {output_path}")


# Global validator instance
_validator = ResultsValidator()


def get_validator() -> ResultsValidator:
    """Get the global results validator instance."""
    return _validator


def validate_results(
    alpha_df: Optional[pd.DataFrame] = None,
    beta_df: Optional[pd.DataFrame] = None,
    stats_results: Optional[Dict[str, pd.DataFrame]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> ResultsValidator:
    """
    Comprehensive results validation.
    
    Parameters
    ----------
    alpha_df : pd.DataFrame, optional
        Alpha diversity results
    beta_df : pd.DataFrame, optional
        Beta diversity results
    stats_results : dict, optional
        Dictionary of statistical test results
    ml_results : dict, optional
        Machine learning results
    output_dir : Path, optional
        Directory to save validation report
        
    Returns
    -------
    ResultsValidator
        Validator instance with results
    """
    validator = ResultsValidator()
    
    # Validate diversity
    if alpha_df is not None or beta_df is not None:
        validator.validate_diversity_results(alpha_df, beta_df)
    
    # Validate statistical tests
    if stats_results:
        for test_name, results in stats_results.items():
            if isinstance(results, pd.DataFrame):
                validator.validate_statistical_results(results, test_name)
    
    # Validate ML results
    if ml_results:
        for target, results in ml_results.items():
            if isinstance(results, dict):
                validator.validate_ml_results(results)
    
    # Generate report
    summary = validator.get_summary()
    logger.info("\n" + summary)
    
    if output_dir:
        report_path = output_dir / "validation_report.txt"
        validator.save_report(report_path)
    
    return validator
