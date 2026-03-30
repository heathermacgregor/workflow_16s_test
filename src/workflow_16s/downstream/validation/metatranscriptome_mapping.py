"""Metatranscriptome Validation

Validate KO-based functional predictions against metatranscriptomic expression data.
Maps predicted KO profiles to actual transcription measurements.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import anndata
from scipy.stats import spearmanr
from dataclasses import dataclass

logger = logging.getLogger("workflow_16s")


@dataclass
class MetatranscriptomicResult:
    """Results from metatranscriptome validation"""
    expression_table: pd.DataFrame  # KO × expression values
    correlations: pd.DataFrame  # Predicted vs measured correlations
    explained_variance: Dict[str, float]  # Per-KO R² values


class MetatranscriptomeValidator:
    """Cross-validate KO predictions with gene expression"""
    
    def __init__(self, logger_obj=None):
        """Initialize validator."""
        self.logger_obj = logger_obj or logger
    
    def load_metatranscriptome(
        self,
        data_file: Path,
        ko_col: str = 'ko_id',
        expression_col: str = 'tpm',
        sample_col: str = 'sample_id',
    ) -> pd.DataFrame:
        """
        Load metatranscriptomic expression data.
        
        Expected format: Long table with sample, KO, TPM columns
        
        Args:
            data_file: Path to expression table (CSV/TSV)
            ko_col: Column name for KO IDs
            expression_col: Column name for expression values (TPM/RPKM/counts)
            sample_col: Column name for sample IDs
            
        Returns:
            DataFrame with samples × KOs and expression values
        """
        self.logger_obj.info(f"📊 Loading metatranscriptome data from {data_file}")
        
        ext = data_file.suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(data_file)
        else:
            df = pd.read_csv(data_file, sep='\t')
        
        # Pivot to wide format: samples × KOs
        expression_wide = df.pivot_table(
            index=sample_col,
            columns=ko_col,
            values=expression_col,
            fill_value=0
        )
        
        self.logger_obj.info(
            f"✓ Loaded {expression_wide.shape[0]} samples × {expression_wide.shape[1]} KOs"
        )
        
        return expression_wide
    
    def compare_abundance_expression(
        self,
        predicted_abundance: np.ndarray,
        measured_expression: pd.DataFrame,
        ko_names: List[str],
        sample_names: List[str],
        correlation_method: str = 'spearman',
    ) -> MetatranscriptomicResult:
        """
        Compare predicted KO abundances with measured expression.
        
        For each KO, correlates predicted abundance with measured transcription.
        
        Args:
            predicted_abundance: Sample × KO matrix (from functional profiling)
            measured_expression: Sample × KO expression matrix
            ko_names: List of KO identifiers
            sample_names: List of sample identifiers
            correlation_method: "pearson" or "spearman"
            
        Returns:
            MetatranscriptomicResult
        """
        self.logger_obj.info("🔬 Comparing KO predictions with expression...")
        
        # Match samples and KOs
        common_samples = list(set(sample_names) & set(measured_expression.index))
        common_kos = list(set(ko_names) & set(measured_expression.columns))
        
        if not common_samples or not common_kos:
            raise ValueError("No common samples or KOs between predictions and measurements")
        
        self.logger_obj.info(f"  Comparing {len(common_samples)} samples × {len(common_kos)} KOs")
        
        # Extract matrices
        sample_idx = [sample_names.index(s) for s in common_samples]
        ko_idx = [ko_names.index(k) for k in common_kos]
        
        predicted = predicted_abundance[np.ix_(sample_idx, ko_idx)]
        measured = measured_expression.loc[common_samples, common_kos].values
        
        # Log transform both
        # Add pseudocount for measured expression
        measured_log = np.log2(measured + 1)
        predicted_log = np.log2(predicted + 1)
        
        # Calculate correlations and R²
        correlations = []
        explained_var = {}
        
        for i, ko in enumerate(common_kos):
            pred_vals = predicted_log[:, i]
            expr_vals = measured_log[:, i]
            
            # Remove NaN pairs
            valid = ~(np.isnan(pred_vals) | np.isnan(expr_vals))
            if valid.sum() < 3:
                continue
            
            pred_vals = pred_vals[valid]
            expr_vals = expr_vals[valid]
            
            # Correlation
            if correlation_method == 'spearman':
                corr, pval = spearmanr(pred_vals, expr_vals)
            else:
                # Pearson
                from scipy.stats import pearsonr
                corr, pval = pearsonr(pred_vals, expr_vals)
            
            # Explained variance (R²)
            from sklearn.metrics import r2_score
            r2 = r2_score(expr_vals, pred_vals)
            
            correlations.append({
                'KO': ko,
                'correlation': corr,
                'pvalue': pval,
                'r_squared': r2,
            })
            explained_var[ko] = r2
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        # Summary statistics
        median_corr = corr_df['correlation'].median()
        n_significant = (corr_df['pvalue'] < 0.05).sum()
        mean_r2 = corr_df['r_squared'].mean()
        
        self.logger_obj.info(
            f"✅ Validation complete:\n"
            f"   Median correlation: {median_corr:.3f}\n"
            f"   Significant (p<0.05): {n_significant}/{len(corr_df)}\n"
            f"   Mean R²: {mean_r2:.3f}"
        )
        
        return MetatranscriptomicResult(
            expression_table=corr_df,
            correlations=corr_df[['KO', 'correlation']].set_index('KO'),
            explained_variance=explained_var,
        )
    
    def detect_expression_outliers(
        self,
        predicted: np.ndarray,
        measured: np.ndarray,
        threshold: float = 2.0,  # Standard deviations
    ) -> List[Tuple[int, int]]:
        """
        Find KOs with high residuals (predictions don't match expression).
        
        Args:
            predicted: Predicted KO abundances
            measured: Measured KO expression
            threshold: Deviation threshold
            
        Returns:
            List of (sample_idx, ko_idx) tuples with outliers
        """
        residuals = (predicted - measured) ** 2
        residuals_norm = (residuals - residuals.mean()) / residuals.std()
        
        outliers = np.argwhere(np.abs(residuals_norm) > threshold).tolist()
        
        self.logger_obj.debug(
            f"Detected {len(outliers)} expression outliers (>{threshold}σ)"
        )
        
        return outliers


def validate_with_metatranscriptome(
    adata: anndata.AnnData,
    metatranscriptome_file: Path,
    ko_matrix: Optional[np.ndarray] = None,
    logger_obj=None,
) -> MetatranscriptomicResult:
    """
    High-level API for metatranscriptome validation.
    
    Args:
        adata: AnnData with predicted KO abundances
        metatranscriptome_file: Path to expression data
        ko_matrix: Optional KO matrix (uses adata.obsm if None)
        logger_obj: Logger instance
        
    Returns:
        MetatranscriptomicResult
    """
    logger_obj = logger_obj or logger
    
    validator = MetatranscriptomeValidator(logger_obj)
    
    # Load expression data
    expression = validator.load_metatranscriptome(metatranscriptome_file)
    
    # Get predicted KO abundances
    if ko_matrix is None:
        if 'KO_CLR' in adata.obsm:
            ko_matrix = adata.obsm['KO_CLR']
        elif 'KO_counts' in adata.obsm:
            ko_matrix = adata.obsm['KO_counts']
        else:
            raise ValueError("KO matrix not found in adata.obsm")
    
    # Get KO names
    ko_names = adata.varm.get('KO_list', [f'KO_{i}' for i in range(ko_matrix.shape[1])])
    
    result = validator.compare_abundance_expression(
        ko_matrix,
        expression,
        ko_names,
        list(adata.obs_names),
    )
    
    return result
