"""Measured Metal Validation

Validate functional predictions against measured metal concentrations in samples.
Correlates KO-predicted metal resistance/uptake genes with experimental measurements.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import anndata
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass

logger = logging.getLogger("workflow_16s")


@dataclass
class MetalValidationResult:
    """Results from metal validation"""
    correlations: pd.DataFrame  # KO × metal correlations
    pvalues: pd.DataFrame  # Corresponding p-values
    significant_pairs: List[Tuple[str, str]]  # (KO, metal) pairs with p < 0.05


class MeasuredMetalValidator:
    """Cross-validate functional predictions with measured metals"""
    
    def __init__(self, logger_obj=None):
        """Initialize validator."""
        self.logger_obj = logger_obj or logger
        
        # Curated KO-metal associations
        self.ko_metal_map = {
            'metal_resistance': {
                'Cu': ['K01594', 'K01595', 'K01596'],  # cupA oxidase
                'Zn': ['K07235', 'K07236'],  # zinc efflux
                'Cd': ['K01307'],  # cadmium resistance
                'Hg': ['K01594'],  # mercury resistance
                'Pb': ['K01307', 'K01308'],  # lead resistance
                'Mn': ['K04758'],  # manganese oxidase
                'Fe': ['K03711', 'K03712'],  # iron transport
            },
            'metal_uptake': {
                'Fe': ['K02014', 'K02015', 'K02016'],  # TonB-dependent
                'Mo': ['K02020', 'K02021'],  # molybdenum ABC transporter
                'Ni': ['K02033', 'K02034'],  # nickel ABC transporter
                'Cu': ['K02015', 'K02016'],  # copper ABC transporter
            }
        }
    
    def load_metal_measurements(
        self,
        data_file: Path,
        sample_col: str = 'sample_id',
        metal_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load measured metal concentrations.
        
        Args:
            data_file: Path to measured metals file (CSV/TSV)
            sample_col: Column name for sample IDs
            metal_cols: Columns to use as metal measurements (all others used if None)
            
        Returns:
            DataFrame with sample_id as index, metals as columns
        """
        self.logger_obj.info(f"📊 Loading metal measurements from {data_file}")
        
        # Detect format
        ext = data_file.suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(data_file)
        elif ext in ['.tsv', '.txt']:
            df = pd.read_csv(data_file, sep='\t')
        else:
            df = pd.read_csv(data_file, sep='\t')
        
        # Set sample ID as index
        df = df.set_index(sample_col)
        
        # Filter metals
        if metal_cols is None:
            # Auto-detect numeric columns
            metal_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_metals = df[metal_cols].astype(float)
        
        self.logger_obj.info(f"✓ Loaded {len(df_metals)} samples × {len(metal_cols)} metals")
        
        return df_metals
    
    def predict_metal_phenotypes(
        self,
        adata: anndata.AnnData,
        ko_matrix: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict metal resistance phenotypes from KO profiles.
        
        For each sample, aggregates KO abundance into metal-associated scores.
        
        Args:
            adata: AnnData object
            ko_matrix: Optional pre-computed KO matrix
            
        Returns:
            DataFrame with predicted metal phenotype scores
        """
        if ko_matrix is None:
            if 'KO_CLR' in adata.obsm:
                ko_matrix = adata.obsm['KO_CLR']
            elif 'KO_counts' in adata.obsm:
                ko_matrix = adata.obsm['KO_counts']
            else:
                raise ValueError("KO matrix not found in adata.obsm")
        
        self.logger_obj.info("🔬 Predicting metal phenotypes from KO profiles...")
        
        predictions = {}
        
        # Map KOs to columns
        ko_names = adata.varm.get('KO_list', [f'KO_{i}' for i in range(ko_matrix.shape[1])])
        ko_idx = {ko: i for i, ko in enumerate(ko_names)}
        
        # Score each metal
        for metal_group, metal_dict in self.ko_metal_map.items():
            for metal, ko_list in metal_dict.items():
                # Find indices of relevant KOs
                ko_indices = [ko_idx[ko] for ko in ko_list if ko in ko_idx]
                
                if not ko_indices:
                    self.logger_obj.debug(f"No KOs found for {metal}")
                    continue
                
                # Aggregate abundance
                metal_score = ko_matrix[:, ko_indices].mean(axis=1)
                predictions[f'{metal_group}_{metal}'] = metal_score
        
        result = pd.DataFrame(predictions, index=adata.obs_names)
        
        self.logger_obj.info(f"✓ Predicted {len(result.columns)} metal phenotype scores")
        
        return result
    
    def validate(
        self,
        adata: anndata.AnnData,
        measured_metals: pd.DataFrame,
        ko_matrix: Optional[np.ndarray] = None,
        correlation_method: str = 'spearman',
        significance_level: float = 0.05,
    ) -> MetalValidationResult:
        """
        Correlate predicted metal phenotypes with measurements.
        
        Args:
            adata: AnnData object
            measured_metals: DataFrame with measured metal concentrations
            ko_matrix: Optional KO matrix
            correlation_method: "pearson" or "spearman"
            significance_level: p-value threshold
            
        Returns:
            MetalValidationResult with correlation table
        """
        self.logger_obj.info("🔍 Validating predictions against measurements...")
        
        # Predict phenotypes
        predictions = self.predict_metal_phenotypes(adata, ko_matrix)
        
        # Match samples
        common_samples = predictions.index.intersection(measured_metals.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between predictions and measurements")
        
        predictions = predictions.loc[common_samples]
        measured_metals = measured_metals.loc[common_samples]
        
        # Compute correlations
        correlations = pd.DataFrame(
            np.zeros((predictions.shape[1], measured_metals.shape[1])),
            index=predictions.columns,
            columns=measured_metals.columns
        )
        pvalues = correlations.copy()
        
        for pred_col in predictions.columns:
            for metal_col in measured_metals.columns:
                pred_vals = predictions[pred_col].values
                metal_vals = measured_metals[metal_col].values
                
                # Remove NaN pairs
                valid = ~(np.isnan(pred_vals) | np.isnan(metal_vals))
                if valid.sum() < 3:  # Need at least 3 points
                    continue
                
                pred_vals = pred_vals[valid]
                metal_vals = metal_vals[valid]
                
                if correlation_method == 'pearson':
                    corr, pval = pearsonr(pred_vals, metal_vals)
                else:
                    corr, pval = spearmanr(pred_vals, metal_vals)
                
                correlations.loc[pred_col, metal_col] = corr
                pvalues.loc[pred_col, metal_col] = pval
        
        # Find significant pairs
        significant_pairs = []
        for pred_col in correlations.index:
            for metal_col in correlations.columns:
                pval = pvalues.loc[pred_col, metal_col]
                if pval < significance_level:
                    corr = correlations.loc[pred_col, metal_col]
                    significant_pairs.append((pred_col, metal_col, corr, pval))
        
        # Sort by correlation strength
        significant_pairs = sorted(significant_pairs, key=lambda x: abs(x[2]), reverse=True)
        
        self.logger_obj.info(
            f"✅ Found {len(significant_pairs)} significant KO-metal associations "
            f"(p < {significance_level})"
        )
        
        return MetalValidationResult(
            correlations=correlations,
            pvalues=pvalues,
            significant_pairs=[(p[0], p[1]) for p in significant_pairs]
        )
