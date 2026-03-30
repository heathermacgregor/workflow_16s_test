"""
Ecotype-Based Feature Engineering for ML Models

Extends the standard ASV feature matrix by incorporating ecotype stratification.
This allows models to learn from cryptic strain variants (ecotypes) within OTUs.

When enabled via config (use_ecotypes: true), creates two feature sets:
1. Standard ASV abundance features
2. Ecotype abundance features (abundance of each ecotype within each OTU)

Then combines them for model training, enabling the pipeline to distinguish
between functional patterns driven by OTU presence vs. ecotype stratification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


class EcotypeFeatureGenerator:
    """
    Generates ecotype-based features from detected ecotypes and OTU abundance table.
    """
    
    def __init__(self, min_ecotype_prevalence: float = 0.05):
        """
        Args:
            min_ecotype_prevalence: Minimum fraction of samples where ecotype must be detected
                                    to be included as a feature (prevents sparse noise)
        """
        self.min_ecotype_prevalence = min_ecotype_prevalence
        self.ecotype_features = None
        self.ecotype_mapping = {}  # Maps ecotype_id → (otu_id, ecotype_num)
    
    def generate_ecotype_features(
        self,
        adata: sc.AnnData,
        ecotype_profiles: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate ecotype-level features from detected stratification.
        
        Args:
            adata: AnnData object with sample × OTU abundance
            ecotype_profiles: Dict mapping OTU_ID → EcotypeProfile from ecotype_detection
            output_path: Optional path to save ecotype feature matrix
        
        Returns:
            pd.DataFrame with samples × ecotype_features
        """
        logger.info(f"Generating ecotype features from {len(ecotype_profiles)} stratified OTUs...")
        
        ecotype_features = []
        n_samples = adata.n_obs
        feature_names = []
        
        # For each OTU with detected ecotypes
        for otu_id, ecotype_profile in ecotype_profiles.items():
            if otu_id not in adata.var_names:
                logger.warning(f"OTU {otu_id} in ecotype_profiles but not in adata")
                continue
            
            # Get OTU abundance across samples
            otu_abundance = adata[:, otu_id].X
            if hasattr(otu_abundance, 'toarray'):
                otu_abundance = otu_abundance.toarray().ravel()
            else:
                otu_abundance = np.asarray(otu_abundance).ravel()
            
            # For each detected ecotype, create a feature based on:
            # - Sample's environmental/niche membership
            # - Expected ecotype abundance in that niche
            for ecotype in ecotype_profile.ecotypes:
                ecotype_id = f"{otu_id}_ecotype{ecotype.ecotype_id}"
                
                # Generate ecotype abundance estimate for each sample
                # (based on ecotype's niche breadth and specialization)
                ecotype_abundance = np.zeros(n_samples)
                
                # Scale by OTU presence and ecotype prevalence
                otu_present = otu_abundance > 0
                ecotype_abundance[otu_present] = (
                    otu_abundance[otu_present] * ecotype.prevalence * 
                    (1 - ecotype.niche_breadth)  # Specialists = more predictable distribution
                )
                
                # Only include if prevalence is sufficient
                n_nonzero = (ecotype_abundance > 0).sum()
                if n_nonzero / n_samples >= self.min_ecotype_prevalence:
                    ecotype_features.append(ecotype_abundance)
                    feature_names.append(ecotype_id)
                    self.ecotype_mapping[ecotype_id] = (otu_id, ecotype.ecotype_id)
                    logger.debug(f"  Added ecotype feature: {ecotype_id} (prevalence: {n_nonzero/n_samples:.1%})")
        
        # Combine into DataFrame
        if ecotype_features:
            self.ecotype_features = pd.DataFrame(
                np.column_stack(ecotype_features),
                index=adata.obs_names,
                columns=feature_names
            )
            logger.info(f"Generated {len(feature_names)} ecotype features ({len(ecotype_profiles)} OTUs)")
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.ecotype_features.to_csv(output_path)
                logger.info(f"✓ Ecotype features saved to {output_path}")
        else:
            logger.warning("No ecotype features generated (insufficient prevalence)")
            self.ecotype_features = pd.DataFrame(index=adata.obs_names)
        
        return self.ecotype_features
    
    def combine_with_asv_features(
        self,
        asv_features: pd.DataFrame,
        weight_asv: float = 0.7,
        weight_ecotype: float = 0.3
    ) -> pd.DataFrame:
        """
        Combine ASV and ecotype features with optional weighting.
        
        Args:
            asv_features: Original ASV abundance features
            weight_asv: Weight for ASV features in normalization (0-1)
            weight_ecotype: Weight for ecotype features in normalization (0-1)
        
        Returns:
            pd.DataFrame with combined features (samples × [ASV + ecotype])
        """
        if self.ecotype_features is None or self.ecotype_features.empty:
            logger.warning("No ecotype features available. Returning ASV features only.")
            return asv_features
        
        # Align indices
        common_samples = asv_features.index.intersection(self.ecotype_features.index)
        asv_aligned = asv_features.loc[common_samples]
        eco_aligned = self.ecotype_features.loc[common_samples]
        
        # Normalize separately, then weight
        asv_norm = asv_aligned / (asv_aligned.sum(axis=1, keepdims=True) + 1e-10)
        eco_norm = eco_aligned / (eco_aligned.sum(axis=1, keepdims=True) + 1e-10)
        
        # Simple concatenation - alternatively, could apply PCA reduction
        combined = pd.concat([asv_norm, eco_norm], axis=1)
        logger.info(f"Combined features: {combined.shape[1]} features ({asv_norm.shape[1]} ASV + {eco_norm.shape[1]} ecotype)")
        
        return combined
    
    def save_feature_annotation(self, output_path: Path):
        """
        Save a detailed mapping of ecotype features for interpretation.
        """
        if not self.ecotype_mapping:
            logger.warning("No ecotype mapping to save")
            return
        
        mapping_df = pd.DataFrame([
            {'ecotype_feature': k, 'otu_id': v[0], 'ecotype_num': v[1]}
            for k, v in self.ecotype_mapping.items()
        ])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(output_path, index=False)
        logger.info(f"✓ Ecotype feature mapping saved to {output_path}")


def create_ecotype_feature_set(
    adata: sc.AnnData,
    ecotype_profiles: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    High-level function to generate ecotype features from a configured pipeline.
    
    Args:
        adata: AnnData object with OTU × sample data
        ecotype_profiles: Dictionary of detected ecotypes
        config: Optional config dict with ecotype_params (e.g., min_prevalence)
        output_dir: Optional output directory for saving feature matrices
    
    Returns:
        Tuple of (asv_features, ecotype_features)
        - asv_features: Original ASV abundance matrix
        - ecotype_features: Generated ecotype abundance matrix (or None if not detected)
    """
    # Extract ASV features (standard abundance)
    asv_features = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    if not ecotype_profiles:
        logger.warning("No ecotype profiles provided. Returning ASV features only.")
        return asv_features, None
    
    # Generate ecotype features
    config = config or {}
    ecotype_gen = EcotypeFeatureGenerator(
        min_ecotype_prevalence=config.get('min_ecotype_prevalence', 0.05)
    )
    
    ecotype_features_path = None
    if output_dir:
        ecotype_features_path = output_dir / "ecotype_features.csv"
    
    ecotype_features = ecotype_gen.generate_ecotype_features(
        adata, ecotype_profiles, output_path=ecotype_features_path
    )
    
    # Save feature mapping
    if output_dir and not ecotype_features.empty:
        mapping_path = output_dir / "ecotype_feature_mapping.csv"
        ecotype_gen.save_feature_annotation(mapping_path)
    
    return asv_features, ecotype_features if not ecotype_features.empty else None
