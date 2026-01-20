"""
Enhanced Contamination Detection - Reference-Based Methods

Extends decontamination.py with methods that don't require negative controls:
1. Reference database matching (known contaminants)
2. Frequency-based detection (no controls needed)
3. Ubiquity-based detection (low abundance across all samples)
4. Cross-sample contamination detection
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr

logger = logging.getLogger('workflow_16s')


# Known contaminant databases (from literature)
KITOME_GENERA = [
    # DNA extraction kit contaminants (Salter et al. 2014)
    'Bradyrhizobium', 'Sphingomonas', 'Phyllobacterium',
    'Burkholderia', 'Ralstonia', 'Cupriavidus',
    'Methylobacterium', 'Novosphingobium',
    # Water contaminants
    'Pseudomonas', 'Acinetobacter', 'Stenotrophomonas',
    'Sphingobium', 'Sphingopyxis',
    # Common environmental ubiquitous
    'Herbaspirillum', 'Achromobacter', 'Rhodococcus',
]

HUMAN_CONTAMINANTS = {
    'skin': ['Propionibacterium', 'Cutibacterium', 'Staphylococcus', 
            'Corynebacterium', 'Micrococcus', 'Kocuria'],
    'gut': ['Bacteroides', 'Prevotella', 'Faecalibacterium',
           'Bifidobacterium', 'Ruminococcus', 'Blautia',
           'Escherichia', 'Enterococcus', 'Clostridium'],
    'oral': ['Streptococcus', 'Veillonella', 'Fusobacterium',
            'Porphyromonas', 'Actinomyces', 'Rothia']
}


def detect_contaminants_reference_based(
    adata: ad.AnnData,
    method: str = 'combined',
    prevalence_threshold: float = 0.9,
    abundance_threshold: float = 0.001,
    exclude_env_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Multi-method contamination detection without requiring controls.
    
    This is critical for public data where negative controls are rarely available.
    
    Methods:
        1. 'database': Match against known contaminant databases
        2. 'frequency': Taxa inversely correlated with total read depth
        3. 'ubiquity': Taxa in >90% samples at low abundance (kitome signature)
        4. 'combined': All three methods with consensus scoring
    
    Args:
        adata: AnnData object with feature table and metadata
        method: Detection method
        prevalence_threshold: Prevalence cutoff for ubiquity method (default: 0.9)
        abundance_threshold: Abundance cutoff for ubiquity method (default: 0.001)
        exclude_env_types: Environment types to exclude from human contamination checks
    
    Returns:
        DataFrame with contamination scores per feature
    
    Example:
        >>> contam_scores = detect_contaminants_reference_based(
        ...     adata,
        ...     method='combined',
        ...     exclude_env_types=['gut', 'skin']  # Don't flag human taxa in human samples
        ... )
        >>> # Filter contaminants
        >>> likely_contaminants = contam_scores[contam_scores['combined_score'] > 0.5]
    """
    logger.info(f"Reference-based contamination detection: method={method}")
    
    results = pd.DataFrame(index=adata.var_names)
    
    # Run selected methods
    if method in ['database', 'combined']:
        results['db_score'] = _database_matching(adata, exclude_env_types)
    
    if method in ['frequency', 'combined']:
        results['freq_score'] = _frequency_based_detection(adata)
    
    if method in ['ubiquity', 'combined']:
        results['ubiq_score'] = _ubiquity_based_detection(
            adata, prevalence_threshold, abundance_threshold
        )
    
    # Combined consensus score
    if method == 'combined':
        score_cols = [c for c in results.columns if c.endswith('_score')]
        results['combined_score'] = results[score_cols].mean(axis=1)
        results['n_methods_flagged'] = (results[score_cols] > 0.5).sum(axis=1)
    
    # Add taxonomy if available
    if 'taxonomy' in adata.var.columns:
        results['taxonomy'] = adata.var['taxonomy']
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
        if level in adata.var.columns:
            results[level] = adata.var[level]
    
    # Add prevalence and mean abundance for context
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    results['prevalence'] = (X > 0).sum(axis=0) / X.shape[0]
    results['mean_abundance'] = X.mean(axis=0)
    
    # Flag as contaminant (conservative threshold)
    if method == 'combined':
        results['is_contaminant'] = results['combined_score'] > 0.5
    else:
        score_col = f"{method}_score"
        results['is_contaminant'] = results[score_col] > 0.5
    
    n_contam = results['is_contaminant'].sum()
    logger.info(f"Flagged {n_contam}/{len(results)} features as likely contaminants")
    
    return results


def _database_matching(adata: ad.AnnData, 
                       exclude_env_types: Optional[List[str]] = None) -> pd.Series:
    """
    Match features against known contaminant databases.
    
    Args:
        adata: AnnData object
        exclude_env_types: Environment types to exclude from human checks
    
    Returns:
        Series with contamination scores (0-1)
    """
    logger.info("Database matching for known contaminants...")
    
    scores = pd.Series(0.0, index=adata.var_names)
    
    if 'Genus' not in adata.var.columns:
        logger.warning("No Genus column in .var, cannot do database matching")
        return scores
    
    genera = adata.var['Genus'].fillna('Unknown')
    
    # Check kitome contaminants
    kitome_mask = genera.isin(KITOME_GENERA)
    scores[kitome_mask] = 0.7  # High confidence for kit contaminants
    
    # Check human contaminants (unless sample is human-associated)
    check_human = True
    if exclude_env_types and 'env_category_type' in adata.obs.columns:
        # If ANY sample is from excluded environments, don't flag human taxa
        env_types = adata.obs['env_category_type'].unique()
        if any(env in env_types for env in exclude_env_types):
            check_human = False
            logger.info(f"Skipping human contamination check (found {exclude_env_types} samples)")
    
    if check_human:
        all_human_genera = []
        for source, genera_list in HUMAN_CONTAMINANTS.items():
            all_human_genera.extend(genera_list)
        
        human_mask = genera.isin(all_human_genera)
        
        # Score based on which source (skin > gut > oral in environmental samples)
        for source, genera_list in HUMAN_CONTAMINANTS.items():
            source_mask = genera.isin(genera_list)
            if source == 'skin':
                scores[source_mask] = 0.8  # Skin contamination very likely in env samples
            elif source == 'gut':
                scores[source_mask] = 0.6  # Gut possible but could be real (animal feces)
            elif source == 'oral':
                scores[source_mask] = 0.5  # Oral less common
    
    n_flagged = (scores > 0).sum()
    logger.info(f"Database matching: {n_flagged} features matched known contaminants")
    
    return scores


def _frequency_based_detection(adata: ad.AnnData) -> pd.Series:
    """
    Frequency-based detection: contaminants inversely correlated with total reads.
    
    This works because:
    - Low-biomass samples have more contamination
    - High-biomass samples have less contamination
    - Real taxa correlate positively or show no correlation
    
    Args:
        adata: AnnData object
    
    Returns:
        Series with contamination scores (0-1)
    """
    logger.info("Frequency-based contamination detection...")
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Calculate total reads per sample
    total_reads = X.sum(axis=1)
    
    if len(total_reads) < 10:
        logger.warning("Too few samples (<10) for frequency-based detection")
        return pd.Series(0.0, index=adata.var_names)
    
    scores = pd.Series(0.0, index=adata.var_names)
    
    # For each feature, calculate correlation with total reads
    for i in range(X.shape[1]):
        feature_abundance = X[:, i]
        
        # Skip features not present in enough samples
        if (feature_abundance > 0).sum() < 5:
            continue
        
        # Calculate Spearman correlation
        try:
            corr, pval = spearmanr(total_reads, feature_abundance)
            
            # Negative correlation suggests contamination
            if corr < 0 and pval < 0.05:
                # Score based on strength of negative correlation
                scores.iloc[i] = min(1.0, abs(corr))
        except (ValueError, RuntimeError) as e:
            logger.debug(f"Correlation calculation failed for feature {i}: {e}")
            continue
    
    n_flagged = (scores > 0.5).sum()
    logger.info(f"Frequency-based: {n_flagged} features negatively correlated with depth")
    
    return scores


def _ubiquity_based_detection(adata: ad.AnnData,
                               prevalence_threshold: float = 0.9,
                               abundance_threshold: float = 0.001) -> pd.Series:
    """
    Ubiquity-based detection: taxa in >90% samples at low abundance.
    
    This is the "kitome signature" - contaminants from reagents appear
    in almost all samples at low, consistent levels.
    
    Args:
        adata: AnnData object
        prevalence_threshold: Minimum prevalence to flag (default: 0.9)
        abundance_threshold: Maximum mean abundance to flag (default: 0.001)
    
    Returns:
        Series with contamination scores (0-1)
    """
    logger.info(f"Ubiquity-based detection (prev>{prevalence_threshold}, "
               f"abund<{abundance_threshold})...")
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Calculate prevalence (fraction of samples)
    prevalence = (X > 0).sum(axis=0) / X.shape[0]
    
    # Calculate mean relative abundance
    X_rel = X / X.sum(axis=1, keepdims=True)
    mean_abundance = X_rel.mean(axis=0)
    
    # Calculate coefficient of variation (consistency of abundance)
    std_abundance = X_rel.std(axis=0)
    cv = std_abundance / (mean_abundance + 1e-10)
    
    scores = pd.Series(0.0, index=adata.var_names)
    
    # Ubiquitous + low abundance + consistent = likely contaminant
    ubiq_mask = (prevalence > prevalence_threshold) & \
                (mean_abundance < abundance_threshold)
    
    scores[ubiq_mask] = 0.6
    
    # Higher score if also very consistent (low CV)
    consistent_mask = ubiq_mask & (cv < 1.0)
    scores[consistent_mask] = 0.8
    
    n_flagged = (scores > 0).sum()
    logger.info(f"Ubiquity-based: {n_flagged} features are ubiquitous at low abundance")
    
    return scores


def detect_cross_sample_contamination(adata: ad.AnnData,
                                       batch_column: str = 'batch',
                                       min_overlap: float = 0.7) -> Dict[str, List[Tuple[str, str]]]:
    """
    Detect potential cross-sample contamination (index hopping, barcode swapping).
    
    Looks for samples that share >70% of their features with another sample
    in the same batch, which suggests contamination or sample swaps.
    
    Args:
        adata: AnnData object
        batch_column: Column indicating batch/sequencing run
        min_overlap: Minimum Jaccard similarity to flag (default: 0.7)
    
    Returns:
        Dict of batch -> list of (sample1, sample2) pairs with high overlap
    """
    logger.info(f"Detecting cross-sample contamination using {batch_column}...")
    
    if batch_column not in adata.obs.columns:
        logger.warning(f"Column {batch_column} not found, cannot detect cross-contamination")
        return {}
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X_binary = (X > 0).astype(int)
    
    suspicious_pairs = {}
    
    # Process each batch separately
    for batch in adata.obs[batch_column].unique():
        if pd.isna(batch):
            continue
        
        batch_mask = adata.obs[batch_column] == batch
        batch_samples = adata.obs_names[batch_mask]
        
        if len(batch_samples) < 2:
            continue
        
        batch_X = X_binary[batch_mask, :]
        
        # Calculate pairwise Jaccard similarity
        pairs = []
        for i in range(len(batch_samples)):
            for j in range(i+1, len(batch_samples)):
                s1, s2 = batch_X[i], batch_X[j]
                
                # Jaccard: intersection / union
                intersection = (s1 & s2).sum()
                union = (s1 | s2).sum()
                
                if union > 0:
                    jaccard = intersection / union
                    
                    if jaccard > min_overlap:
                        pairs.append((batch_samples[i], batch_samples[j], jaccard))
        
        if pairs:
            suspicious_pairs[str(batch)] = pairs
            logger.warning(f"Batch {batch}: {len(pairs)} suspicious sample pairs detected")
    
    return suspicious_pairs


def remove_contaminants_enhanced(adata: ad.AnnData,
                                  contamination_scores: pd.DataFrame,
                                  threshold: float = 0.5,
                                  inplace: bool = False) -> ad.AnnData:
    """
    Remove contaminants identified by reference-based detection.
    
    Args:
        adata: AnnData object
        contamination_scores: DataFrame from detect_contaminants_reference_based()
        threshold: Score threshold for removal (default: 0.5)
        inplace: Whether to modify adata in place
    
    Returns:
        AnnData with contaminants removed
    """
    if not inplace:
        adata = adata.copy()
    
    # Determine which features to keep
    if 'combined_score' in contamination_scores.columns:
        keep_mask = contamination_scores['combined_score'] <= threshold
    elif 'is_contaminant' in contamination_scores.columns:
        keep_mask = ~contamination_scores['is_contaminant']
    else:
        # Use first score column
        score_col = [c for c in contamination_scores.columns if c.endswith('_score')][0]
        keep_mask = contamination_scores[score_col] <= threshold
    
    keep_features = contamination_scores.index[keep_mask].tolist()
    
    n_removed = len(adata.var_names) - len(keep_features)
    logger.info(f"Removing {n_removed} contaminant features (threshold={threshold})")
    
    # Filter AnnData
    adata = adata[:, keep_features].copy()
    
    return adata
