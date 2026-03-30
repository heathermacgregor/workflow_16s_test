"""
Phylogenetic Signal Analysis

Calculates Pagel's lambda and other phylogenetic signal metrics to assess
whether functional traits follow evolutionary relationships or are acquired
via horizontal gene transfer.

Key insight (Adam Arkin's guidance):
- High lambda (λ > 0.7): Trait is phylogenetically conserved (vertical inheritance)
- Low lambda (λ < 0.3): Trait is randomly distributed (horizontal transfer)
- Mid-range (0.3-0.7): Mixed pattern (some families have it, some don't)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PhylogeneticSignalResult:
    """Results from phylogenetic signal calculation."""
    trait_name: str
    lambda_value: float  # Pagel's lambda (0-1)
    p_value: Optional[float]  # Statistical significance
    interpretation: str  # "conserved", "random", "mixed"
    confidence: float  # How confident in this estimate
    n_otus_with_trait: int
    n_otus_total: int


class PhylogeneticSignalCalculator:
    """
    Calculates phylogenetic signal using taxonomic relationships.
    """
    
    def __init__(self, taxonomy_data: pd.DataFrame = None):
        """
        Initialize calculator with taxonomy data.
        
        Parameters
        ----------
        taxonomy_data : pd.DataFrame
            DataFrame with OTU as index and taxonomy columns
            (Kingdom, Phylum, Class, Order, Family, Genus, Species)
        """
        self.taxonomy_data = taxonomy_data
        self.taxonomic_distances = None
    
    def _build_taxonomic_distance_matrix(self, otu_list: List[str]) -> np.ndarray:
        """
        Build distance matrix based on taxonomic ranks.
        
        Closer taxonomic relationships = smaller distances
        Kingdom difference: 6 steps
        Phylum: 5 steps
        Class: 4 steps
        Order: 3 steps
        Family: 2 steps
        Genus: 1 step
        Species: 0 steps (same species)
        """
        n_otus = len(otu_list)
        distances = np.zeros((n_otus, n_otus))
        
        if self.taxonomy_data is None:
            logger.warning("No taxonomy data provided. Cannot calculate phylogenetic distances.")
            return distances
        
        taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        weights = [6, 5, 4, 3, 2, 1, 0]  # Lower weights for deeper taxonomy
        
        for i, otu_i in enumerate(otu_list):
            if otu_i not in self.taxonomy_data.index:
                continue
            
            for j, otu_j in enumerate(otu_list):
                if j <= i:  # Symmetric matrix
                    continue
                if otu_j not in self.taxonomy_data.index:
                    continue
                
                # Calculate distance based on most specific common taxonomy
                distance = 0
                for level, weight in zip(taxonomy_levels, weights):
                    if level in self.taxonomy_data.columns:
                        val_i = self.taxonomy_data.loc[otu_i, level]
                        val_j = self.taxonomy_data.loc[otu_j, level]
                        
                        if str(val_i) != str(val_j):
                            distance = weight
                            break
                
                distances[i, j] = distance
                distances[j, i] = distance
        
        return distances
    
    def calculate_pagels_lambda_simple(
        self,
        trait_values: Dict[str, float],
        otu_list: List[str],
        threshold: float = 0.5
    ) -> PhylogeneticSignalResult:
        """
        Calculate simplified Pagel's lambda using phylogenetic contrasts.
        
        This is a simplified implementation suitable for large datasets.
        High lambda (close to 1) means trait follows the evolutionary tree.
        Low lambda (close to 0) means trait distribution is random.
        
        Parameters
        ----------
        trait_values : Dict[str, float]
            OTU -> trait confidence/presence score
        otu_list : List[str]
            List of OTUs to analyze
        threshold : float
            Confidence threshold for presence/absence (binary)
        
        Returns
        -------
        PhylogeneticSignalResult
            Lambda value and interpretation
        """
        # Convert to binary presence/absence
        trait_presence = {}
        for otu in otu_list:
            if otu in trait_values:
                trait_presence[otu] = 1 if trait_values[otu] >= threshold else 0
            else:
                trait_presence[otu] = 0
        
        otus_with_trait = sum(trait_presence.values())
        
        # Build distance matrix
        distances = self._build_taxonomic_distance_matrix(otu_list)
        
        # Calculate phylogenetic contrast correlation
        # If trait presence is correlated with phylogenetic similarity, lambda is high
        if otus_with_trait == 0 or otus_with_trait == len(otu_list):
            # Trait is fixed (all have it or none have it) - no signal
            lambda_value = 0.0
            interpretation = "fixed"
        else:
            # Calculate correlation between trait differences and phylogenetic distance
            n_otus = len(otu_list)
            otu_array = list(otu_list)
            
            trait_diffs = []
            phylo_dists = []
            
            for i in range(n_otus):
                for j in range(i + 1, n_otus):
                    trait_diff = abs(trait_presence[otu_array[i]] - trait_presence[otu_array[j]])
                    if distances[i, j] > 0:  # Only pairs with different taxonomy
                        trait_diffs.append(trait_diff)
                        phylo_dists.append(distances[i, j])
            
            if len(trait_diffs) == 0 or len(phylo_dists) == 0:
                lambda_value = 0.0
                interpretation = "insufficient_pairs"
            else:
                # Correlation: high = lambda is high (phylogenetically conserved)
                # Low correlation = lambda is low (random distribution)
                # Normalize the correlation to [0, 1] range
                corr = np.corrcoef(trait_diffs, phylo_dists)[0, 1]
                
                # Transform: negative correlation (closer splits have same trait) = high lambda
                if np.isnan(corr):
                    lambda_value = 0.5
                else:
                    # Higher phylogenetic distance should NOT correlate with trait difference
                    # So we invert: if correlation is negative, that means closer taxa are similar
                    lambda_value = max(0, -1 * corr)  # Invert and bound at 0
        
        # Interpret lambda
        if lambda_value > 0.7:
            interpretation = "phylogenetically_conserved"
        elif lambda_value < 0.3:
            interpretation = "randomly_distributed"
        else:
            interpretation = "mixed_pattern"
        
        p_value = None  # Would need proper statistical test
        
        return PhylogeneticSignalResult(
            trait_name="",
            lambda_value=lambda_value,
            p_value=p_value,
            interpretation=interpretation,
            confidence=min(1.0, otus_with_trait / max(1, len(otu_list))),  # More OTUs = more confidence
            n_otus_with_trait=otus_with_trait,
            n_otus_total=len(otu_list)
        )


def calculate_phylogenetic_signal(
    trait_matrix: pd.DataFrame,
    taxonomy_data: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Calculate phylogenetic signal for all traits in the matrix.
    
    Parameters
    ----------
    trait_matrix : pd.DataFrame
        OTU x Trait matrix with confidence scores
    taxonomy_data : pd.DataFrame
        OTU taxonomy data (Kingdom, Phylum, Class, etc.)
    threshold : float
        Confidence threshold for trait presence
    
    Returns
    -------
    pd.DataFrame
        Results with one row per trait and lambda values
    """
    calculator = PhylogeneticSignalCalculator(taxonomy_data)
    
    results = []
    
    for trait_col in trait_matrix.columns:
        trait_values = trait_matrix[trait_col].to_dict()
        otu_list = trait_matrix.index.tolist()
        
        result = calculator.calculate_pagels_lambda_simple(
            trait_values,
            otu_list,
            threshold=threshold
        )
        result.trait_name = trait_col
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'trait': r.trait_name,
            'pagels_lambda': r.lambda_value,
            'interpretation': r.interpretation,
            'n_otus_with_trait': r.n_otus_with_trait,
            'n_otus_total': r.n_otus_total,
            'proportion_with_trait': r.n_otus_with_trait / r.n_otus_total
        }
        for r in results
    ])
    
    logger.info(f"Calculated phylogenetic signal for {len(results)} traits")
    logger.info(f"Conserved traits (λ>0.7): {(results_df['pagels_lambda'] > 0.7).sum()}")
    logger.info(f"Random distribution (λ<0.3): {(results_df['pagels_lambda'] < 0.3).sum()}")
    
    return results_df


def calculate_pagels_lambda(
    trait_presence: pd.Series,
    taxonomy_data: pd.DataFrame
) -> Tuple[float, str]:
    """
    Calculate single Pagel's lambda value for a trait.
    
    Parameters
    ----------
    trait_presence : pd.Series
        Presence/absence for each OTU (index is OTU ID)
    taxonomy_data : pd.DataFrame
        Taxonomy with OTU as index
    
    Returns
    -------
    Tuple[float, str]
        (lambda value, interpretation)
    """
    calculator = PhylogeneticSignalCalculator(taxonomy_data)
    
    result = calculator.calculate_pagels_lambda_simple(
        trait_presence.to_dict(),
        trait_presence.index.tolist()
    )
    
    return result.lambda_value, result.interpretation


def assess_trait_phylogenetic_structure(
    trait_matrix: pd.DataFrame,
    taxonomy_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Comprehensive assessment of how traits relate to evolutionary structure.
    
    Returns metrics for:
    - Phylogenetic signal (Pagel's lambda)
    - Vertical vs. horizontal inheritance patterns
    - Potential for horizontal gene transfer
    
    Parameters
    ----------
    trait_matrix : pd.DataFrame
        OTU x Trait matrix
    taxonomy_data : pd.DataFrame
        Taxonomy data
    
    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results
    """
    signal_results = calculate_phylogenetic_signal(trait_matrix, taxonomy_data)
    
    # Classify traits
    conserved = signal_results[signal_results['pagels_lambda'] > 0.7]
    random = signal_results[signal_results['pagels_lambda'] < 0.3]
    mixed = signal_results[
        (signal_results['pagels_lambda'] >= 0.3) & 
        (signal_results['pagels_lambda'] <= 0.7)
    ]
    
    summary = {
        'total_traits_analyzed': len(signal_results),
        'phylogenetically_conserved_traits': len(conserved),
        'randomly_distributed_traits': len(random),
        'mixed_pattern_traits': len(mixed),
        'conserved_trait_names': conserved['trait'].tolist(),
        'random_trait_names': random['trait'].tolist(),
        'mean_lambda': signal_results['pagels_lambda'].mean(),
        'std_lambda': signal_results['pagels_lambda'].std(),
        'full_results': signal_results
    }
    
    logger.info(f"Phylogenetic structure assessment complete:")
    logger.info(f"  Conserved: {len(conserved)} traits")
    logger.info(f"  Random: {len(random)} traits")
    logger.info(f"  Mixed: {len(mixed)} traits")
    logger.info(f"  Mean lambda: {summary['mean_lambda']:.3f}")
    
    return summary
