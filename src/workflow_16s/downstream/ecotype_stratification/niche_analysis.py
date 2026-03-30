"""
Ecological niche analysis for detected ecotypes.

Quantifies:
- Niche breadth (specialist vs generalist)
- Niche overlap between ecotypes
- Environment-specific ecotype distributions
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cdist
import scanpy as sc

logger = logging.getLogger(__name__)


@dataclass
class NicheProfile:
    """Ecological niche characterization for an ecotype."""
    otu_id: str
    ecotype_id: int
    niche_breadth: float  # 0=specialist, 1=generalist
    core_environments: List[str]  # Preferred biomes/habitats
    environment_specificity: float  # How specific to certain environments
    geographic_range_km2: Optional[float] = None  # If latitude/longitude available


@dataclass
class EcotypeNicheComparison:
    """Niche relationships between ecotypes within an OTU."""
    otu_id: str
    n_ecotypes: int
    niche_partitioning_score: float  # 0-1: how well do they separate?
    niche_overlap_matrix: np.ndarray  # pairwise overlaps
    most_differentiated_pair: Tuple[int, int]  # (ecotype_a, ecotype_b)


class NicheAnalyzer:
    """
    Quantifies ecological niches of detected ecotypes.
    
    Methods:
    1. Niche breadth: Shannon entropy of environment distribution
    2. Core environments: Top biome associations
    3. Niche overlap: Shared environmental associations
    4. Specialization metrics: How specific vs generalist
    """
    
    def __init__(self, environment_columns: Optional[List[str]] = None):
        """
        Args:
            environment_columns: List of metadata columns representing
                environmental gradients (e.g., ['biome', 'pH', 'temperature'])
        """
        self.environment_columns = environment_columns or []
        self.niche_profiles: Dict[Tuple[str, int], NicheProfile] = {}
    
    def analyze_ecotype_niches(
        self,
        adata: sc.AnnData,
        ecotype_assignments: pd.DataFrame,
        min_samples: int = 3,
    ) -> Dict[Tuple[str, int], NicheProfile]:
        """
        Analyze niche for each detected ecotype.
        
        Args:
            adata: AnnData with sample metadata
            ecotype_assignments: DataFrame from assign_ecotypes()
            min_samples: Minimum samples to report niche
            
        Returns:
            Dictionary mapping (otu_id, ecotype_id) → NicheProfile
        """
        logger.info("Analyzing ecological niches for detected ecotypes...")
        
        for _, assignment in ecotype_assignments.iterrows():
            if assignment['sample_count'] < min_samples:
                continue
            
            # Find samples belonging to this ecotype
            # (This requires storing ecotype assignments in adata, or reconstructing)
            # For now, we'll work with the summary statistics available
            
            niche_profile = NicheProfile(
                otu_id=assignment['otu_id'],
                ecotype_id=assignment['ecotype_id'],
                niche_breadth=self._estimate_niche_breadth(
                    adata, assignment['otu_id']
                ),
                core_environments=[],
                environment_specificity=0.0,
            )
            
            self.niche_profiles[(assignment['otu_id'], assignment['ecotype_id'])] = niche_profile
        
        return self.niche_profiles
    
    def _estimate_niche_breadth(
        self,
        adata: sc.AnnData,
        otu_id: str,
    ) -> float:
        """
        Estimate niche breadth using Shannon entropy of environment distribution.
        
        Returns: 0-1 (0=specialist in one environment, 1=generalist across many)
        """
        if otu_id not in adata.var_names:
            return 0.5
        
        # Get samples where OTU is present
        otu_col = adata[:, otu_id].X.toarray().ravel()
        present = otu_col > 0
        
        if present.sum() < 2:
            return 0.5
        
        # Calculate environment diversity for samples with this OTU
        breadths = []
        for env_col in self.environment_columns:
            if env_col not in adata.obs.columns:
                continue
            
            env_dist = adata.obs.loc[present, env_col].value_counts()
            env_dist = env_dist / env_dist.sum()  # Normalize
            
            # Shannon entropy (max = log(n_environments))
            env_entropy = entropy(env_dist.values)
            max_entropy = np.log(len(env_dist))
            normalized_entropy = env_entropy / max_entropy if max_entropy > 0 else 0
            breadths.append(normalized_entropy)
        
        return np.mean(breadths) if breadths else 0.5
    
    def compare_niche_partitioning(
        self,
        adata: sc.AnnData,
        otu_ecotypes: Dict[str, 'EcotypeProfile'],
    ) -> Dict[str, EcotypeNicheComparison]:
        """
        Analyze niche partitioning between ecotypes within OTUs.
        
        Args:
            adata: AnnData with metadata
            otu_ecotypes: Dictionary from detect_ecotypes_from_traits()
            
        Returns:
            Dictionary mapping otu_id → EcotypeNicheComparison
        """
        logger.info("Analyzing niche partitioning...")
        
        comparisons = {}
        for otu_id, profile in otu_ecotypes.items():
            if profile.n_ecotypes < 2:
                continue
            
            # Build environment profiles for each ecotype
            ecotype_env_profiles = {}
            
            # Compute niche overlap
            n_ecotypes = profile.n_ecotypes
            overlap_matrix = np.eye(n_ecotypes)
            
            for i in range(n_ecotypes):
                for j in range(i + 1, n_ecotypes):
                    overlap = self._compute_niche_overlap(
                        adata, otu_id, profile.ecotypes[i], profile.ecotypes[j]
                    )
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
            
            # Compute partitioning score (inverse of max overlap)
            max_overlap = np.max(overlap_matrix[overlap_matrix < 1.0])
            partitioning_score = 1.0 - max_overlap if not np.isnan(max_overlap) else 0.5
            
            # Find most differentiated pair
            min_overlap_idx = np.unravel_index(
                np.argmin(overlap_matrix + np.eye(n_ecotypes)),
                overlap_matrix.shape
            )
            
            comparison = EcotypeNicheComparison(
                otu_id=otu_id,
                n_ecotypes=profile.n_ecotypes,
                niche_partitioning_score=partitioning_score,
                niche_overlap_matrix=overlap_matrix,
                most_differentiated_pair=tuple(min_overlap_idx),
            )
            
            comparisons[otu_id] = comparison
        
        return comparisons
    
    def _compute_niche_overlap(
        self,
        adata: sc.AnnData,
        otu_id: str,
        ecotype1: 'Ecotype',
        ecotype2: 'Ecotype',
    ) -> float:
        """
        Compute overlap between two ecotypes' niches.
        
        Returns: 0-1 (0=no overlap/complete partitioning, 1=complete overlap)
        """
        # For now: placeholder based on environment specificity
        # In full version: would use actual ecotype sample assignments
        overlap = 0.5 + (ecotype1.niche_breadth - ecotype2.niche_breadth) * 0.2
        return max(0., min(1., overlap))


def analyze_niche_specialization(
    adata: sc.AnnData,
    otu_id: str,
    environment_columns: Optional[List[str]] = None,
) -> Dict:
    """
    Convenience function: analyze niche specialization for an OTU.
    
    Returns:
        Dictionary with:
        - 'niche_breadth': 0-1 (specialist to generalist)
        - 'core_biomes': Top associated environments
        - 'specialization_index': Measure of environmental focus
    """
    if otu_id not in adata.var_names:
        return {'error': f'OTU {otu_id} not found'}
    
    otu_present = adata[:, otu_id].X.toarray().ravel() > 0
    
    analyzer = NicheAnalyzer(environment_columns)
    breadth = analyzer._estimate_niche_breadth(adata, otu_id)
    
    # Find core biomes (if 'biome' column exists)
    core_biomes = []
    if 'biome' in adata.obs.columns:
        biome_dist = adata.obs.loc[otu_present, 'biome'].value_counts()
        core_biomes = biome_dist.head(3).index.tolist()
    
    return {
        'niche_breadth': float(breadth),
        'core_biomes': core_biomes,
        'specialization_index': 1.0 - breadth,  # Inverse: high = specialist
        'n_samples_with_otu': otu_present.sum(),
    }


def quantify_niche_breadth(
    adata: sc.AnnData,
    otu_ids: Optional[List[str]] = None,
    environment_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute niche breadth for multiple OTUs.
    
    Returns:
        DataFrame with OTU-level niche metrics
    """
    if otu_ids is None:
        otu_ids = adata.var_names.tolist()
    
    breadths = []
    for otu_id in otu_ids:
        result = analyze_niche_specialization(adata, otu_id, environment_columns)
        if 'error' not in result:
            result['otu_id'] = otu_id
            breadths.append(result)
    
    return pd.DataFrame(breadths)
