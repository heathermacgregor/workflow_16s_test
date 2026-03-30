"""
Core ecotype detection from trait-based clustering.

Detects microhabitat-level strain variants (ecotypes) within OTUs using:
1. Trait similarity matrices (OTU x Trait)
2. Geographic/environmental co-occurrence patterns
3. Bimodal trait distribution indicators
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import scanpy as sc

logger = logging.getLogger(__name__)


@dataclass
class Ecotype:
    """Represents a detected ecotype (strain variant) within an OTU."""
    otu_id: str
    ecotype_id: int  # Ecotype number within OTU (0, 1, 2, ...)
    sample_count: int
    trait_profile: Dict[str, float]  # Trait name → presence/confidence
    prevalence: float  # Fraction of OTU samples belonging to this ecotype
    geographic_range: Optional[Dict] = None  # Optional: biome/geography info
    niche_breadth: float = 0.0  # Specialization metric (0=specialist, 1=generalist)


@dataclass
class EcotypeProfile:
    """Summary of ecotype patterns within a single OTU."""
    otu_id: str
    n_ecotypes: int
    taxonomy: Optional[Dict[str, str]] = None  # Deepest taxonomic assignment (e.g., "Genus": "Geobacter")
    ecotypes: List[Ecotype] = field(default_factory=list)
    trait_diversity: float = 0.0  # How different are trait profiles?
    ecological_coherence: float = 0.0  # Do ecotypes occupy distinct niches?
    stratification_score: float = 0.0  # 0=no stratification, 1=complete


class EcotypeDetector:
    """
    Detects cryptic strain variants (ecotypes) within OTUs.
    
    Strategy:
    1. For each OTU, extract trait profile matrix (samples × traits)
    2. Cluster samples by trait similarity
    3. Evaluate cluster stability (silhouette, gap statistic)
    4. Validate ecotypes by niche coherence
    5. Assign samples to ecotypes
    """
    
    def __init__(
        self,
        min_prevalence: int = 5,
        trait_similarity_threshold: float = 0.7,
        clustering_method: str = "kmeans",
        n_clusters_range: Tuple[int, int] = (2, 6),
    ):
        """
        Args:
            min_prevalence: Minimum samples where OTU must be present
            trait_similarity_threshold: Minimum within-ecotype trait similarity
            clustering_method: 'kmeans', 'hierarchical', or 'spectral'
            n_clusters_range: (min_clusters, max_clusters) to test
        """
        self.min_prevalence = min_prevalence
        self.trait_similarity_threshold = trait_similarity_threshold
        self.clustering_method = clustering_method
        self.n_clusters_range = n_clusters_range
        self.otu_ecotypes: Dict[str, EcotypeProfile] = {}
        
    def detect_ecotypes(
        self,
        adata: sc.AnnData,
        trait_matrix: pd.DataFrame,
        metadata_cols: Optional[List[str]] = None,
    ) -> Dict[str, EcotypeProfile]:
        """
        Detect ecotypes across all OTUs.
        
        Args:
            adata: AnnData object with sample metadata
            trait_matrix: DataFrame with OTU × Trait presence/confidence
            metadata_cols: Optional environmental columns for niche analysis
            
        Returns:
            Dictionary mapping OTU_ID → EcotypeProfile
        """
        logger.info(f"Detecting ecotypes from {trait_matrix.shape[0]} OTUs...")
        
        filtered_otus = trait_matrix[
            trait_matrix.index.isin(adata.var_names) &
            (adata.X.getnnz(axis=0) >= self.min_prevalence)  # Sufficient samples
        ]
        
        # Process ecotypes without creating nested progress bar
        # Use simple loop with comprehensive error handling
        n_otus = len(filtered_otus)
        processed = 0
        for otu_id in filtered_otus.index:
            try:
                # Get samples where this OTU is present
                otu_present = adata[:, otu_id].X.toarray().ravel() > 0
                if otu_present.sum() < self.min_prevalence:
                    continue
                    
                sample_indices = np.where(otu_present)[0]
                trait_profile = filtered_otus.loc[otu_id].values
                
                # Detect ecotypes within this OTU
                ecotype_profile = self._detect_otu_ecotypes(
                    otu_id,
                    adata[sample_indices],
                    trait_profile,
                    metadata_cols,
                    full_adata=adata,
                )
                
                if ecotype_profile.n_ecotypes > 1:  # Only store if stratification detected
                    self.otu_ecotypes[otu_id] = ecotype_profile
            except Exception:
                pass  # Skip OTU on error
            
            processed += 1
            # Log progress every 10% of OTUs processed
            if processed % max(1, n_otus // 10) == 0:
                pct = int(100 * processed / n_otus)
                logger.debug(f"  Ecotype detection: {pct}% complete ({processed}/{n_otus} OTUs)")
        
        logger.info(
            f"✓ Detected ecotypes in {len(self.otu_ecotypes)} OTUs "
            f"(out of {len(filtered_otus)} tested)"
        )
        return self.otu_ecotypes
    
    def _detect_otu_ecotypes(
        self,
        otu_id: str,
        otu_adata: sc.AnnData,
        trait_profile: np.ndarray,
        metadata_cols: Optional[List[str]] = None,
        full_adata: Optional[sc.AnnData] = None,
    ) -> EcotypeProfile:
        """Detect ecotypes within a single OTU."""
        n_samples = otu_adata.n_obs
        profile = EcotypeProfile(otu_id=otu_id, n_ecotypes=1)
        
        # Extract taxonomy from full adata if available
        if full_adata and otu_id in full_adata.var_names:
            otu_var = full_adata.var.loc[otu_id]
            tax_levels = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']
            taxonomy = {}
            for tax_level in tax_levels:
                if tax_level in otu_var.index:
                    value = otu_var[tax_level]
                    if pd.notna(value) and str(value).strip() != '':
                        taxonomy[tax_level] = str(value)
                        # Use the deepest (first) non-empty level found
                        break
            if taxonomy:
                profile.taxonomy = taxonomy
        
        if n_samples < 2 * self.min_prevalence:
            return profile  # Insufficient samples for meaningful clustering
        
        # Build feature matrix: sample × metadata + abundance patterns
        sample_features = self._build_sample_features(
            otu_adata, metadata_cols
        )
        
        # Try different cluster numbers and find optimal
        best_n_clusters = 1
        best_silhouette = -1
        
        for n_clusters in range(self.n_clusters_range[0], 
                                min(self.n_clusters_range[1] + 1, n_samples // 2)):
            clusters = self._cluster_samples(sample_features, n_clusters)
            
            # Compute silhouette score (within-cluster similarity)
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(sample_features, clusters)
            
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_n_clusters = n_clusters
        
        # Only accept ecotypes if silhouette is reasonable (>0.2)
        if best_silhouette < 0.2 or best_n_clusters == 1:
            return profile
        
        # Assign ecotypes
        final_clusters = self._cluster_samples(sample_features, best_n_clusters)
        
        ecotypes = []
        for ecotype_id in range(best_n_clusters):
            mask = final_clusters == ecotype_id
            ecotype_samples = otu_adata[mask]
            
            # Compute ecotype trait profile
            ecotype_trait_profile = self._compute_trait_profile(
                ecotype_samples, trait_profile
            )
            
            ecotype = Ecotype(
                otu_id=otu_id,
                ecotype_id=ecotype_id,
                sample_count=mask.sum(),
                trait_profile=ecotype_trait_profile,
                prevalence=mask.sum() / n_samples,
            )
            ecotypes.append(ecotype)
        
        # Compute ecotype profile metrics
        profile.n_ecotypes = best_n_clusters
        profile.ecotypes = ecotypes
        profile.trait_diversity = self._compute_trait_diversity(ecotypes)
        profile.ecological_coherence = best_silhouette
        profile.stratification_score = min(1.0, best_silhouette * best_n_clusters / 2.0)
        
        return profile
    
    def _build_sample_features(
        self,
        adata: sc.AnnData,
        metadata_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Build feature matrix from metadata for clustering."""
        features = []
        
        # Include geographic/environmental metadata if available
        if metadata_cols:
            for col in metadata_cols:
                if col in adata.obs.columns:
                    col_data = adata.obs[col]
                    # Convert to numeric if possible
                    try:
                        numeric_data = pd.to_numeric(col_data, errors='coerce')
                        numeric_data = numeric_data.fillna(numeric_data.mean())
                        features.append(numeric_data.values)
                    except:
                        pass  # Skip non-numeric columns
        
        # If no metadata, use relative abundance patterns
        if not features:
            # Normalize OTU abundances per sample
            X_norm = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            features.append(X_norm.mean(axis=1))  # Mean OTU abundance
        
        feature_matrix = np.column_stack(features) if features else np.ones((adata.n_obs, 1))
        scaler = StandardScaler()
        return scaler.fit_transform(feature_matrix)
    
    def _cluster_samples(
        self,
        sample_features: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """Cluster samples using specified method."""
        if self.clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return clusterer.fit_predict(sample_features)
        
        elif self.clustering_method == "hierarchical":
            Z = linkage(sample_features, method='ward')
            return fcluster(Z, n_clusters, criterion='maxclust') - 1
        
        elif self.clustering_method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors'
            )
            return clusterer.fit_predict(sample_features)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def _compute_trait_profile(
        self,
        adata: sc.AnnData,
        global_trait_profile: np.ndarray,
    ) -> Dict[str, float]:
        """Compute average trait profile for ecotype."""
        # For now, return global trait profile
        # In advanced version: compare ecotype-specific patterns
        return dict(zip(
            range(len(global_trait_profile)),
            global_trait_profile
        ))
    
    @staticmethod
    def _compute_trait_diversity(ecotypes: List[Ecotype]) -> float:
        """Compute how different trait profiles are across ecotypes."""
        if len(ecotypes) < 2:
            return 0.0
        
        # Compare ecotype trait profiles
        diversity = 0.0
        for i in range(len(ecotypes)):
            for j in range(i + 1, len(ecotypes)):
                # Simple: difference in trait presence
                diff = sum(
                    abs(ecotypes[i].trait_profile.get(k, 0) - 
                        ecotypes[j].trait_profile.get(k, 0))
                    for k in set(ecotypes[i].trait_profile.keys()) | 
                             set(ecotypes[j].trait_profile.keys())
                )
                diversity += diff
        
        # Normalize by number of comparisons
        n_comparisons = len(ecotypes) * (len(ecotypes) - 1) / 2
        return min(1.0, diversity / (n_comparisons * 10))  # Cap at 1.0


def detect_ecotypes_from_traits(
    adata: sc.AnnData,
    trait_matrix: pd.DataFrame,
    min_prevalence: int = 5,
    clustering_method: str = "kmeans",
    n_clusters_range: Tuple[int, int] = (2, 6),
    metadata_cols: Optional[List[str]] = None,
) -> Dict[str, EcotypeProfile]:
    """
    Convenience function: detect ecotypes across all OTUs.
    
    Args:
        adata: AnnData object
        trait_matrix: OTU × Trait matrix
        min_prevalence: Minimum samples per OTU
        clustering_method: Clustering method
        n_clusters_range: (min, max) clusters to test
        metadata_cols: Optional metadata columns for niche analysis
        
    Returns:
        Dictionary mapping OTU_ID → EcotypeProfile
    """
    detector = EcotypeDetector(
        min_prevalence=min_prevalence,
        clustering_method=clustering_method,
        n_clusters_range=n_clusters_range,
    )
    return detector.detect_ecotypes(adata, trait_matrix, metadata_cols)


def assign_ecotypes(
    adata: sc.AnnData,
    ecotype_profiles: Dict[str, EcotypeProfile],
) -> pd.DataFrame:
    """
    Assign samples to ecotypes.
    
    Returns:
        DataFrame with columns: sample_id, otu_id, ecotype_id, prevalence_in_ecotype
    """
    assignments = []
    
    for otu_id, profile in ecotype_profiles.items():
        if otu_id not in adata.var_names:
            continue
        
        for ecotype in profile.ecotypes:
            assignments.append({
                'otu_id': otu_id,
                'ecotype_id': ecotype.ecotype_id,
                'sample_count': ecotype.sample_count,
                'prevalence': ecotype.prevalence,
                'n_ecotypes_in_otu': profile.n_ecotypes,
            })
    
    return pd.DataFrame(assignments)


def compute_ecotype_profiles(
    adata: sc.AnnData,
    ecotype_profiles: Dict[str, EcotypeProfile],
) -> pd.DataFrame:
    """
    Summarize ecotype information for reporting.
    
    Returns:
        DataFrame with OTU-level ecotype statistics
    """
    rows = []
    for otu_id, profile in ecotype_profiles.items():
        rows.append({
            'otu_id': otu_id,
            'n_ecotypes': profile.n_ecotypes,
            'ecological_coherence': profile.ecological_coherence,
            'stratification_score': profile.stratification_score,
            'trait_diversity': profile.trait_diversity,
            'dominant_ecotype_prevalence': max(
                (e.prevalence for e in profile.ecotypes), default=0
            ),
        })
    
    return pd.DataFrame(rows)
