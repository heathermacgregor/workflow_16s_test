"""
ASV-to-MAG Mapping Pipeline

Maps 16S rRNA amplicon sequence variants (ASVs) to metagenome-assembled genomes (MAGs)
using VSEARCH alignment with configurable identity/coverage thresholds.

Supports multiple MAG sources:
- GTDB (Genome Taxonomy Database)
- JGI IMG/M (requires authentication)
- MicrobeAtlas
- User-provided MAGs

Output: Sparse COO matrix (ASV × MAG) with alignment weights for downstream
functional profiling.

Example:
>>> from workflow_16s.downstream.asv_mag_mapping import map_asvs_to_mags
>>> weights = map_asvs_to_mags(adata, config)
>>> # weights.shape = (n_asvs, n_mags) sparse COO matrix
"""

from .gtdb_client import GTDBClient
from .vsearch_wrapper import VSEARCHWrapper, run_vsearch_alignment
from .assignment_engine import AssignmentEngine, allocate_multiple_hits

__all__ = [
    "GTDBClient",
    "VSEARCHWrapper",
    "run_vsearch_alignment",
    "AssignmentEngine",
    "allocate_multiple_hits",
]
