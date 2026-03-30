"""
Functional Profile Construction Module

Builds sample × KEGG Ortholog (KO) abundance matrix through efficient matrix multiplication:

    sample_KO = (sample × ASV counts) × (ASV × MAG weights) × (MAG → KO presence)

Supports:
- DRAM output parsing (KEGG annotation from MAGs)
- Sparse matrix operations for memory efficiency
- CLR transformation for compositional analysis

Example:
>>> from workflow_16s.downstream.functional_profiling import build_sample_ko_matrix
>>> adata = build_sample_ko_matrix(adata, asv_mag_weights, config)
>>> # adata.obsm['KO_counts'] = sample × KO matrix
>>> # adata.obsm['KO_CLR'] = CLR-transformed matrix
"""

from .ko_matrix import (
    build_sample_ko_matrix,
    load_dram_output,
    clr_transform,
    export_ko_matrix,
)
from .dram_parser import (
    DRAMParser,
    DRAMGene,
    AnnotationSource,
    load_dram_workspace,
)

__all__ = [
    "build_sample_ko_matrix",
    "load_dram_output",
    "clr_transform",
    "export_ko_matrix",
    "DRAMParser",
    "DRAMGene",
    "AnnotationSource",
    "load_dram_workspace",
]
