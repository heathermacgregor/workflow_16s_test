"""ASV-to-MAG Assignment Engine

Builds sparse weight matrix (ASV × MAG) from VSEARCH alignments.
Handles:
- Single best-hit assignment
- Proportional multi-hit allocation
- Weight normalization
- Sparse matrix construction
"""

import logging
from typing import Optional, Literal, Dict
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path

logger = logging.getLogger("workflow_16s")


class AssignmentEngine:
    """
    Converts VSEARCH alignment results into ASV-to-MAG weight matrix.
    
    Methods:
    - best_hit: Assign each ASV to single best-matching MAG
    - proportional: Distribute ASV counts among all matching MAGs by alignment quality
    """
    
    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.alignment_history = {}  # For validation/debugging
    
    def best_hit(
        self,
        alignments: pd.DataFrame,
        asv_names: list,
        mag_names: list,
        min_pident: float = 99.0,
    ) -> sparse.coo_matrix:
        """
        Assign each ASV to its highest-scoring MAG match (binary matrix).
        
        Args:
            alignments: BLAST-6 format DataFrame from VSEARCH
            asv_names: List of all ASV identifiers
            mag_names: List of all MAG identifiers
            min_pident: Minimum percent identity to keep assignment
            
        Returns:
            Sparse COO matrix (n_asvs, n_mags) with binary weights (0 or 1)
        """
        if alignments.empty:
            self.logger.warning("⚠ No alignments provided for best_hit assignment")
            n_asvs, n_mags = len(asv_names), len(mag_names)
            return sparse.coo_matrix((n_asvs, n_mags))
        
        # Filter by min identity
        filt = alignments[alignments['pident'] >= min_pident].copy()
        if filt.empty:
            self.logger.warning(f"⚠ No alignments >= {min_pident}% identity")
            n_asvs, n_mags = len(asv_names), len(mag_names)
            return sparse.coo_matrix((n_asvs, n_mags))
        
        # Create ID-to-index mappings
        asv_idx = {name: i for i, name in enumerate(asv_names)}
        mag_idx = {name: i for i, name in enumerate(mag_names)}
        
        # For each ASV, take top hit by bitscore
        best_hits = filt.sort_values('bitscore', ascending=False).drop_duplicates('qseqid')
        
        # Build COO matrix
        row_idx = [asv_idx[q] for q in best_hits['qseqid'] if q in asv_idx]
        col_idx = [mag_idx[s] for s in best_hits.loc[best_hits['qseqid'].isin(asv_idx), 'sseqid'] if s in mag_idx]
        
        if len(row_idx) != len(col_idx):
            self.logger.warning("⚠ Index mismatch in best_hit assignment")
            # Truncate to match
            min_len = min(len(row_idx), len(col_idx))
            row_idx, col_idx = row_idx[:min_len], col_idx[:min_len]
        
        data = np.ones(len(row_idx), dtype=np.float32)
        matrix = sparse.coo_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(asv_names), len(mag_names))
        )
        
        self.logger.info(
            f"✓ Best-hit assignment: {matrix.nnz} ASVs → MAGs "
            f"(sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2%})"
        )
        self.alignment_history['best_hit'] = {
            'n_assignments': matrix.nnz,
            'n_asvs': len(asv_names),
            'n_mags': len(mag_names),
        }
        
        return matrix
    
    def proportional(
        self,
        alignments: pd.DataFrame,
        asv_names: list,
        mag_names: list,
        min_pident: float = 99.0,
        weighting: Literal['uniform', 'bitscore', 'pident'] = 'bitscore',
    ) -> sparse.coo_matrix:
        """
        Distribute each ASV's abundance proportionally among all matching MAGs.
        
        Args:
            alignments: BLAST-6 format DataFrame
            asv_names: List of ASV identifiers
            mag_names: List of MAG identifiers
            min_pident: Minimum percent identity to include
            weighting: How to weight multiple hits
                - 'uniform': Equal weight per hit
                - 'bitscore': Weight by bitscore
                - 'pident': Weight by percent identity
            
        Returns:
            Sparse COO matrix (n_asvs, n_mags) with normalized weights
        """
        if alignments.empty:
            n_asvs, n_mags = len(asv_names), len(mag_names)
            return sparse.coo_matrix((n_asvs, n_mags))
        
        # Filter
        filt = alignments[alignments['pident'] >= min_pident].copy()
        if filt.empty:
            n_asvs, n_mags = len(asv_names), len(mag_names)
            return sparse.coo_matrix((n_asvs, n_mags))
        
        # Create mappings
        asv_idx = {name: i for i, name in enumerate(asv_names)}
        mag_idx = {name: i for i, name in enumerate(mag_names)}
        
        # Group by ASV and calculate weights
        row_idx_list = []
        col_idx_list = []
        data_list = []
        
        for asv, group in filt.groupby('qseqid'):
            if asv not in asv_idx:
                continue
            
            # Calculate weights
            if weighting == 'uniform':
                weights = np.ones(len(group))
            elif weighting == 'bitscore':
                weights = group['bitscore'].values
            elif weighting == 'pident':
                weights = group['pident'].values / 100.0  # Normalize to 0-1
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            
            # Normalize to sum to 1 per ASV
            weights = weights / weights.sum()
            
            # Add to matrix
            for (_, row), weight in zip(group.iterrows(), weights):
                if row['sseqid'] in mag_idx:
                    row_idx_list.append(asv_idx[asv])
                    col_idx_list.append(mag_idx[row['sseqid']])
                    data_list.append(weight)
        
        matrix = sparse.coo_matrix(
            (data_list, (row_idx_list, col_idx_list)),
            shape=(len(asv_names), len(mag_names))
        )
        
        self.logger.info(
            f"✓ Proportional assignment ({weighting}): {matrix.nnz} edges "
            f"(sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2%})"
        )
        self.alignment_history['proportional'] = {
            'method': weighting,
            'n_edges': matrix.nnz,
            'n_asvs': len(asv_names),
            'n_mags': len(mag_names),
        }
        
        return matrix
    
    def get_summary(self) -> Dict:
        """Return summary statistics of assignments"""
        return self.alignment_history


def allocate_multiple_hits(
    alignments: pd.DataFrame,
    asv_names: list,
    mag_names: list,
    method: Literal['best_hit', 'proportional'] = 'proportional',
    **kwargs
) -> sparse.coo_matrix:
    """
    Convenience function to allocate multiple hits.
    
    Args:
        alignments: BLAST-6 format DataFrame
        asv_names: List of ASV identifiers
        mag_names: List of MAG identifiers
        method: Assignment method ('best_hit' or 'proportional')
        **kwargs: Additional arguments passed to engine method
        
    Returns:
        Sparse COO matrix (n_asvs, n_mags)
    """
    engine = AssignmentEngine()
    
    if method == 'best_hit':
        return engine.best_hit(alignments, asv_names, mag_names, **kwargs)
    elif method == 'proportional':
        return engine.proportional(alignments, asv_names, mag_names, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
