"""KO Matrix Construction & CLR Transformation

Efficient sparse matrix operations for functional profiling:
1. Load MAG → KO assignments from DRAM output
2. Multiply: (samples × ASVs) @ (ASVs × MAGs) @ (MAGs × KOs) = samples × KOs
3. Apply CLR transformation with pseudocount handling
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import anndata
import pandas as pd
import numpy as np
from scipy import sparse
import h5py

logger = logging.getLogger("workflow_16s")


def load_dram_output(
    dram_dir: Path,
    format: str = "tsv",
    logger_obj=None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load DRAM gene-to-KO annotations.
    
    DRAM outputs:
    - annotations.tsv: gene_id → KO, EC, etc.
    - genes.gff: Gene coordinates
    
    Args:
        dram_dir: Path to DRAM output directory
        format: "tsv" or "h5" 
        logger_obj: Logger instance
        
    Returns:
        (DataFrame with MAG × KO, List of MAG identifiers)
    """
    logger_obj = logger_obj or logger
    dram_dir = Path(dram_dir)
    
    if not dram_dir.exists():
        raise FileNotFoundError(f"DRAM directory not found: {dram_dir}")
    
    # Find annotations file
    annotations_file = dram_dir / "annotations.tsv"
    if not annotations_file.exists():
        raise FileNotFoundError(f"DRAM annotations not found: {annotations_file}")
    
    logger_obj.info(f"📖 Loading DRAM output from {dram_dir}")
    
    try:
        # Load with minimal columns: gene_id, KO
        df = pd.read_csv(
            annotations_file,
            sep='\t',
            usecols=['gene_id', 'ko_id'],
            dtype={'gene_id': str, 'ko_id': str},
            na_values=['', 'nan', '-']
        )
        
        # Extract MAG from gene_id (typically format: MAG_NAME_*geneID)
        df['mag_id'] = df['gene_id'].str.split('_', expand=True)[0]
        
        # Remove genes without KO assignment
        df = df.dropna(subset=['ko_id'])
        
        logger_obj.info(f"✓ Loaded {len(df)} gene-KO pairs from {len(df['mag_id'].unique())} MAGs")
        
        return df, list(df['mag_id'].unique())
        
    except Exception as e:
        logger_obj.error(f"❌ Failed to parse DRAM output: {e}")
        raise


def clr_transform(
    matrix: Union[np.ndarray, sparse.spmatrix],
    pseudocount: float = 0.5,
    zero_replacement: str = "pseudocount",
    logger_obj=None,
) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Apply Centered Log-Ratio (CLR) transformation.
    
    CLR(x_i) = log(x_i) - mean(log(x))
    
    Handles:
    - Zero values (pseudocount replacement or geometric mean)
    - Sparse matrices (preserves sparsity)
    
    Args:
        matrix: Count matrix (samples × features)
        pseudocount: Value to add to zeros
        zero_replacement: "pseudocount" or "geometric_mean"
        logger_obj: Logger instance
        
    Returns:
        CLR-transformed matrix (same sparsity structure)
    """
    logger_obj = logger_obj or logger
    is_sparse = sparse.issparse(matrix)
    
    if is_sparse:
        # Convert to lil_matrix for efficient operations
        matrix = matrix.tolil()
        n_samples, n_features = matrix.shape
        
        # Add pseudocount to all (non-zero) values
        # Note: This is composition-aware
        clr_matrix = matrix.copy().astype(np.float32)
        clr_matrix.data += pseudocount
        
        # Compute geometric mean per sample
        log_matrix = clr_matrix.copy()
        log_matrix.data = np.log(log_matrix.data)
        row_means = np.array(log_matrix.mean(axis=1)).ravel()
        
        # Subtract row means (centering)
        for i in range(n_samples):
            clr_matrix[i, :] -= row_means[i]
        
        return clr_matrix.tocoo()
    else:
        # Dense array
        matrix = matrix.astype(np.float32).copy()
        
        # Add pseudocount
        matrix[matrix == 0] = pseudocount
        matrix[matrix < pseudocount] = pseudocount
        
        # Log transformation
        log_matrix = np.log(matrix)
        
        # Center by geometric mean
        row_means = log_matrix.mean(axis=1, keepdims=True)
        clr_matrix = log_matrix - row_means
        
        return clr_matrix


def build_sample_ko_matrix(
    adata: anndata.AnnData,
    asv_mag_weights: sparse.coo_matrix,
    dram_dir: Path,
    config: Optional[dict] = None,
    pseudocount: float = 0.5,
    logger_obj=None,
) -> anndata.AnnData:
    """
    Build sample × KO abundance matrix.
    
    Pipeline:
    1. Load MAG → KO assignments from DRAM
    2. Build MAG × KO sparse matrix
    3. Multiply: (samples × ASVs) @ (ASVs × MAGs) @ (MAGs × KOs)
    4. Apply CLR transformation
    5. Store in adata.obsm
    
    Args:
        adata: AnnData object with ASV counts in .X
        asv_mag_weights: Sparse matrix from asv_mag_mapping (ASVs × MAGs)
        dram_dir: Path to DRAM output directory
        config: Configuration dict (optional)
        pseudocount: Pseudocount for CLR transformation
        logger_obj: Logger instance
        
    Returns:
        Updated AnnData with .obsm['KO_counts'] and .obsm['KO_CLR']
    """
    logger_obj = logger_obj or logger
    
    logger_obj.info("🔧 Building sample × KO matrix...")
    
    # 1. Load DRAM output
    dram_df, mag_list = load_dram_output(dram_dir, logger_obj=logger_obj)
    
    # 2. Build MAG × KO binary matrix
    mag_idx = {m: i for i, m in enumerate(mag_list)}
    ko_list = list(dram_df['ko_id'].unique())
    ko_idx = {k: i for i, k in enumerate(ko_list)}
    
    mag_ko_rows = [mag_idx[m] for m in dram_df['mag_id']]
    mag_ko_cols = [ko_idx[k] for k in dram_df['ko_id']]
    mag_ko_data = np.ones(len(dram_df), dtype=np.float32)
    
    mag_ko = sparse.coo_matrix(
        (mag_ko_data, (mag_ko_rows, mag_ko_cols)),
        shape=(len(mag_list), len(ko_list))
    )
    logger_obj.debug(f"  MAG × KO matrix: {mag_ko.shape}, {mag_ko.nnz} edges")
    
    # 3. Matrix multiplication: samples × ASVs @ ASVs × MAGs @ MAGs × KOs
    logger_obj.info("  🔢 Multiplying matrices...")
    
    # Ensure ASV-MAG matrix matches
    if adata.n_vars != asv_mag_weights.shape[0]:
        raise ValueError(
            f"ASV count mismatch: adata has {adata.n_vars} ASVs, "
            f"but asv_mag_weights has {asv_mag_weights.shape[0]}"
        )
    
    # (n_samples, n_asvs) @ (n_asvs, n_mags)
    sample_mag = adata.X @ asv_mag_weights.tocsr()
    logger_obj.debug(f"  sample × MAG: {sample_mag.shape}")
    
    # (n_samples, n_mags) @ (n_mags, n_kos)
    sample_ko = sample_mag @ mag_ko.tocsr()
    logger_obj.debug(f"  sample × KO: {sample_ko.shape}")
    
    # 4. CLR transformation
    logger_obj.info("  📊 Applying CLR transformation...")
    sample_ko_clr = clr_transform(
        sample_ko,
        pseudocount=pseudocount,
        logger_obj=logger_obj
    )
    
    # 5. Store in AnnData
    adata.obsm['KO_counts'] = sample_ko.toarray() if sparse.issparse(sample_ko) else sample_ko
    adata.obsm['KO_CLR'] = sample_ko_clr.toarray() if sparse.issparse(sample_ko_clr) else sample_ko_clr
    
    # Store feature names
    adata.varm['KO_list'] = ko_list
    
    logger_obj.info(
        f"✅ Built sample × KO matrix: {adata.obsm['KO_counts'].shape}"
    )
    
    return adata


def export_ko_matrix(
    matrix: np.ndarray,
    ko_names: list,
    sample_names: list,
    output_file: Path,
    format: str = "csv",
) -> None:
    """
    Export KO matrix to file.
    
    Args:
        matrix: Sample × KO matrix (n × m)
        ko_names: List of KO identifiers
        sample_names: List of sample identifiers
        output_file: Path to output file
        format: "csv", "hdf5", or "npz"
    """
    output_file = Path(output_file)
    
    if format == "csv":
        df = pd.DataFrame(matrix, index=sample_names, columns=ko_names)
        df.to_csv(output_file)
    elif format == "hdf5":
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('data', data=matrix)
            f.create_dataset('ko_names', data=np.array(ko_names, dtype='S'))
            f.create_dataset('sample_names', data=np.array(sample_names, dtype='S'))
    elif format == "npz":
        np.savez_compressed(
            output_file,
            data=matrix,
            ko_names=ko_names,
            sample_names=sample_names
        )
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"✅ Exported KO matrix to {output_file}")
