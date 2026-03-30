# Utility functions for sequence file I/O operations

from pathlib import Path
from typing import Dict, List
import pandas as pd
import shutil


def write_manifest_tsv(seq_paths: Dict[str, List[Path]], manifest_path: Path, library_layout: str = 'paired') -> None:
    """
    Write a QIIME 2 compatible manifest TSV file.
    
    Args:
        seq_paths: Dictionary mapping sample IDs to list of sequence file paths.
                   For paired-end: {sample_id: [forward_path, reverse_path]}
                   For single-end: {sample_id: [forward_path]}
        manifest_path: Path where to write the manifest file.
        library_layout: 'paired' for paired-end or 'single' for single-end sequences.
    """
    manifest_path = Path(manifest_path)
    library_layout = library_layout.lower()
    
    # Validate and filter out entries with missing paths
    valid_entries = []
    for sample_id, paths in seq_paths.items():
        if not paths or len(paths) < 1:
            continue
        
        if library_layout == 'paired':
            # For paired-end sequences: require both forward and reverse
            if len(paths) >= 2:
                fwd_path, rev_path = paths[0], paths[1]
                if fwd_path and rev_path:
                    valid_entries.append({
                        'sample-id': sample_id,
                        'forward-absolute-filepath': str(fwd_path.resolve()),
                        'reverse-absolute-filepath': str(rev_path.resolve())
                    })
        else:  # single-end
            # For single-end sequences: only forward filepath
            fwd_path = paths[0]
            if fwd_path:
                valid_entries.append({
                    'sample-id': sample_id,
                    'absolute-filepath': str(fwd_path.resolve())
                })
    
    # Write the manifest file
    if valid_entries:
        df = pd.DataFrame(valid_entries)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, sep='\t', index=False)
    else:
        raise ValueError(f"No valid sequence paths found to write manifest for {library_layout}-end sequences")


def write_metadata_tsv(metadata: pd.DataFrame, metadata_path: Path) -> None:
    """
    Write metadata to a TSV file.
    
    Args:
        metadata: Pandas DataFrame containing metadata.
        metadata_path: Path where to write the metadata file.
    """
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, sep='\t', index=True)


def safe_delete(path: Path) -> None:
    """
    Safely delete a file or directory.
    
    Args:
        path: Path to file or directory to delete.
    """
    path = Path(path)
    if not path.exists():
        return
    
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def import_table_biom(biom_path: Path) -> dict:
    """
    Import a BIOM table (placeholder - actual implementation would use biom package).
    
    Args:
        biom_path: Path to BIOM file.
        
    Returns:
        Dictionary representation of BIOM table.
    """
    try:
        import biom
        return biom.load_table(str(biom_path))
    except ImportError:
        raise ImportError("biom package is required to import BIOM files")


def import_metadata_tsv(metadata_path: Path) -> pd.DataFrame:
    """
    Import metadata from a TSV file.
    
    Args:
        metadata_path: Path to metadata TSV file.
        
    Returns:
        Pandas DataFrame containing metadata.
    """
    return pd.read_csv(metadata_path, sep='\t', index_col=0, low_memory=False)


def dataset_first_match(metadata: pd.DataFrame, column: str, value: str) -> dict:
    """
    Find the first row in metadata matching a column value.
    
    Args:
        metadata: Pandas DataFrame containing metadata.
        column: Column name to search.
        value: Value to match.
        
    Returns:
        Dictionary representation of matching row, or None if no match found.
    """
    mask = metadata[column] == value
    if mask.any():
        return metadata[mask].iloc[0].to_dict()
    return None
