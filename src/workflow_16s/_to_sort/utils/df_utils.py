# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import re
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from Bio import SeqIO
from biom import load_table
from biom import Table as BiomTable
from scipy import sparse
from scipy.spatial.distance import cdist
from tabulate import tabulate
import matplotlib.pyplot as plt
import logging

# ================================== LOCAL IMPORTS =================================== #

#from workflow_16s.time import timer

# ================================== LOGGER SETUP ==================================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

# Metadata column standardization
COLUMN_ORDER: List[str] = [
    'dataset_id', 'dataset_type', 'ena_project_accession', 'ena_project_description',
    'instrument_platform', 'instrument_model', 'library_layout', 'target_subfragment',
    'sequence_length_bp', 'pcr_primer_fwd', 'pcr_primer_rev', 'pcr_primer_fwd_seq',
    'pcr_primer_rev_seq', 'publication_url', 'principal_investigator', 'dna_extraction_method',
    'sequencing_center', 'nuclear_contamination_status', 'nuclear_contamination_level',
    'nuclear_contamination_source', 'nuclear_contamination_source_type',
    'distance_from_nuclear_contamination_source_m', 'run_accession', 'sample_accession',
    'experiment_accession', 'submission_accession', 'secondary_study_accession',
    'secondary_sample_accession', 'sample_alias', 'run_alias', 'sample_internal_id',
    'sample_description', 'replicate', 'collection_date', 'treatment', 'city',
    'state_or_province', 'country', 'sample_sub_area', 'sample_area', 'sample_site',
    'latitude_deg', 'longitude_deg', 'elevation_m', 'altitude_m', 'depth_m', 'mass_g',
    'temperature_c', 'env_biome', 'env_feature', 'env_material', 'ph'
]

# ENA metadata columns to drop
ENA_METADATA_UNNECCESSARY_COLUMNS: List[str] = [
    'sra_bytes', 'sra_aspera', 'sra_galaxy', 'sra_md5', 'sra_ftp', 
    'fastq_bytes', 'fastq_aspera', 'fastq_galaxy', 'fastq_md5',
    'collection_date_start', 'collection_date_end',
    'location_start', 'location_end',
    'ncbi_reporting_standard',
    'datahub',
    'tax_lineage', 'tax_id', 'scientific_name', 'isolation_source',
    'first_created', 'first_public', 'last_updated', 'status'
]

# ENA metadata columns to rename
ENA_METADATA_COLUMNS_TO_RENAME: Dict[str, str] = {
    'lat': 'latitude_deg',
    'lon': 'longitude_deg'
}

# =================================== DATA UTILS ===================================== #

def table_to_dataframe(table: Union[Dict, BiomTable]) -> pd.DataFrame:
    """
    Convert a BIOM table to pandas DataFrame with samples as rows.
    
    Args:
        table: Input table as either a dictionary or BIOM Table object.
    
    Returns:
        DataFrame with samples as rows and features as columns.
    """
    if isinstance(table, BiomTable):
        df = table.to_dataframe(dense=True)  # features x samples
        return df.T                          # samples  x features
    if isinstance(table, Dict):
        return pd.DataFrame(table)           # samples  x features
    raise TypeError("Input must be BIOM Table or dictionary")


# ================================ METADATA HANDLING ================================ #

def combine_ena_and_manual_metadata(
    ena_meta: pd.DataFrame,
    manual_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge ENA metadata with manual curation, resolving column conflicts.
    
    Steps:
    1. Standardizes column names to lowercase
    2. Checks for required 'run_accession' column
    3. Resolves duplicate columns (manual takes precedence)
    4. Drops unnecessary ENA columns
    5. Renames ENA columns to standardized names
    6. Merges datasets on run_accession
    
    Args:
        ena_meta:    DataFrame from European Nucleotide Archive.
        manual_meta: DataFrame from local user curation.
    
    Returns:
        Merged metadata DataFrame.
    
    Raises:
        ValueError: If required 'run_accession' column is missing.
    """
    # Standardize column names
    ena_meta.columns = ena_meta.columns.str.lower().str.strip()
    manual_meta.columns = manual_meta.columns.str.lower().str.strip()
    
    # Validate required columns
    required = ['run_accession']
    if not set(required).issubset(ena_meta.columns):
        missing = set(required) - set(ena_meta.columns)
        raise ValueError(f"ENA metadata missing required columns: {missing}")
    if not set(required).issubset(manual_meta.columns):
        missing = set(required) - set(manual_meta.columns)
        raise ValueError(f"Manual metadata missing required columns: {missing}")

    def resolve_column_conflicts(
        manual: pd.DataFrame, 
        ena: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle duplicate columns between metadata sources."""
        common = set(ena.columns) & set(manual.columns) - {'run_accession'}
        ena_processed = ena.copy()
        
        for col in common:
            if manual[col].equals(ena[col]):
                ena_processed = ena_processed.drop(columns=col)
            else:
                ena_processed = ena_processed.rename(
                    columns={col: f"{col}_ena"}
                )
        return manual, ena_processed
    
    # Resolve conflicts
    manual_meta, ena_meta = resolve_column_conflicts(manual_meta, ena_meta)
    
    # Clean ENA metadata
    ena_meta = ena_meta.drop(
        columns=ena_meta.columns.intersection(ENA_METADATA_UNNECCESSARY_COLUMNS),
        errors='ignore'
    )
    ena_meta = ena_meta.rename(columns={
        col: new for col, new in ENA_METADATA_COLUMNS_TO_RENAME.items() 
        if col in ena_meta.columns
    })
    
    # Merge datasets
    return manual_meta.merge(ena_meta, on='run_accession', how='left')


def combine_metadata(
    ena_meta: pd.DataFrame,
    manual_meta: pd.DataFrame,
    dataset_id: str
) -> pd.DataFrame:
    """
    Robustly merge ENA and manual metadata with comprehensive validation.
    
    Features:
    - Column name standardization
    - Missing column handling
    - Conflict resolution (manual data priority)
    - Type validation
    - Final column ordering
    
    Args:
        ena_meta:    DataFrame from European Nucleotide Archive.
        manual_meta: DataFrame from local user input.
        dataset_id:  Unique project identifier.
        
    Returns:
        Integrated DataFrame with standardized columns.
    """
    # Standardize column names
    ena_meta.columns = ena_meta.columns.str.lower().str.strip()
    manual_meta.columns = manual_meta.columns.str.lower().str.strip()

    # Add dataset identifier
    for df in [ena_meta, manual_meta]:
        if not df.empty:
            df.insert(0, 'dataset_id', dataset_id)

    # Ensure required columns exist
    required = set(COLUMN_ORDER)
    for df in [ena_meta, manual_meta]:
        for col in required - set(df.columns):
            df[col] = pd.NA  # Add missing columns

    # Merge with conflict resolution
    merged = pd.merge(
        ena_meta,
        manual_meta,
        on='run_accession',
        how='outer',
        suffixes=('_ena', '_manual'),
        indicator=True
    )

    # Resolve column conflicts
    conflict_cols = set(ena_meta.columns) & set(manual_meta.columns) - {'run_accession'}
    for col in conflict_cols:
        merged[col] = merged[f"{col}_manual"].combine_first(merged[f"{col}_ena"])
        merged = merged.drop(columns=[f"{col}_ena", f"{col}_manual"])

    # Final validation and cleanup
    for col in required - set(merged.columns):
        merged[col] = pd.NA
        
    merged['run_accession'] = merged['run_accession'].astype(str)
    merged['dataset_id'] = merged['dataset_id'].astype(str)
    
    return merged[COLUMN_ORDER].dropna(axis=1, how='all')


def get_first_existing_column(
    df: pd.DataFrame, 
    columns: List[str]
) -> Optional[pd.Series]:
    """
    Retrieve the first existing column from a list of candidates.
    
    Args:
        df:      DataFrame to search.
        columns: Ordered list of column names to try.
        
    Returns:
        Series from first existing column, or None if none found.
    """
    for col in columns:
        if col in df.columns:
            return df[col]
    return None


# ============================== DATA ALIGNMENT UTILS ================================ #

def match_indices_or_transpose(
    df1: pd.DataFrame, 
    df2: Union[pd.DataFrame, BiomTable]
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Align DataFrames by index or transpose to find matches.
    
    Checks for index matches between:
    - df1.index and df2.index
    - df1.index and df2.columns (after transpose)
    
    Args:
        df1: Primary DataFrame with index to match.
        df2: Secondary DataFrame or BIOM table.
        
    Returns:
        Tuple: 
            - df1 (unchanged)
            - Aligned df2 (possibly transposed)
            - Boolean indicating if transpose occurred
    """
    # Convert BIOM tables to DataFrame
    if not isinstance(df2, pd.DataFrame):
        df2 = table_to_dataframe(df2)
        
    # Ensure df1 has sample IDs in index
    if '#sampleid' in df1.columns:
        df1 = df1.set_index('#sampleid')
    
    # Check direct index match
    if df1.index.intersection(df2.index).any():
        return df1, df2, False

    # Try transposing df2
    df2_t = df2.T
    if df1.index.intersection(df2_t.index).any():
        return df1, df2_t, True

    # No matches found
    return df1, df2, False


def check_matching_index(
    metadata: pd.DataFrame, 
    features: pd.DataFrame
) -> bool:
    """
    Check for overlapping indices between metadata and features.
    
    Args:
        metadata: DataFrame with sample metadata.
        features: Feature table (samples x features).
        
    Returns:
        True if any matching indices found, False otherwise.
    """
    samples = set(metadata.index)
    return samples.intersection(features.index) or samples.intersection(features.columns)


def match_samples(
    metadata: pd.DataFrame, 
    features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter and align metadata and features to common samples.
    
    Args:
        metadata: Sample metadata.
        features: Feature table (samples x features).
        
    Returns:
        Tuple of aligned (metadata, features) DataFrames.
    """
    common = metadata.index.intersection(features.columns)
    return metadata.loc[common], features[common]


# ============================== FEATURE PROCESSING ================================== #

def classify_feature_format(columns: Iterable[str]) -> Dict[str, int]:
    """
    Classify feature IDs into taxonomic, hash, sequence, or unknown types.
    
    Uses regex patterns to identify:
    - Taxonomic strings (e.g., 'd__Bacteria;p__Firmicutes')
    - QIIME-style hashes (32/64 character hex strings)
    - IUPAC nucleotide sequences
    - Unknown patterns
    
    Args:
        columns: Feature IDs to classify.
        
    Returns:
        Dictionary with counts for each category.
    """
    patterns = {
        "taxonomic": re.compile(
            r'^d__[\w]+(;p__[\w]+)?(;c__[\w]+)?(;o__[\w]+)?'
            r'(;f__[\w]+)?(;g__[\w]+)?(;s__[\w]+)?$'
        ),
        "hashes": re.compile(r'^[a-f0-9]{32}$|^[a-f0-9]{64}$'),
        "raw_sequences": re.compile(r'^[ACGTRYSWKMBDHVN]+$', re.IGNORECASE)
    }
    
    counts = {k: 0 for k in patterns}
    counts["unknown"] = 0
    
    for col in map(str, columns):
        col = col.strip()
        if not col:
            counts["unknown"] += 1
            continue
            
        matched = False
        for name, pattern in patterns.items():
            if pattern.match(col):
                counts[name] += 1
                matched = True
                break
                
        if not matched:
            counts["unknown"] += 1
            
    # Print classification summary
    total = sum(counts.values())
    if total > 0:
        dominant = max(counts, key=counts.get)
        confidence = counts[dominant] / total
        logger.info(
            f"Feature classification: {dominant} "
            f"({confidence:.0%} confidence)"
        )
        
    return counts


def trim_and_merge_asvs(
    asv_table: pd.DataFrame,
    asv_sequences: List[str],
    trim_length: int = 250
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Trim ASV sequences and merge identical trimmed variants.
    
    Optimized workflow:
    1. Parallel sequence trimming
    2. BIOM table creation with temporary feature IDs
    3. Collapse by trimmed sequences
    4. Convert back to DataFrame
    
    Args:
        asv_table:        Feature table (features x samples).
        asv_sequences:    Raw sequences corresponding to features.
        trim_length:      Number of bases to keep from sequence start.
        
    Returns:
        Tuple:
            - Merged feature table (features x samples)
            - Series of trimmed sequences
            - Temporary feature IDs
            
    Raises:
        ValueError: For dimension mismatch or invalid trim length.
    """
    # Validate inputs
    if len(asv_sequences) != asv_table.shape[0]:
        raise ValueError(
            f"ASV count mismatch: "
            f"{len(asv_sequences)} sequences vs {asv_table.shape[0]} features"
        )
    if trim_length < 1:
        raise ValueError(f"Invalid trim length: {trim_length}")
    
    # Initialize parallel processing
    pandarallel.initialize(progress_bar=False, nb_workers=8)
    
    # Trim sequences in parallel
    trimmed_seqs = (
        pd.Series(asv_sequences)
        .parallel_apply(lambda x: x[:trim_length])
    )
    trimmed_seqs.index = asv_sequences

    # Create BIOM table with temporary feature IDs
    obs_ids = [f"TMP_FEATURE_{i}" for i in range(len(asv_sequences))]
    biom_table = BiomTable(
        data=asv_table.values.astype(np.uint32),
        observation_ids=obs_ids,
        sample_ids=asv_table.columns.tolist(),
        observation_metadata=[{'trimmed_seq': s} for s in trimmed_seqs]
    )

    # Collapse features by trimmed sequences
    merged_biom = biom_table.collapse(
        lambda id_, md: md['trimmed_seq'],
        axis='observation',
        norm=False,
        min_group_size=1,
        include_collapsed_metadata=False
    )

    # Convert to DataFrame
    merged_df = merged_biom.to_dataframe(dense=True).astype(np.uint32)
    
    # Log reduction statistics
    orig = len(asv_sequences)
    new = merged_df.shape[0]
    logger.info(
        f"Feature reduction: {orig} → {new} "
        f"({new/orig:.1%}) after {trim_length}bp trim"
    )
    
    return merged_df, trimmed_seqs, obs_ids
    
