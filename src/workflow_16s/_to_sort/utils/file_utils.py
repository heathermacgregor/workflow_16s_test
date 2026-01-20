# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import h5py
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.stats.utils import table_to_dataframe

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')  

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N: int = 65 
DEFAULT_GROUP_COLUMN: str = 'nuclear_contamination_status'  
DEFAULT_GROUP_COLUMN_VALUES: List[bool] = [True, False]  

# ANSI color codes for terminal output
RED: str = "\033[91m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"
RESET: str = "\033[0m"

# ==================================== FUNCTIONS ===================================== #   

# ------------------------------- File Operations ------------------------------------ #

def safe_delete(file_path: Union[str, Path]) -> None:
    """
    Safely delete a file if it exists, logging warnings on errors.
    
    Args:
        file_path: Path to the file to be deleted.
    """
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Error deleting {file_path}: {e}")


def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """
    Identify missing files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check for existence.
    
    Returns:
        List of Path objects for files that don't exist.
    """
    return [Path(file) for file in file_list if not Path(file).exists()]


# ------------------------------- Dataset Loading ------------------------------------ #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """
    Load dataset IDs from a text file.
    
    Parses a text file where each line contains a single dataset ID, 
    ignoring empty lines and lines containing only whitespace.
    
    Args:
        path: Path to the dataset list file.
    
    Returns:
        List of dataset ID strings.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load dataset metadata from TSV file.
    
    Args:
        tsv_path: Path to TSV file containing dataset metadata.
    
    Returns:
        DataFrame with dataset information.
    """
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str})
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]


def fetch_first_match(
    dataset: str,
    dataset_info: pd.DataFrame
) -> pd.Series:
    """
    Find the best matching metadata record for a dataset.
    
    Args:
        dataset:      Dataset identifier to search for.
        dataset_info: DataFrame containing dataset metadata.
    
    Returns:
        First matching row as a pandas Series.
    
    Raises:
        ValueError: If no matches found for the dataset.
    """
    mask_ena_type = dataset_info['dataset_type'].str.lower().eq('ena')
    mask_manual_type = dataset_info['dataset_type'].str.lower().eq('manual')
    
    mask_ena = (
        dataset_info['ena_project_accession'].str.contains(dataset, case=False, regex=False) |
        dataset_info['dataset_id'].str.contains(dataset, case=False, regex=False)
    ) & mask_ena_type

    mask_manual = (
        dataset_info['dataset_id'].str.contains(dataset, case=False, regex=False)
    ) & mask_manual_type

    combined_mask = mask_ena | mask_manual
    matching_rows = dataset_info[combined_mask]

    if matching_rows.empty:
        raise ValueError(f"No metadata matches found for dataset: {dataset}")

    return matching_rows.sort_values(
        by='dataset_type', 
        key=lambda x: x.str.lower().map({'ena': 0, 'manual': 1})
    ).iloc[0]


# ------------------------------ File Path Handling ---------------------------------- #

def processed_dataset_files(
    dirs: SubDirs, 
    dataset: str, 
    params: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> Dict[str, Path]:
    """
    Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs:    Project directory structure object.
        dataset: Dataset identifier.
        params:  Processing parameters dictionary.
        cfg:     Configuration dictionary.
    
    Returns:
        Dictionary mapping file types to absolute paths.
    """
    classifier: str = cfg["classifier"]
    base_dir = (
        Path(dirs.qiime_data_per_dataset) / dataset / 
        params['instrument_platform'].lower() / 
        params['library_layout'].lower() / 
        params['target_subfragment'].lower() / 
        f"FWD_{params['pcr_primer_fwd_seq']}_REV_{params['pcr_primer_rev_seq']}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'metadata_tsv': Path(dirs.metadata_per_dataset) / dataset / 'metadata.tsv',
        'manifest_tsv': base_dir / 'manifest.tsv',
        'table_biom': base_dir / 'table' / 'feature-table.biom',
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',
    }


def find_required_qiime_output_files(test: Dict[str, Path]) -> Optional[Dict[str, Path]]:
    """
    Check for required output files in QIIME directories.
    
    Args:
        test: Dictionary containing base paths for QIIME output directories.
    
    Returns:
        Dictionary of found file paths keyed by file type, or None if any 
        required file is missing.
    """
    targets: List[Tuple[str, Optional[str]]] = [
        ("feature-table.biom", "table"),
        ("feature-table.biom", "table_6"),
        ("dna-sequences.fasta", "rep-seqs"),
        ("taxonomy.tsv", "taxonomy"),
        ("sample-metadata.tsv", None)
    ]
    qiime_base: Optional[Path] = test.get('qiime')
    metadata_base: Optional[Path] = test.get('metadata')
    found: Dict[str, Path] = {}
    
    for fname, subdir in targets:
        if subdir:
            base = qiime_base
            pattern = f"{subdir}/{fname}"
        else:
            base = metadata_base
            pattern = fname
            
        if not base:
            continue
            
        for p in Path(base).rglob(pattern):
            if p.is_file():
                key = f"{subdir}/{fname}" if subdir else fname
                found[key] = p.resolve()
                break
                
    required_keys: List[str] = [f"{subdir}/{fname}" if subdir else fname 
                    for fname, subdir in targets]
    return found if all(k in found for k in required_keys) else None


# ------------------------------ Metadata Handling ----------------------------------- #

def import_metadata_tsv(
    tsv_path: Union[str, Path],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Load and standardize a sample metadata TSV file.
    
    Args:
        tsv_path:       Path to metadata TSV file.
        column_renames: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Standardized metadata DataFrame.
    
    Raises:
        FileNotFoundError: If specified path doesn't exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    column_renames = column_renames or []
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.lower()

    sample_id_col = next(
        (col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns),
        None
    )
    df['SAMPLE ID'] = (
        df[sample_id_col] 
        if sample_id_col 
        else [f"{tsv_path.parents[5].name}_x{i}" for i in range(1, len(df)+1)]
    )

    dataset_id_col = next(
        (col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns),
        None
    )
    df['DATASET ID'] = (
        df[dataset_id_col] 
        if dataset_id_col 
        else tsv_path.parents[5].name
    )

    if 'nuclear_contamination_status' not in df.columns:
        df['nuclear_contamination_status'] = False

    for old, new in column_renames:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def import_merged_metadata_tsv(
    meta_paths: List[Union[str, Path]],
    column_renames: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Merge multiple metadata files into a single DataFrame.
    
    Args:
        meta_paths:     List of paths to metadata files.
        column_renames: List of (old_name, new_name) tuples for column renaming.
        verbose:        Enable detailed logging during loading.
    
    Returns:
        Concatenated metadata DataFrame.
    
    Raises:
        FileNotFoundError: If no valid metadata files could be loaded.
    """
    dfs: List[pd.DataFrame] = []

    if verbose:
        for path in meta_paths:
            try:
                df = import_metadata_tsv(path, column_renames)
                dfs.append(df)
                logger.info(f"Loaded {Path(path).name} with {len(df)} samples")
            except Exception as e:
                logger.error(f"Metadata load failed for {path}: {e!r}")
    else:
        with get_progress_bar() as progress:
            task = progress.add_task(
                "Loading metadata files".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(meta_paths)
            )
            for path in meta_paths:
                try:
                    dfs.append(import_metadata_tsv(path, column_renames))
                except Exception as e:
                    logger.error(f"Metadata load failed for {path}: {e!r}")
                finally:
                    progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError(
            "No valid metadata files loaded. Check paths and file formats."
        )

    return pd.concat(dfs, ignore_index=True)


def write_metadata_tsv(
    df: pd.DataFrame, 
    tsv_path: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Write metadata DataFrame to standardized TSV format.
    
    Args:
        df:       Metadata DataFrame.
        tsv_path: Output file path.
        verbose:  Whether to log success message.
    """
    df = df.copy()
    if '#SampleID' not in df.columns and 'run_accession' in df.columns:
        df['#SampleID'] = df['run_accession']
    df.set_index('#SampleID', inplace=True)
    df.to_csv(tsv_path, sep='\t', index=True)
    if verbose:
        logger.info(f"Wrote metadata TSV to '{tsv_path}'")


def manual_meta(
    dataset: str, 
    metadata_dir: Union[str, Path]
) -> pd.DataFrame:
    """
    Load manually curated metadata for a dataset.
    
    Args:
        dataset:      Dataset identifier.
        metadata_dir: Base directory containing metadata files.
    
    Returns:
        DataFrame with manual metadata, or empty DataFrame if not found.
    """
    path = Path(metadata_dir) / dataset / 'manual-metadata.tsv'
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


# -------------------------------- Manifest Handling --------------------------------- #

def write_manifest_tsv(
    seq_paths: Dict[str, List[str]], 
    tsv_path: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Generate QIIME2 manifest file from sequencing file paths.
    
    Args:
        seq_paths: Dictionary mapping sample IDs to file paths.
        tsv_path:  Output file path.
        verbose:   Whether to log success message.
    """
    rows: List[Dict[str, str]] = []
    for sample_id, paths in seq_paths.items():
        if len(paths) == 1:
            rows.append({'sample-id': sample_id, 'absolute-filepath': str(paths[0])})
        elif len(paths) == 2:
            rows.append({
                'sample-id': sample_id,
                'forward-absolute-filepath': str(paths[0]),
                'reverse-absolute-filepath': str(paths[1])
            })
    pd.DataFrame(rows).to_csv(tsv_path, sep='\t', index=False)
    if verbose:
        logger.info(f"Wrote manifest TSV to '{tsv_path}'")


# ------------------------------- BIOM Table Handling -------------------------------- #

def import_table_biom(
    biom_path: Union[str, Path], 
    as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """
    Load a BIOM table from file.
    
    Args:
        biom_path: Path to .biom file.
        as_type:   Output format ('table' or 'dataframe').
    
    Returns:
        BIOM Table object or pandas DataFrame.
    
    Raises:
        ValueError: For invalid as_type values.
    """
    try:
        with h5py.File(biom_path) as f:
            table = Table.from_hdf5(f)
    except:
        table = load_table(biom_path)
        
    if as_type == 'table':
        return table
    elif as_type == 'dataframe':
        return table_to_dataframe(table)
    else:
        raise ValueError(
            f"Invalid output type: {as_type}. Use 'table' or 'dataframe'"
        )


def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """
    Merge multiple BIOM tables into a single unified table.
    
    Args:
        biom_paths: List of paths to .biom files.
        as_type:    Output format ('table' or 'dataframe').
        verbose:    Enable detailed logging during loading.
    
    Returns:
        Merged BIOM Table or DataFrame.
    
    Raises:
        ValueError: If no valid tables are loaded.
    """
    tables: List[Table] = []

    if verbose:
        for path in biom_paths:
            try:
                table = import_table_biom(path, 'table')
                tables.append(table)
                logger.info(f"Loaded {Path(path).name} with {table.shape[1]} samples")
            except Exception as e:
                logger.error(f"BIOM load failed for {path}: {str(e)}")
    else:
        with get_progress_bar() as progress:
            task = progress.add_task(
                "Loading BIOM tables".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(biom_paths))
            for path in biom_paths:
                try:
                    tables.append(import_table_biom(path, 'table'))
                except Exception as e:
                    logger.error(f"BIOM load failed for {path}: {str(e)}")
                finally:
                    progress.update(task, advance=1)

    if not tables:
        raise ValueError("No valid BIOM tables loaded")

    merged_table = reduce(lambda t1, t2: t1.merge(t2), tables)
    return merged_table if as_type == 'table' else table_to_dataframe(merged_table)


# ----------------------------- BIOM-Metadata Alignment ----------------------------- #

def filter_and_reorder_biom_and_metadata(
    table: Table,
    metadata_df: pd.DataFrame,
    sample_column: str = '#sampleid'
) -> Tuple[Table, pd.DataFrame]:
    """
    Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata_df:   Sample metadata DataFrame.
        sample_column: Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    norm_meta = _normalize_metadata(metadata_df, sample_column)
    biom_mapping = _create_biom_id_mapping(table)
    
    shared_ids = [sid for sid in norm_meta[sample_column] if sid in biom_mapping]
    filtered_meta = norm_meta[norm_meta[sample_column].isin(shared_ids)]
    original_ids = [biom_mapping[sid] for sid in filtered_meta[sample_column]]
    
    return table.filter(original_ids, axis='sample', inplace=False), filtered_meta


def _normalize_metadata(
    metadata_df: pd.DataFrame, 
    sample_column: str
) -> pd.DataFrame:
    """
    Normalize sample IDs and remove duplicates.
    
    Args:
        metadata_df: Sample metadata DataFrame.
        sample_column: Column containing sample IDs.
    
    Returns:
        Normalized metadata with lowercase IDs and duplicates removed.
    """
    df = metadata_df.copy()
    df[sample_column] = df[sample_column].astype(str).str.lower()
    return df.drop_duplicates(subset=[sample_column])


def _create_biom_id_mapping(table: Table) -> Dict[str, str]:
    """
    Create lowercase to original-case ID mapping for BIOM table samples.
    
    Args:
        table: BIOM feature table.
    
    Returns:
        Dictionary mapping lowercase IDs to original-case IDs.
    
    Raises:
        ValueError: If duplicate lowercase IDs are detected.
    """
    mapping: Dict[str, str] = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = orig_id.lower()
        if lower_id in mapping:
            raise ValueError(
                f"Duplicate lowercase sample ID: '{lower_id}' "
                f"(from '{orig_id}' and '{mapping[lower_id]}')"
            )
        mapping[lower_id] = orig_id
    return mapping


# -------------------------------- Sequence Handling --------------------------------- #

def import_seqs_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load sequences from FASTA file into a dictionary.
    
    Args:
        fasta_path: Path to FASTA file.
    
    Returns:
        Dictionary mapping sequence IDs to sequences.
    """
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}


# -------------------------------- FAPROTAX Handling --------------------------------- #

def import_faprotax_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load FAPROTAX functional prediction results.
    
    Args:
        tsv_path: Path to FAPROTAX output TSV.
    
    Returns:
        Transposed DataFrame with samples as rows and functions as columns.
    """
    return pd.read_csv(tsv_path, sep="\t", index_col=0).T


# ================================== TAXONOMY CLASS ================================== #

class Taxonomy:
    """
    Handler for taxonomic classification data.
    
    Attributes:
        taxonomy (pd.DataFrame): Parsed taxonomy data with columns:
            - id:              Feature ID
            - taxonomy:        Raw taxonomy string
            - confidence:      Classification confidence score
            - taxstring:       Cleaned taxonomy string
            - [D/P/C/O/F/G/S]: Taxonomic levels (Domain to Species)
    """
    
    def __init__(self, tsv_path: Union[str, Path]) -> None:
        """
        Initialize Taxonomy object from TSV file.
        
        Args:
            tsv_path: Path to QIIME2 taxonomy TSV.
        """
        self.taxonomy: pd.DataFrame = self._import_taxonomy_tsv(tsv_path)
        
    def _import_taxonomy_tsv(self, tsv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Parse taxonomy TSV into structured DataFrame.
        
        Args:
            tsv_path: Path to taxonomy TSV file.
        
        Returns:
            Structured taxonomy DataFrame.
        """
        tsv_path = Path(tsv_path)
        df = pd.read_csv(tsv_path, sep='\t')
        df = df.rename(columns={
            'Feature ID': 'id', 
            'Taxon': 'taxonomy', 
            'Consensus': 'confidence'
        }).set_index('id')
        
        df['taxstring'] = df['taxonomy'].str.replace(r' *[dpcofgs]__', '', regex=True)
        
        for level in ['d', 'p', 'c', 'o', 'f', 'g', 's']:
            df[level.upper()] = df['taxonomy'].apply(
                lambda x: self._extract_level(x, level))
            
        return df.rename(columns={
            'D': 'Domain', 'P': 'Phylum', 'C': 'Class',
            'O': 'Order', 'F': 'Family', 'G': 'Genus', 'S': 'Species'
        })
        
    def _extract_level(
        self, 
        taxonomy: str, 
        level: str
    ) -> Optional[str]:
        """
        Extract specific taxonomic level from taxonomy string.
        
        Args:
            taxonomy: Raw taxonomy string.
            level:    Taxonomic level prefix (d/p/c/o/f/g/s).
        
        Returns:
            Taxonomic name for specified level, or None if not found.
        """
        prefix = level + '__'
        if not taxonomy or taxonomy in ['Unassigned', 'Unclassified']:
            return 'Unclassified'
            
        start = taxonomy.find(prefix)
        if start == -1:
            return None
            
        end = taxonomy.find(';', start)
        return (
            taxonomy[start+len(prefix):end] 
            if end != -1 else 
            taxonomy[start+len(prefix):]
        )
        
    def get_taxstring_by_id(self, feature_id: str) -> Optional[str]:
        """
        Retrieve clean taxonomy string for a feature ID.
        
        Args:
            feature_id: Feature identifier.
        
        Returns:
            Cleaned taxonomy string, or None if feature not found.
        """
        return (
            self.taxonomy.loc[feature_id, 'taxstring'] 
            if feature_id in self.taxonomy.index else 
            None
        )
