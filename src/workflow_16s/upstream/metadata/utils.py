# workflow_16s/upstream/metadata/utils.py

# Standard Library Imports
import io
import re
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-Party Imports
import anndata as ad
import biom
import pandas as pd
import skbio
from skbio import TreeNode

# Local Imports
# Assuming get_logger is appropriately defined elsewhere (e.g., workflow_16s.utils.logger)
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ================================ AnnData UTILITIES ================================= #

def create_anndata_from_qiime_artifacts(
    feature_table_biom_path: Path,
    taxonomy_tsv_path: Path,
    rep_seqs_fasta_path: Path,
    rooted_tree_nwk_path: Path,
    metadata_path: Path
) -> ad.AnnData:
    """
    Loads exported QIIME 2 files and creates a comprehensive AnnData object.
    The phylogenetic tree is serialized to a Newick string for file compatibility.
    """
    logger.info("Starting AnnData object creation process from QIIME 2 artifacts.")

    # 1. Load BIOM Table (Counts)
    logger.info(f"--> Step 1: Loading BIOM feature table from: {feature_table_biom_path}")
    biom_table = biom.load_table(str(feature_table_biom_path))
    table_df = biom_table.to_dataframe(dense=True)
    logger.info(f"       ...Loaded table with {table_df.shape[0]} features and {table_df.shape[1]} samples.")

    # 2. Load Taxonomy
    logger.info(f"--> Step 2: Loading taxonomy data from: {taxonomy_tsv_path}")
    tax_df = pd.read_csv(taxonomy_tsv_path, sep='\t', index_col=0)
    tax_df.index.name = "feature-id"
    logger.info(f"       ...Loaded taxonomy for {len(tax_df)} features.")

    # 3. Create initial AnnData object (observations x variables)
    logger.info("--> Step 3: Transposing table and creating initial AnnData object.")
    adata = ad.AnnData(X=table_df.T)
    logger.info(f"       ...Created AnnData object with shape: {adata.n_obs} samples (obs) x {adata.n_vars} features (vars).")

    # 4. Add Sample Metadata to .obs
    logger.info(f"--> Step 4: Loading and attaching sample metadata to .obs from: {metadata_path}")
    if metadata_path.exists():
        original_adata_ids = adata.obs_names.copy()
        # Ensure index_col=0 to use the first column (sample IDs) as index
        sample_metadata = pd.read_csv(metadata_path, sep='\t', index_col=0, dtype=str)
        sample_metadata.dropna(how='all', axis=1, inplace=True) # Drop fully empty columns
        logger.info(f"       ...Loaded metadata for {sample_metadata.shape[0]} samples and {sample_metadata.shape[1]} columns.")

        original_obs_count = adata.n_obs
        # Reindex metadata to match AnnData's observation names (sample IDs)
        adata.obs = sample_metadata.reindex(adata.obs_names)
        # Check how many samples in AnnData found a match in the metadata
        matches = adata.obs.notna().any(axis=1).sum()
        logger.info(f"       ...Aligned metadata. {matches} of {original_obs_count} samples had matching metadata.")

        if matches == 0 and original_obs_count > 0:
            logger.error("CRITICAL: Sample ID mismatch detected. No samples in the feature table could be matched with the metadata file.")
            logger.error(f"       First 5 sample IDs from feature table (biom): {original_adata_ids[:5].tolist()}")
            logger.error(f"       First 5 sample IDs from metadata file (.tsv): {sample_metadata.index[:5].tolist()}")
            raise ValueError("Could not align sample metadata. Check sample IDs in BIOM table and metadata file.")

        logger.info("       ...Ensuring all sample metadata columns are string-formatted for compatibility.")
        # Fill NaNs with empty string and convert object columns to string
        for col in adata.obs.select_dtypes(include=['object']).columns:
            adata.obs[col] = adata.obs[col].fillna('').astype(str)
        # Also convert any remaining non-numeric, non-string columns if necessary
        for col in adata.obs.columns:
            if adata.obs[col].dtype not in ['float64', 'int64', 'bool', 'str']:
                adata.obs[col] = adata.obs[col].astype(str)

    # 5. Add Taxonomy to .var
    logger.info("--> Step 5: Attaching feature taxonomy data to .var")
    original_var_count = adata.n_vars
    # Join taxonomy data, aligning by feature ID (index)
    adata.var = adata.var.join(tax_df.reindex(adata.var_names))
    # Check how many features got taxonomy info
    tax_col = 'Taxon' # Default QIIME taxonomy column name
    if tax_col in adata.var.columns:
        matches = adata.var[tax_col].notna().sum()
        logger.info(f"       ...Aligned taxonomy. {matches} of {original_var_count} features had matching taxonomy.")
    else:
        logger.warning(f"       ...Taxonomy column '{tax_col}' not found after join.")

    # 6. Add Representative Sequences to .var
    logger.info(f"--> Step 6: Loading and attaching representative sequences to .var from: {rep_seqs_fasta_path}")
    try:
        seqs = {seq.metadata['id']: str(seq) for seq in skbio.read(str(rep_seqs_fasta_path), format='fasta')}
        logger.info(f"       ...Parsed {len(seqs)} sequences from FASTA file.")
        # Create a pandas Series, reindex to match AnnData's variable names (feature IDs)
        seq_series = pd.Series(seqs, name="sequence").reindex(adata.var_names)
        adata.var['sequence'] = seq_series
        matches = adata.var['sequence'].notna().sum()
        logger.info(f"       ...Attached sequences. {matches} of {original_var_count} features had a matching sequence.")
    except Exception as e:
        logger.error(f"       ...Error reading or processing representative sequences FASTA file: {e}")
        adata.var['sequence'] = pd.NA # Add column but indicate failure

    # 7. Add Phylogenetic Tree to .uns
    logger.info(f"--> Step 7: Loading and serializing phylogenetic tree to .uns from: {rooted_tree_nwk_path}")
    if not rooted_tree_nwk_path.exists():
        logger.warning(f"       ...Phylogenetic tree file not found at {rooted_tree_nwk_path}. This may indicate tree building failed in QIIME 2.")
        logger.warning("       ...Downstream phylogenetic diversity analysis will not be available for this dataset.")
        adata.uns['phylogenetic_tree'] = None
    else:
        try:
            phylogenetic_tree = TreeNode.read(str(rooted_tree_nwk_path), format='newick')

            # Serialize tree to Newick string format
            with io.StringIO() as fh:
                phylogenetic_tree.write(fh, format='newick')
                newick_string = fh.getvalue()
            adata.uns['phylogenetic_tree'] = newick_string
            logger.info("       ...Successfully stored phylogenetic tree as a Newick string in adata.uns['phylogenetic_tree'].")
        except Exception as e:
            logger.error(f"       ...Error reading or processing phylogenetic tree file: {e}")
            adata.uns['phylogenetic_tree'] = None # Store None to indicate failure

    logger.info("--> Final Step: Ensuring all feature metadata columns are string-formatted for compatibility.")
    # Fill NaNs and convert object columns to string in .var
    for col in adata.var.select_dtypes(include=['object']).columns:
        adata.var[col] = adata.var[col].fillna('').astype(str)
    # Also convert any remaining non-numeric, non-string columns if necessary
    for col in adata.var.columns:
        if adata.var[col].dtype not in ['float64', 'int64', 'bool', 'str']:
            adata.var[col] = adata.var[col].astype(str)

    logger.info("✅ AnnData object creation complete.")
    return adata


def validate_anndata_file(anndata_path: Path, subset_id: str):
    """
    Performs quality control checks on a newly created AnnData file.
    Checks for essential components like obs/var indices, taxonomy, sequences, and tree.
    """
    logger.info(f"Starting AnnData validation for subset '{subset_id}'...")
    errors = []
    try:
        adata = ad.read_h5ad(anndata_path)

        # Basic structure checks
        if adata.n_obs == 0 or adata.n_vars == 0:
            errors.append("AnnData object is empty (n_obs=0 or n_vars=0).")
        # Check obs (samples)
        if not adata.obs.index.is_unique:
            errors.append("Sample IDs in '.obs.index' are not unique.")
        # Check if 'run_accession' exists, crucial for linking back
        if 'run_accession' not in adata.obs.columns:
            # If not present, check if the index itself looks like run accessions
            # This is a fallback check, ideally 'run_accession' column should exist
             if not all(re.match(r'[ESD]RR\d+', str(idx)) for idx in adata.obs.index):
                 errors.append("'.obs' is missing the 'run_accession' column, and index does not appear to be run accessions.")


        # Check var (features)
        if not adata.var.index.is_unique:
             errors.append("Feature IDs in '.var.index' are not unique.")
        if 'Taxon' not in adata.var.columns: # Assuming 'Taxon' is the standard QIIME output column name
            errors.append("'.var' is missing the 'Taxon' column for taxonomy.")
        if 'sequence' not in adata.var.columns:
            errors.append("'.var' is missing the 'sequence' column for representative sequences.")
        elif adata.var['sequence'].isnull().any():
            errors.append("The 'sequence' column in '.var' contains null/missing values.")
        """
        # Check uns (unstructured data - tree)
        if 'phylogenetic_tree' not in adata.uns:
            errors.append("'.uns' is missing the 'phylogenetic_tree'.")
        else:
            tree_data = adata.uns['phylogenetic_tree']
            if not isinstance(tree_data, str) or not tree_data: # Check if it's a non-empty string
                errors.append(f"'.uns['phylogenetic_tree']' should be a non-empty string, but found type {type(tree_data)}.")
            else:
                # Validate if the string is parseable as Newick
                try:
                    TreeNode.read(io.StringIO(tree_data))
                except Exception as e:
                    errors.append(f"'.uns['phylogenetic_tree']' is not a valid Newick string. Parser error: {e}")
        """
        if 'phylogenetic_tree' in adata.uns:
            if adata.uns['phylogenetic_tree'] is not None:
                logger.info("  - Validation: Found valid 'phylogenetic_tree' in .uns.")
                # You could add more checks here, e.g., type(adata.uns['phylogenetic_tree'])
            else:
                logger.info("  - Validation: 'phylogenetic_tree' is present but None (accepted).")
        else:
            # This is the case you are hitting. It's now accepted.
            logger.info("  - Validation: No 'phylogenetic_tree' found in .uns (accepted).")
            
    except Exception as e:
        errors.append(f"Failed to read or perform basic validation on the H5AD file: {e}")

    # Report errors or success
    if errors:
        error_summary = "\n - ".join(errors)
        raise ValueError(f"AnnData validation failed for '{subset_id}' ({anndata_path}):\n - {error_summary}")
    else:
        logger.info(f"✅ AnnData file for '{subset_id}' passed all validation checks.")


# ================================== CACHE MANAGER =================================== #

class PartitionCacheManager:
    """Manages a persistent SQLite cache for dataset and run processing status."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary database tables if they don't exist."""
        with self.conn: # Use connection as context manager for transactions
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS run_status (
                    run_accession TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL, /* e.g., SUCCESS, FAILED_ANALYSIS, FAILED_PARTITIONING, UNDETERMINED */
                    predicted_region TEXT, /* e.g., V4, V3-V4 */
                    report_yaml TEXT,      /* YAML dump of the analysis report */
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Index for faster lookup by dataset
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_id ON run_status (dataset_id)")

    def add_failed_dataset(self, dataset_id: str, reason: str):
        """Adds or updates a dataset marked as failed."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO failed_datasets (dataset_id, reason) VALUES (?, ?)", (dataset_id, reason))
        logger.info(f"Cached dataset '{dataset_id}' as failed. Reason: {reason}")

    def is_dataset_failed(self, dataset_id: str) -> bool:
        """Checks if a dataset is marked as failed in the cache."""
        cur = self.conn.execute("SELECT 1 FROM failed_datasets WHERE dataset_id = ?", (dataset_id,))
        return cur.fetchone() is not None

    def get_dataset_run_statuses(self, dataset_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieves all cached run statuses for a specific dataset."""
        cur = self.conn.execute("SELECT run_accession, status, predicted_region FROM run_status WHERE dataset_id = ?", (dataset_id,))
        # Return a dictionary mapping run_accession to its status info
        return {row['run_accession']: dict(row) for row in cur.fetchall()}

    def add_run_status(
        self, run_accession: str, dataset_id: str, status: str,
        predicted_region: Optional[str] = None, report_yaml: Optional[str] = None
    ):
        """Adds or updates the status of a specific run."""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO run_status
                   (run_accession, dataset_id, status, predicted_region, report_yaml, timestamp)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (run_accession, dataset_id, status, predicted_region, report_yaml)
            )
        # Optional: Add logging if needed, but might be verbose if called often
        # logger.debug(f"Cached status for run '{run_accession}' (Dataset: {dataset_id}): {status}")


    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Partition cache database connection closed.")


# ================================== HELPER FUNCTIONS ================================== #

def format_bytes(size: int) -> str:
    """Converts bytes to a human-readable string (KB, MB, GB)."""
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024**2:
        return f"{size/1024:.2f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.2f} MB"
    else:
        return f"{size/1024**3:.2f} GB"

def display_and_save_summary_table(results_df: pd.DataFrame, title: str, output_path: Path):
    """Displays a DataFrame using rich table and saves as HTML, falls back to text."""
    if results_df.empty:
        logger.info(f"No results to display or save for '{title}'.")
        return
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console(record=True) # Enable recording for saving
        table = Table(show_header=True, header_style="bold cyan", title=title)
        # Add columns dynamically from DataFrame
        for column in results_df.columns:
            table.add_column(str(column), justify="left")
        # Add rows, converting all items to string
        for _, row in results_df.iterrows():
            table.add_row(*[str(item) for item in row.values])

        console.print(table) # Print to console

        # Save as HTML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        console.save_html(str(output_path))
        logger.info(f"Saved summary table '{title}' to {output_path}")

    except ImportError:
        logger.warning("'rich' library not installed. Falling back to plain text display for summary table.")
        print(f"\n--- {title} ---")
        print(results_df.to_string()) # Print full DataFrame to console
        # Optionally save as CSV as a fallback?
        # output_csv_path = output_path.with_suffix('.csv')
        # results_df.to_csv(output_csv_path, index=False)
        # logger.info(f"Saved summary table '{title}' as CSV to {output_csv_path}")


def is_host_associated(text: str, keywords: List[str]) -> Optional[str]:
    """
    Checks if text contains any host-associated keywords (whole word match).
    Returns the specific keyword found, or None if no match. Case-insensitive.
    """
    if not text or not keywords: # Check for empty inputs
        return None
    # Compile a single regex pattern: \b(word1|word2|word3)\b
    # \b ensures whole word match. re.escape handles special characters in keywords.
    try:
        pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1) # Return the actual matched keyword (maintains case if needed later)
    except re.error as e:
        logger.error(f"Regex error in is_host_associated with keywords: {keywords[:5]}... Error: {e}")
    return None

def find_keyword_matches(df, keywords):
    """
    Searches DataFrame columns for keywords (whole words, case-insensitive).
    Optimized to process column by column.
    Returns list of dicts with match details ('keyword', 'column', 'count', 'example_context').
    """
    matches_list = []
    if not keywords or df.empty:
        return []

    # Compile pattern once
    try:
        pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
    except re.error as e:
        logger.error(f"Regex compilation error in find_keyword_matches: {e}")
        return []

    # Map lower-case keywords back to original case provided
    lower_to_original_kw = {k.lower(): k for k in keywords}

    # Convert DataFrame to string *once* (potential memory issue for huge DFs)
    try:
        df_str = df.astype(str)
    except Exception as e:
        logger.error(f"Failed to convert DataFrame to string for keyword search: {e}")
        return []

    # Iterate through columns
    for column in df_str.columns:
        # Skip columns unlikely to contain relevant text (e.g., purely numeric) - optional optimization
        # if pd.api.types.is_numeric_dtype(df[column].dtype): continue

        try:
            # Extract the *first* matching keyword per row in the column
            # expand=False returns a Series, flags=re.IGNORECASE makes it case-insensitive
            matches_series = df_str[column].str.extract(pattern, flags=re.IGNORECASE, expand=False)

            # Process only rows where a match was found
            all_matches = matches_series.dropna()
            if all_matches.empty:
                continue # No matches in this column

            # Group by the lower-case version of the matched keyword for counting
            grouped = all_matches.groupby(all_matches.str.lower())
            counts = grouped.size() # Get counts per unique keyword (case-insensitive)
            # Get the *original DataFrame index* of the first occurrence of each keyword
            first_indices = grouped.apply(lambda x: x.index[0])

            # Store results for this column
            for lower_keyword, count in counts.items():
                original_keyword = lower_to_original_kw.get(lower_keyword, lower_keyword) # Get original case
                first_idx = first_indices[lower_keyword] # Get index of first example
                # Get the original value from the *original* DataFrame for context
                example_context = df.loc[first_idx, column]

                match_details = {
                    'keyword': original_keyword,
                    'column': column,
                    'count': int(count), # Ensure native int type
                    'example_context': str(example_context) # Ensure string type
                }
                matches_list.append(match_details)

        # --- CORRECTED INDENTATION ---
        except Exception as e:
            # Log error for the specific column but continue with others
            logger.warning(f"Warning: Error searching column '{column}': {e}")
        # --- END CORRECTION ---

    return matches_list