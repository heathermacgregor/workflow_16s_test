# ==================================================================================== #
#                           downstream/steps/preprocessing.py
# ==================================================================================== #

import math
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import cpu_count

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, issparse
import joblib
from joblib import Parallel, delayed

from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

logger = get_logger("workflow_16s")

# --- Constants ---
TAX_LEVELS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
EXPECTED_VAR_DTYPES = {
    'Taxon': 'string', 
    'Confidence': 'Float64', 
    'sequence': 'string',
    **{level: 'string' for level in TAX_LEVELS}
}
TARGET_GENE_NORMALIZATION = {
    '16S': ['16S', '16S rRNA', '16S rRNA gene', '16s', '16s rrna'],
    '18S': ['18S', '18S rRNA', '18s'],
    'ITS': ['ITS', 'ITS1', 'ITS2', 'its'],
}

# --- Helper Functions ---

def get_cfg_value(cfg_obj, key, default=None):
    """Helper to safely get config values from dict or object."""
    if isinstance(cfg_obj, dict): return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)

def normalize_target_gene(value: str) -> str:
    if pd.isna(value) or value in ['', 'nan', 'None', 'NA']: return pd.NA
    value_str = str(value).strip()
    for canonical, synonyms in TARGET_GENE_NORMALIZATION.items():
        if value_str in synonyms: return canonical
    return value_str

def standardize_dates(obs_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    date_patterns = ['date', 'time', 'temporal', 'year', 'month', 'day']
    date_columns = [col for col in obs_df.columns if any(pattern in col.lower() for pattern in date_patterns)]
    standardized_count = 0
    for col in date_columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(obs_df[col]):
                obs_df[col] = obs_df[col].dt.strftime('%Y-%m-%d')
                standardized_count += 1
                continue
            if pd.api.types.is_numeric_dtype(obs_df[col]): continue
            if obs_df[col].dtype == 'object' or pd.api.types.is_string_dtype(obs_df[col]):
                dt_series = pd.to_datetime(obs_df[col], errors='coerce', infer_datetime_format=True)
                original_valid = obs_df[col].notna().sum()
                parsed_valid = dt_series.notna().sum()
                if parsed_valid >= original_valid * 0.1 and parsed_valid > 0:
                    obs_df[col] = dt_series.dt.strftime('%Y-%m-%d').replace('NaT', pd.NA)
                    standardized_count += 1
        except Exception: continue
    return obs_df, standardized_count

def _validate_one_file(f: Path) -> Tuple[str, Union[Path, str]]:
    try:
        adata_individual = sc.read_h5ad(f, backed='r')
        if adata_individual.n_vars == 0: return (f.name, "Zero features.")
        return (f.stem, f)
    except Exception as e: 
        return (f.name, f"Failed read. Error: {e}")

# --- Core Processing Steps ---
def _clean_numeric_series(series: pd.Series, col_name: str) -> pd.Series:
    """
    Safely attempts to convert a column to numeric.
    PROTECTIONS:
    - Skips columns that look like IDs (e.g. contain 'accession', 'id', 'alias').
    - Skips date columns (handled separately).
    - Requires >80% valid conversion rate to accept changes.
    """
    col_lower = col_name.lower()
    
    # 1. SKIP IDENTIFIERS & DATES explicitly
    # These often contain numbers but should REMAIN strings/objects
    protected_terms = [
        'accession', 'alias', 'id', 'name', 'sra', 'project', 'study', 'experiment', 'run', 
        'sample', 'submission', 'ftp', 'url', 'link', 'md5', 'date', 'created', 'updated', 
        'time', 'tax_lineage', 'refs', 'publication', 'citation', 'description'
    ]
    if any(term in col_lower for term in protected_terms):
        return series

    # 2. Standardize Missing Values
    missing_indicators = ["nan", "NAN", "NaN", "Null", "null", "None", "none", "", " ", "Missing", "missing", "na", "NA", "unknown"]
    clean = series.copy().astype(str).str.strip()
    is_missing = clean.isin(missing_indicators) | clean.isna() | (clean.str.lower() == 'nan')
    
    # 3. Try Simple Coercion (e.g., "10.5", "-5")
    numeric_simple = pd.to_numeric(clean, errors='coerce')
    
    non_missing_count = (~is_missing).sum()
    if non_missing_count == 0:
        return series # Return original if empty

    valid_simple = (~numeric_simple.isna()).sum()
    ratio_simple = valid_simple / non_missing_count

    if ratio_simple > 0.90:
        return numeric_simple

    # 4. Aggressive Cleaning (Units)
    # Only try this if it's NOT a protected ID column
    # Regex: Extract first float/int (e.g., "10.5 cm" -> 10.5)
    numeric_extracted = clean.str.extract(r'^(-?\d+\.?\d*)')[0]
    numeric_aggressive = pd.to_numeric(numeric_extracted, errors='coerce')
    
    valid_aggressive = (~numeric_aggressive.isna()).sum()
    ratio_aggressive = valid_aggressive / non_missing_count

    # Higher threshold for aggressive cleaning to avoid accidents
    if ratio_aggressive > 0.85:
        # LOGGING: Only log if we actually changed non-numeric text to numbers
        salvaged_mask = numeric_simple.isna() & ~numeric_aggressive.isna() & ~is_missing
        if salvaged_mask.sum() > 0:
            examples = series[salvaged_mask].head(3).to_dict()
            logger.info(f"    🔧 Column '{col_name}': detected units/text mixed with numbers. Converting to numeric.")
            logger.info(f"       Salvaged {salvaged_mask.sum()} values. Examples: {examples} -> {[numeric_aggressive[i] for i in examples]}")
            
        return numeric_aggressive

    return series

def clean_metadata(adata, config=None):
    """
    Standardizes metadata: handles missing values, unifies date formats, 
    and enforces numeric types ONLY for measurement columns.
    """
    # 1. Standardize Missing Values (Global)
    missing_indicators = ["nan", "Null", "null", "None", "none", "", " ", "Unknown", "unknown", "Missing"]
    adata.obs = adata.obs.replace(missing_indicators, np.nan)
    
    # 2. Iterate Columns for Type Inference
    for col in adata.obs.columns:
        # Only process object/categorical columns that are NOT already numeric
        if not pd.api.types.is_numeric_dtype(adata.obs[col]):
            cleaned_series = _clean_numeric_series(adata.obs[col], col)
            
            # Update only if conversion happened
            if pd.api.types.is_numeric_dtype(cleaned_series):
                adata.obs[col] = cleaned_series

    # 3. Standardize Dates (Vectorized)
    # Look for 'date' or 'time' in name, but ignore numeric years (e.g. 2020) if possible
    date_cols = [c for c in adata.obs.columns if any(x in c.lower() for x in ['date', 'time', 'created', 'updated'])]
    
    for col in date_cols:
        # Skip if already numeric (like 'year' = 2020) unless it's a full timestamp
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
             continue
             
        try:
            # Force to datetime -> ISO format
            adata.obs[col] = pd.to_datetime(adata.obs[col], errors='coerce').dt.strftime('%Y-%m-%d')
        except Exception:
            continue

    return adata

def parse_taxonomy(adata):
    """
    Parses taxonomy strings into ranks (Kingdom..Species).
    Handles whitespace, prefixes, and 'unclassified' inheritance logic.
    """
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # Identify taxonomy column
    tax_col = next((c for c in adata.var.columns if c.lower() in ['taxon', 'taxonomy', 'lineage']), None)
    if not tax_col:
        for rank in ranks: adata.var[rank] = np.nan
        return adata

    try:
        # 1. Split Taxonomy String
        tax_df = adata.var[tax_col].astype(str).str.split(';', expand=True)
        
        if tax_df.shape[1] < len(ranks):
            for i in range(tax_df.shape[1], len(ranks)):
                tax_df[i] = np.nan
        
        tax_df = tax_df.iloc[:, :len(ranks)]
        tax_df.columns = ranks

        # 2. Vectorized Cleaning
        for rank in ranks:
            # Remove prefixes (d__, p__) and strip whitespace
            tax_df[rank] = tax_df[rank].str.replace(r'^[kpcofgsd]__', '', regex=True).str.strip()

        # 3. Handle 'Unclassified' / Missing Logic
        bad_values = ['unclassified', 'uncultured', 'ambiguous_taxa', '', 'nan', 'None']
        mask = tax_df.isin(bad_values) | tax_df.isna()
        clean_df = tax_df.where(~mask, np.nan)
        
        # Forward fill last valid rank
        filled_df = clean_df.ffill(axis=1)
        filled_df = filled_df.fillna("Unclassified")
        
        # Construct final "Unclassified Rank" strings
        final_df = clean_df.copy()
        for col in ranks:
            fallback = "Unclassified " + filled_df[col]
            final_df[col] = clean_df[col].combine_first(fallback)

        adata.var[ranks] = final_df[ranks]

    except Exception as e:
        logger.debug(f"Taxonomy parsing warning: {e}")
        for rank in ranks:
            if rank not in adata.var.columns: adata.var[rank] = np.nan

    return adata

def filter_samples_and_features(adata, config=None):
    """
    Removes Eukaryota, Mitochondria, Chloroplasts, and empty samples.
    """
    if adata.n_obs == 0: return adata
    
    to_drop = np.zeros(adata.n_vars, dtype=bool)

    # 1. Check for Contaminants
    if 'Kingdom' in adata.var.columns:
        is_euk = adata.var['Kingdom'].astype(str).str.contains('Eukaryota|Eukarya', case=False, na=False)
        to_drop = to_drop | is_euk

    if 'Family' in adata.var.columns:
        is_mito = adata.var['Family'].astype(str).str.contains('mitochondria', case=False, na=False)
        to_drop = to_drop | is_mito
        
    if 'Order' in adata.var.columns:
        is_chloro = adata.var['Order'].astype(str).str.contains('chloroplast', case=False, na=False)
        to_drop = to_drop | is_chloro

    if to_drop.sum() > 0:
        logger.debug(f"Dropping {to_drop.sum()} features (Eukaryota/Mito/Chloro).")
        adata = adata[:, ~to_drop].copy()

    # 2. Filter Empty Samples
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    n_pre_samples = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1)
    
    if n_pre_samples - adata.n_obs > 0:
        logger.debug(f"Dropped {n_pre_samples - adata.n_obs} empty samples.")

    return adata

def filter_low_depth_and_prevalence(adata: ad.AnnData, config: Union[AppConfig, dict]) -> Union[ad.AnnData, None]:
    """Filters by depth/prevalence."""
    if isinstance(config, dict):
        filter_config = config.get('preprocessing', {}).get('filter', {})
    else:
        filter_config = getattr(config.preprocessing, 'filter', None)

    if not get_cfg_value(filter_config, 'enabled', False): return adata
    
    min_depth = get_cfg_value(filter_config, 'min_sequencing_depth', 5000)
    min_prev = get_cfg_value(filter_config, 'min_sample_prevalence', 2)
    
    sc.settings.verbosity = 0
    sc.pp.filter_cells(adata, min_counts=min_depth)
    if adata.n_obs > 0:
        actual_min = min(min_prev, adata.n_obs)
        if actual_min > 1:
            if issparse(adata.X): adata.X = csc_matrix(adata.X)
            sc.pp.filter_genes(adata, min_cells=actual_min)
            if issparse(adata.X): adata.X = csr_matrix(adata.X)
    sc.pp.filter_cells(adata, min_counts=1)
    
    return adata

def qc_metrics(adata: ad.AnnData, output_dir: Union[str, Path]) -> None:
    """Calculates and plots basic QC metrics."""
    if adata is None or adata.n_obs == 0: return
    logger.info("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    plot_path = Path(output_dir) / "qc_metrics.png"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=adata.obs, x='total_counts', ax=axes[0], bins=30)
        sns.histplot(data=adata.obs, x='n_genes_by_counts', ax=axes[1], bins=30)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved QC plot: {plot_path}")
    except Exception: pass

def export_fasta(adata: ad.AnnData, config: Union[AppConfig, dict], output_dir: Union[str, Path]) -> None:
    """Exports sequences to FASTA."""
    if 'sequence' not in adata.var.columns: return
    fasta_path = Path(output_dir) / "all_features.fasta"
    try:
        with open(fasta_path, "w") as f:
            for feat_id, seq in adata.var['sequence'].dropna().items():
                f.write(f">{feat_id}\n{seq}\n")
        logger.info(f"FASTA exported to {fasta_path}")
    except Exception as e: logger.error(f"FASTA export failed: {e}")

# --- Utilities Class ---
class AnalysisUtils:
    TAX_LEVELS_ALL = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    @staticmethod
    def _aggregate_chunk(chunk_X, chunk_taxa, level):
        """Helper to aggregate a subset of the data."""
        # Create small DF for this chunk
        df = pd.DataFrame(chunk_X, columns=chunk_taxa)
        # Group by taxonomy level and sum
        return df.groupby(level, axis=1).sum()
    
    @staticmethod
    def get_analysis_adata(adata: ad.AnnData, level: str = 'Genus', n_jobs: int = -1):
        """
        Aggregates counts to a taxonomic level using Parallel Processing.
        """
        if level not in adata.var.columns:
            print(f"Taxonomy level {level} not found.")
            return None

        # 1. Get Taxonomy Mapping
        # Fill NaNs with "Unclassified" to prevent dropping data
        taxa_series = adata.var[level].fillna(f"Unclassified_{level}").astype(str)
        unique_taxa = taxa_series.unique()
        
        # 2. Parallel Aggregation strategy
        # Instead of converting the huge 32GB matrix to a DataFrame at once (SLOW),
        # we split the COLUMNS (Features) into chunks based on their taxonomy.
        
        # However, a faster way for sparse matrices is using a matrix multiplication approach.
        # Construct a transformation matrix (Features x Taxa)
        
        try:
            # OPTION A: Matrix Multiplication (Super Fast, Low Memory)
            # Create a dummy dataframe to get the grouping dummies
            dummies = pd.get_dummies(taxa_series)
            
            # Sparse dot product: (Samples x Features) @ (Features x Taxa) = (Samples x Taxa)
            # This sums the counts for features belonging to the same taxa
            if scipy.sparse.issparse(adata.X):
                X_agg = adata.X @ scipy.sparse.csr_matrix(dummies.values)
            else:
                X_agg = adata.X @ dummies.values
                
            new_obs = adata.obs.copy()
            new_var = pd.DataFrame(index=dummies.columns)
            
            return ad.AnnData(X=X_agg, obs=new_obs, var=new_var)

        except Exception as e:
            # Fallback to Parallel implementation if matrix algebra fails (e.g. memory)
            print(f"Matrix aggregation failed ({e}), switching to Parallel Pandas...")
            
            # Split samples into chunks to process in parallel
            n_samples = adata.n_obs
            chunk_size = int(np.ceil(n_samples / 20)) # 20 chunks
            chunks = [range(i, min(i + chunk_size, n_samples)) for i in range(0, n_samples, chunk_size)]
            
            # Use data.X directly
            if scipy.sparse.issparse(adata.X):
                X_dense = adata.X.todense()
            else:
                X_dense = adata.X

            results = Parallel(n_jobs=n_jobs)(
                delayed(AnalysisUtils._aggregate_chunk)(
                    X_dense[chunk_idx, :], 
                    taxa_series, 
                    level
                ) for chunk_idx in chunks
            )
            
            # Combine results
            final_df = pd.concat(results, axis=0)
            
            return ad.AnnData(X=final_df.values, obs=adata.obs.copy(), var=pd.DataFrame(index=final_df.columns))

    @staticmethod
    def _clr_transform(adata: ad.AnnData, pseudocount: float = 1) -> pd.DataFrame:
        """Performs Center Log Ratio (CLR) transformation."""
        try:
            if issparse(adata.X): X = adata.X.toarray()
            else: X = adata.X.copy()
            
            X = np.log(X + pseudocount)
            gm = X.mean(axis=1, keepdims=True)
            X_clr = X - gm
            return pd.DataFrame(X_clr, index=adata.obs_names, columns=adata.var_names)
        except Exception as e:
            logger.error(f"CLR transform failed: {e}")
            return pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    @staticmethod
    def apply_transform(adata: ad.AnnData, method: str) -> pd.DataFrame:
        """Applies requested transformation to adata.X and returns DataFrame."""
        
        # 1. Get Raw Data (Handle Sparse)
        if issparse(adata.X): X = adata.X.toarray()
        else: X = adata.X.copy()
        
        # 2. Apply Transformation
        if method == 'raw':
            X_trans = X
            
        elif method == 'binary':
            X_trans = (X > 0).astype(int)
            
        elif method == 'relative': # Total Sum Scaling
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1 # Avoid div by zero
            X_trans = X / row_sums
            
        elif method == 'log1p':
            X_trans = np.log1p(X)
            
        elif method == 'clr':
            # Use existing robust CLR
            return AnalysisUtils._clr_transform(adata, pseudocount=1)
            
        else:
            logger.warning(f"Unknown transformation '{method}', using raw.")
            X_trans = X

        return pd.DataFrame(X_trans, index=adata.obs_names, columns=adata.var_names)
    
    @staticmethod
    def find_plottable_metadata(adata: ad.AnnData, admin_noise_columns: Optional[List[str]] = None, fullness_threshold: float = 0.25, max_categories: int = 50) -> Dict[str, List[str]]:
        """Identifies metadata columns suitable for plotting."""
        if admin_noise_columns is None:
            admin_noise_columns = ['barcode', 'primer', 'linker', 'description', 'run_id']
            
        plottable = {'categorical': [], 'numeric': []}
        for col in adata.obs.columns:
            if any(noise in col.lower() for noise in admin_noise_columns): continue
            if adata.obs[col].notna().mean() < fullness_threshold: continue
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                if adata.obs[col].nunique() < 10: plottable['categorical'].append(col)
                else: plottable['numeric'].append(col)
            else:
                if adata.obs[col].nunique() <= max_categories and adata.obs[col].nunique() > 1:
                    plottable['categorical'].append(col)
        return plottable

# --- Main Entry Point ---

def run_preprocessing_pipeline(workflow):
    """Main entry point for the preprocessing pipeline."""
    logger.info("=== Starting Preprocessing Pipeline ===")
    
    if workflow.adata is None:
        logger.error("No AnnData object found in workflow. Skipping preprocessing.")
        return

    workflow.adata = clean_metadata(workflow.adata, workflow.config)
    workflow.adata = parse_taxonomy(workflow.adata)
    workflow.adata = filter_samples_and_features(workflow.adata, workflow.config)
    workflow.adata = filter_low_depth_and_prevalence(workflow.adata, workflow.config)
    
    if workflow.adata.n_obs == 0:
        logger.error("All samples removed during preprocessing.")
        workflow.adata = None
        return

    qc_metrics(workflow.adata, workflow.output_dir)
    export_fasta(workflow.adata, workflow.config, workflow.output_dir)
    
    # --- FIX: Ensure raw_counts layer exists for downstream steps ---
    if 'raw_counts' not in workflow.adata.layers:
        logger.info("Initializing 'raw_counts' layer from X (assuming raw inputs)...")
        workflow.adata.layers['raw_counts'] = workflow.adata.X.copy()
    
    logger.info("=== Preprocessing Pipeline Complete ===")