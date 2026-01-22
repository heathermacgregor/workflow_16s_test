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
from scipy.sparse import csc_matrix, csr_matrix, issparse
import joblib

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

def clean_metadata(adata: ad.AnnData, config: Union[AppConfig, dict]) -> Union[ad.AnnData, None]:
    """Clean metadata with robust dict/object support."""
    if isinstance(config, dict):
        metadata_config = config.get('preprocessing', {}).get('clean_metadata', None)
    else:
        metadata_config = getattr(config.preprocessing, 'clean_metadata', None) if hasattr(config, 'preprocessing') else None

    is_enabled = False
    if isinstance(metadata_config, dict): is_enabled = metadata_config.get('enabled', True)
    elif hasattr(metadata_config, 'enabled'): is_enabled = metadata_config.enabled
    
    if not is_enabled: return adata

    obs_df = adata.obs.copy()

    # Find text variations of "missing" and converts them to a single numpy NaN
    missing_variants = [
        'nan', 'NaN', 'NAN', 'null', 'Null', 'NULL',  # Text variations
        '', ' ', 'None'                               # Empty strings or python None
    ]
    
    # Mtch the exact string so we don't accidentally replace parts of real words (e.g., 'banana' -> 'ba')
    obs_df = obs_df.replace(to_replace=missing_variants, value=np.nan)

    obs_df, _ = standardize_dates(obs_df)
    adata.obs = obs_df
    return adata

def _parse_taxonomy_chunk(taxon_series_chunk: pd.Series) -> pd.DataFrame:
    """Helper to parse a chunk of taxonomy strings."""
    # Split taxonomy string by semicolon
    parsed = taxon_series_chunk.astype(str).str.split(';', expand=True)
    
    # Ensure we only take up to species (7 levels)
    num_levels = min(parsed.shape[1], len(TAX_LEVELS))
    parsed = parsed.iloc[:, :num_levels]
    parsed.columns = TAX_LEVELS[:num_levels]
    
    # Clean up prefixes (d__, p__, etc.) and handle empty values
    for col in parsed.columns:
        parsed[col] = parsed[col].str.replace(r'^[dpcofgs]__', '', regex=True).replace(['', 'Unassigned', 'nan'], np.nan).astype('string')
    
    return parsed

def parse_taxonomy(adata: ad.AnnData) -> Union[ad.AnnData, None]:
    """
    Parses 'Taxon' column.
    FORCED SEQUENTIAL EXECUTION to prevent thread bombs.
    """
    if 'Taxon' not in adata.var.columns: return adata
    
    taxon_series = adata.var['Taxon']
    
    # Execute sequentially
    parsed_taxonomy = _parse_taxonomy_chunk(taxon_series)
        
    cols_to_drop = [lvl for lvl in TAX_LEVELS if lvl in adata.var.columns]
    if cols_to_drop: adata.var.drop(columns=cols_to_drop, inplace=True)
    
    adata.var = pd.concat([adata.var, parsed_taxonomy], axis=1)
    
    return adata

def filter_samples_and_features(adata: ad.AnnData, config: Union[AppConfig, dict]) -> Union[ad.AnnData, None]:
    """Filters samples and features, robust to config type."""
    if isinstance(config, dict):
        filter_config = config.get('preprocessing', {}).get('filter', {})
    else:
        filter_config = getattr(config.preprocessing, 'filter', None)

    if not get_cfg_value(filter_config, 'enabled', False): return adata
    
    sc.settings.verbosity = 0 
    
    # 1. Target Gene
    target_gene_config = get_cfg_value(filter_config, 'target_gene')
    if get_cfg_value(target_gene_config, 'enabled', False):
        keep_genes = get_cfg_value(target_gene_config, 'keep_genes', ['16S'])
        meta_col = get_cfg_value(target_gene_config, 'metadata_column', 'target_gene')
        if meta_col in adata.obs.columns:
            adata.obs[meta_col] = adata.obs[meta_col].apply(normalize_target_gene)
            adata = adata[adata.obs[meta_col].astype(str).isin(keep_genes), :].copy()
    
    # 2. Empty Samples
    sc.pp.filter_cells(adata, min_counts=1)
    
    # 3. Contaminants
    contam_terms = get_cfg_value(filter_config, 'contaminant_terms', ['chloroplast', 'mitochondria'])
    present_cols = [c for c in TAX_LEVELS if c in adata.var.columns]
    
    features_to_remove = pd.Series(False, index=adata.var_names)
    if present_cols:
        all_taxa = pd.unique(adata.var[present_cols].values.ravel('K')).astype(str)
        contam_re = re.compile('|'.join(re.escape(t) for t in contam_terms), re.I)
        bad_taxa = set(t for t in all_taxa if contam_re.search(t))
        if bad_taxa:
            features_to_remove = adata.var[present_cols].isin(bad_taxa).any(axis=1)
            
    if features_to_remove.any():
        adata = adata[:, ~features_to_remove].copy()
        if issparse(adata.X): adata.X = csr_matrix(adata.X)

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
    def aggregate_adata_by_taxonomy(adata_in: ad.AnnData, tax_level: str = 'Genus') -> Optional[ad.AnnData]:
        if tax_level not in adata_in.var.columns: return None
        
        # --- FIX: THE NUCLEAR OPTION (Fixes "Cannot setitem") ---
        # 1. Force convert to Object (removes Categorical constraints)
        # 2. Fill NaNs
        # 3. Convert to String
        groups = adata_in.var[tax_level].astype(object).fillna('Unassigned').astype(str)
        
        # Create group mapping
        group_map = pd.get_dummies(groups)
        
        X = adata_in.X if not issparse(adata_in.X) else adata_in.X.toarray()
        agg_X = X @ group_map.values
        
        new_adata = ad.AnnData(agg_X, obs=adata_in.obs.copy())
        new_adata.var_names = group_map.columns
        new_adata.var.index.name = tax_level
        if issparse(adata_in.X): new_adata.X = csr_matrix(new_adata.X)
        
        # --- FIX: Ensure raw_counts layer is populated in aggregated object ---
        new_adata.layers['raw_counts'] = new_adata.X.copy()
        
        return new_adata

    @staticmethod
    def get_analysis_adata(adata_in: ad.AnnData, level: str) -> Optional[ad.AnnData]:
        if level == 'ASV': return adata_in.copy()
        if level in AnalysisUtils.TAX_LEVELS_ALL: 
            return AnalysisUtils.aggregate_adata_by_taxonomy(adata_in, tax_level=level)
        return None

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