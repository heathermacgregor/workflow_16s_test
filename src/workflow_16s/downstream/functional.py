# ==================================================================================== #

# Standard Library Imports
import numpy as np
import pandas as pd
import anndata as ad
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List

# Third Party Imports
from sklearn.preprocessing import MultiLabelBinarizer 
from scipy.sparse import issparse

# Local Imports
from workflow_16s.utils.faprotax import FaprotaxDB
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger("workflow_16s")

# ==================================================================================== #

def predict_functions_faprotax(adata: ad.AnnData, faprotax_db: FaprotaxDB) -> ad.AnnData:
    """
    Predicts functional groups for taxa in an AnnData object using FAPROTAX.
    This version is optimized to use the parallel `predict_functions_batch` method.
    """
    logger.info("--- Predicting functions using FAPROTAX ---")
    if 'Taxon' not in adata.var.columns:
        logger.error("FAPROTAX requires 'Taxon' column in adata.var. Run parse_taxonomy first.")
        return adata

    # 1. Get the FULL list of all taxonomy strings (not just unique ones)
    full_taxa_list = adata.var['Taxon'].astype(str).tolist()
    logger.info(f"Predicting functions for {len(full_taxa_list)} total features (in parallel)...")

    # 2. Call the batch function with the FULL list.
    try:
        all_function_lists = faprotax_db.predict_functions_batch(full_taxa_list)
    except Exception as e:
        logger.error(f"FAPROTAX batch prediction failed: {e}")
        return adata

    logger.info("Batch prediction complete. Binarizing results...")

    # 3. Binarize the results (this is fast)
    mlb = MultiLabelBinarizer()
    function_matrix = mlb.fit_transform(all_function_lists)
    function_names = mlb.classes_
    n_found = len(function_names)

    if n_found == 0:
        logger.warning("FAPROTAX prediction returned no functional groups for this dataset.")
        return adata

    logger.info(f"Found {n_found} unique FAPROTAX functions. Adding to adata.var...")

    # 4. Create the final boolean DataFrame
    func_df = pd.DataFrame(data=function_matrix, index=adata.var_names, columns=[f"faprotax:{name}" for name in function_names]).astype(bool) # type: ignore

    # 5. Remove old columns and concatenate the new ones
    old_cols = [c for c in adata.var.columns if c.startswith("faprotax:")]
    if old_cols:
        logger.debug(f"Removing {len(old_cols)} old FAPROTAX columns.")
        adata.var.drop(columns=old_cols, inplace=True)

    adata.var = pd.concat([adata.var, func_df], axis=1)
    logger.info(f"Added {n_found} FAPROTAX function columns to adata.var.")

    return adata


def run_picrust2_pipeline(picrust2_output_dir: Path, fasta_path: Path, abund_table_path: Path, n_cpus: int = 4, conda_env: Optional[str] = None) -> bool:
    """Runs the full PICRUSt2 pipeline using a conda environment."""
    logger.info("--- Running PICRUSt2 Pipeline ---")
    if picrust2_output_dir.exists(): logger.warning(f"Removing existing PICRUSt2 directory: {picrust2_output_dir}"); shutil.rmtree(picrust2_output_dir)
    # Define the core command
    cmd_base = ['picrust2_pipeline.py', '-s', str(fasta_path.resolve()), '-i', str(abund_table_path.resolve()), '-o', str(picrust2_output_dir.resolve()), '-p', str(n_cpus)]
    # Prepend 'conda run' if an environment is specified
    if conda_env: cmd_full = ['conda', 'run', '-n', conda_env] + cmd_base; logger.info(f"Using conda env: {conda_env}")
    else: cmd_full = cmd_base; logger.warning("No conda_env specified for PICRUSt2. Assuming 'picrust2_pipeline.py' is in the current PATH.")
    logger.info(f"Command: {' '.join(cmd_full)}")
    try:
        result = subprocess.run(cmd_full, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info("✅ `picrust2_pipeline.py` completed successfully.")
        if result.stdout: logger.debug(f"PICRUSt2 stdout:\n{result.stdout}")
        if result.stderr: logger.debug(f"PICRUSt2 stderr:\n{result.stderr}")
    except FileNotFoundError:
        if conda_env: logger.error(f"`conda` command not found. Is it installed and in your system's PATH?")
        else: logger.error(f"`picrust2_pipeline.py` command not found. Is it installed and in your PATH?")
        return False
    except subprocess.CalledProcessError as e: logger.error(f"PICRUSt2 pipeline failed with error:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"); return False
    pathway_dir = picrust2_output_dir / "pathways_out"; ec_metagenome_dir = picrust2_output_dir / "EC_metagenome_out"; ko_metagenome_dir = picrust2_output_dir / "KO_metagenome_out"
    final_pathway_file = pathway_dir / "path_abun_unstrat.tsv.gz"; final_ec_file = ec_metagenome_dir / "pred_metagenome_unstrat.tsv.gz"
    if not final_pathway_file.exists(): logger.warning(f"Expected PICRUSt2 pathway output file not found: {final_pathway_file}")
    if not final_ec_file.exists(): logger.warning(f"Expected PICRUSt2 EC output file not found: {final_ec_file}")
    if not final_pathway_file.exists() and not final_ec_file.exists(): logger.error("No primary PICRUSt2 outputs were generated."); return False
    return True 


def load_picrust2_results(adata: ad.AnnData, picrust2_output_dir: Path) -> ad.AnnData:
    """
    Loads PICRUSt2 pathway and EC number predictions into the AnnData object.
    This creates new AnnData objects for pathways and ECs and stores them in adata.obsm as DataFrames, which is a common pattern for storing aggregated data.
    """
    logger.info("--- Loading PICRUSt2 Results ---")
    pathway_file = picrust2_output_dir / "pathways_out" / "path_abun_unstrat.tsv.gz"; ec_file = picrust2_output_dir / "EC_metagenome_out" / "pred_metagenome_unstrat.tsv.gz"
    if not hasattr(adata, 'obsm'): adata.obsm = {}
    # Load Pathway Abundances
    if pathway_file.exists():
        try:
            logger.info(f"Loading PICRUSt2 pathways from: {pathway_file}")
            path_df = pd.read_csv(pathway_file, sep='\t', index_col=0).T
            # Align with adata.obs index
            path_df = path_df.reindex(adata.obs_names).fillna(0)
            # Store as a DataFrame in obsm
            adata.obsm['picrust2_pathways'] = path_df; logger.info(f"Successfully loaded {path_df.shape[1]} pathways into adata.obsm['picrust2_pathways']")
            # Create a simple .uns lookup for pathway names
            pathway_names_file = picrust2_output_dir / "pathways_out" / "path_abun_unstrat_descrip.tsv.gz"
            if pathway_names_file.exists(): names_df = pd.read_csv(pathway_names_file, sep='\t', index_col=0); adata.uns['picrust2_pathway_names'] = names_df['description'].to_dict()
        except Exception as e: logger.error(f"Failed to load PICRUSt2 pathway results: {e}")
    else: logger.warning(f"PICRUSt2 pathway file not found: {pathway_file}")
    # Load EC Number Abundances
    if ec_file.exists():
        try:
            logger.info(f"Loading PICRUSt2 EC numbers from: {ec_file}"); ec_df = pd.read_csv(ec_file, sep='\t', index_col=0).T
            # Align with adata.obs index
            ec_df = ec_df.reindex(adata.obs_names).fillna(0)
            # Store as a DataFrame in obsm
            adata.obsm['picrust2_ec'] = ec_df; logger.info(f"Successfully loaded {ec_df.shape[1]} EC numbers into adata.obsm['picrust2_ec']")
            # Create a simple .uns lookup for EC names
            ec_names_file = picrust2_output_dir / "EC_metagenome_out" / "pred_metagenome_unstrat_descrip.tsv.gz"
            if ec_names_file.exists(): names_df = pd.read_csv(ec_names_file, sep='\t', index_col=0); adata.uns['picrust2_ec_names'] = names_df['description'].to_dict()
        except Exception as e: logger.error(f"Failed to load PICRUSt2 EC number results: {e}")
    else: logger.warning(f"PICRUSt2 EC file not found: {ec_file}")
    return adata


def run_conqur_correction(adata: ad.AnnData, batch_col: str, key_vars: List[str], other_covariates: Optional[List[str]], output_dir: Path, r_script_path: Path) -> Optional[pd.DataFrame]:
    """
    Runs the ConQuR R script via subprocess to perform batch correction.

    Args:
        adata: The AnnData object with 'raw_counts' layer.
        batch_col: The column name in adata.obs identifying the batch.
        key_vars: List of biological variables to preserve (from adata.obs).
        other_covariates: List of other covariates to adjust for (from adata.obs).
        output_dir: The main output directory.
        r_script_path: Path to the 'run_conqur.R' script.

    Returns:
        A pandas DataFrame of batch-corrected counts, or None if failed.
    """
    logger.info("--- Starting ConQuR Batch Correction ---")
    # 1. Check if Rscript is available
    if not shutil.which("Rscript"): logger.error("`Rscript` command not found. Please install R and ensure it's in your PATH."); return None
    if not r_script_path.exists(): logger.error(f"ConQuR R script not found at: {r_script_path}"); return None
    # 2. Define temp file paths
    conqur_temp_dir = output_dir / "conqur_temp"; conqur_temp_dir.mkdir(exist_ok=True, parents=True)
    counts_file = conqur_temp_dir / "conqur_input_counts.tsv"; meta_file = conqur_temp_dir / "conqur_input_metadata.tsv"; output_file = conqur_temp_dir / "conqur_corrected_counts.tsv"
    # 3. Export data from AnnData
    try:
        # Export raw counts (samples x features)
        # Ensure it's a dense numpy array before converting to DataFrame
        raw_counts = adata.layers['raw_counts']
        if issparse(raw_counts): counts_data = raw_counts.toarray() # type: ignore
        else: counts_data = np.asarray(raw_counts)
        counts_df = pd.DataFrame(counts_data, index=adata.obs_names, columns=adata.var_names); counts_df.to_csv(counts_file, sep="\t")
        # Export metadata
        meta_cols = [batch_col] + key_vars
        if other_covariates: meta_cols.extend(other_covariates)
        # Get unique column names, preserving order
        unique_meta_cols = list(dict.fromkeys(meta_cols))
        meta_df = adata.obs[unique_meta_cols]; meta_df.to_csv(meta_file, sep="\t")
    except Exception as e: logger.error(f"Failed to export data for ConQuR: {e}"); return None
    # 4. Build and Run Subprocess Command
    cmd = ["Rscript", str(r_script_path.resolve()), "--input_counts", str(counts_file.resolve()), "--input_metadata", str(meta_file.resolve()), "--output_counts", str(output_file.resolve()), "--batch_col", batch_col, "--key_vars", ",".join(key_vars)]
    if other_covariates: cmd.extend(["--covariates", ",".join(other_covariates)])
    logger.info("Running ConQuR R script... (This can take a long time)")
    logger.debug(f"Command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        stdout_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line: logger.info(f"[Rscript]: {line}"); stdout_lines.append(line)
        process.wait() # Wait for the process to complete
        if process.returncode != 0: logger.error(f"ConQuR R script failed.\nReturn Code: {process.returncode}"); return None
        logger.info("ConQuR R script completed successfully.")
    except FileNotFoundError: logger.error("`Rscript` command not found. Please install R and ensure it's in your PATH."); return None
    except Exception as e: logger.error(f"An unexpected error occurred while running ConQuR: {e}"); return None
    # 5. Read corrected data back into Python
    if not output_file.exists(): logger.error("ConQuR ran but did not produce the output file."); return None  
    logger.info("Loading corrected count table...")
    corrected_counts_df = pd.read_csv(output_file, sep="\t", index_col=0)
    # Align DataFrame back to adata (in case ConQuR changed order)
    corrected_counts_df = corrected_counts_df.reindex(index=adata.obs_names, columns=adata.var_names).fillna(0)
    # ConQuR can produce negative numbers in rare cases, ensure counts are >= 0
    corrected_counts_df[corrected_counts_df < 0] = 0
    # Round to nearest integer as we started with counts
    corrected_counts_df = pd.DataFrame(np.round(corrected_counts_df).astype(np.float32), index=corrected_counts_df.index, columns=corrected_counts_df.columns)
    logger.info("✅ ConQuR Batch Correction successful.")
    return corrected_counts_df