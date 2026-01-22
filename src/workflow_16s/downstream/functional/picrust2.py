"""
Functional Prediction Module.

Tools for inferring metabolic functions from taxonomy (FAPROTAX) 
or gene content (PICRUSt2).
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import anndata as ad
import pandas as pd

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

def run_picrust2_pipeline(
    picrust2_output_dir: Path, 
    fasta_path: Path, 
    abund_table_path: Path, 
    n_cpus: int = 4, 
    conda_env: Optional[str] = None
) -> bool:
    """Runs the full PICRUSt2 pipeline using a conda environment."""
    logger.info("--- Running PICRUSt2 Pipeline ---")
    if picrust2_output_dir.exists():
        logger.warning(f"Removing existing PICRUSt2 directory: {picrust2_output_dir}")
        shutil.rmtree(picrust2_output_dir)
        
    cmd_base = [
        'picrust2_pipeline.py', '-s', str(fasta_path.resolve()), 
        '-i', str(abund_table_path.resolve()), 
        '-o', str(picrust2_output_dir.resolve()), '-p', str(n_cpus)
    ]
    
    if conda_env:
        cmd_full = ['conda', 'run', '-n', conda_env] + cmd_base
        logger.info(f"Using conda env: {conda_env}")
    else:
        cmd_full = cmd_base
        
    logger.info(f"Command: {' '.join(cmd_full)}")
    
    try:
        subprocess.run(cmd_full, check=True, capture_output=True, text=True)
        logger.info("✅ `picrust2_pipeline.py` completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"PICRUSt2 failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("PICRUSt2 command not found.")
        return False

def load_picrust2_results(adata: ad.AnnData, picrust2_output_dir: Path) -> ad.AnnData:
    """Loads PICRUSt2 pathway and EC predictions into adata.obsm."""
    logger.info("--- Loading PICRUSt2 Results ---")
    
    pathway_file = picrust2_output_dir / "pathways_out" / "path_abun_unstrat.tsv.gz"
    ec_file = picrust2_output_dir / "EC_metagenome_out" / "pred_metagenome_unstrat.tsv.gz"
    
    if not hasattr(adata, 'obsm'): adata.obsm = {}
    
    # Load Pathways
    if pathway_file.exists():
        try:
            path_df = pd.read_csv(pathway_file, sep='\t', index_col=0).T
            path_df = path_df.reindex(adata.obs_names).fillna(0)
            adata.obsm['picrust2_pathways'] = path_df
            
            # Load descriptions
            desc_file = picrust2_output_dir / "pathways_out" / "path_abun_unstrat_descrip.tsv.gz"
            if desc_file.exists():
                names_df = pd.read_csv(desc_file, sep='\t', index_col=0)
                adata.uns['picrust2_pathway_names'] = names_df['description'].to_dict()
                
            logger.info(f"Loaded {path_df.shape[1]} pathways.")
        except Exception as e:
            logger.error(f"Failed loading pathways: {e}")
            
    # Load ECs
    if ec_file.exists():
        try:
            ec_df = pd.read_csv(ec_file, sep='\t', index_col=0).T
            ec_df = ec_df.reindex(adata.obs_names).fillna(0)
            adata.obsm['picrust2_ec'] = ec_df
            logger.info(f"Loaded {ec_df.shape[1]} EC numbers.")
        except Exception as e:
            logger.error(f"Failed loading ECs: {e}")
            
    return adata