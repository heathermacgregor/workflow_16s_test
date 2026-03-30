# downstream/functional_annotation.py

"""
Functional Annotation Module: Map OTUs to gene functions via RAST or KEGG.

Answers: Which OTUs have which functions? Are functions phylogenetically conserved?
Works with sparse taxonomy + optional RAST annotation files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.sparse import csr_matrix
import logging

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


# --- Functional Gene Categories (from RAST/KEGG) ---
FUNCTIONAL_GENES = {
    "uranium_reduction": {
        "kegg_orthologs": ["K00531", "K00532"],  # c-type cytochrome oxidoreductase
        "rast_roles": ["Uranium reduction"],
        "genera": ["Geobacter", "Anaeromyxobacter", "Desulfosporosinus"],
    },
    "arsenic_metabolism": {
        "kegg_orthologs": ["K03893", "K03894"],  # Arsenic resistance proteins
        "rast_roles": ["Arsenic resistance"],
        "genera": ["Sulfurospirillum", "Thermodesulfobacterium", "Bacillus"],
    },
    "heavy_metal_efflux": {
        "kegg_orthologs": ["K01507", "K01520"],  # Metal-transporting ATPases
        "rast_roles": ["Heavy metal resistance", "Metal efflux"],
        "genera": ["Cupriavidus", "Ralstonia", "Pseudomonas"],
    },
    "nitrate_reduction": {
        "kegg_orthologs": ["K00370", "K00371", "K00372"],  # Nitrate reductase
        "rast_roles": ["Nitrate reduction", "Denitrification"],
        "genera": ["Bacillus", "Paracoccus", "Pseudomonas"],
    },
    "sulfur_metabolism": {
        "kegg_orthologs": ["K00394", "K00395"],  # Sulfite oxidase
        "rast_roles": ["Sulfur metabolism"],
        "genera": ["Thiobacillus", "Allochromatium", "Desulfovibrio"],
    },
    "biofilm_formation": {
        "kegg_orthologs": ["K03707", "K03708"],  # Biofilm formation proteins
        "rast_roles": ["Biofilm formation"],
        "genera": ["Pseudomonas", "Bacillus", "Vibrio"],
    },
}


def parse_taxonomy_to_genus(taxonomy_str: str) -> str:
    """
    Extract genus from QIIME2 taxonomy string.
    
    Example: 'k__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus;s__'
    Returns: 'Bacillus'
    """
    if pd.isna(taxonomy_str):
        return "Unknown"
    
    parts = str(taxonomy_str).split(";")
    for part in reversed(parts):
        part_clean = part.strip()
        if part_clean.startswith("g__"):
            genus = part_clean.replace("g__", "").strip()
            if genus and genus.lower() not in ["unclassified", "uncultured"]:
                return genus
    
    # Fallback: return last classify level
    for part in reversed(parts):
        part_clean = part.replace("__", "").replace("_", " ").strip()
        if part_clean and part_clean.lower() not in ["unclassified", "uncultured"]:
            return part_clean
    
    return "Unknown"


def assign_functions_from_taxonomy(
    taxonomy_df: pd.DataFrame,
    genes_of_interest: List[str],
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Assign functions based on taxonomic genus.
    
    Args:
        taxonomy_df: DataFrame with index=OTU_ID, columns include 'taxonomy'
        genes_of_interest: List of functional categories to assign
        confidence_threshold: Confidence score (0-1) for genus match
    
    Returns:
        DataFrame: OTU × Function matrix (containing 0/1 or confidence scores)
    """
    
    logger.info(f"🧬 Assigning functions to {len(taxonomy_df)} OTUs based on taxonomy...")
    
    # Extract genera
    taxonomy_df["genus"] = taxonomy_df.get("taxonomy", "Unknown").apply(parse_taxonomy_to_genus)
    
    # Initialize function matrix
    function_matrix = pd.DataFrame(
        0.0,
        index=taxonomy_df.index,
        columns=genes_of_interest
    )
    
    # Assign functions based on genus membership
    for func_name, func_info in FUNCTIONAL_GENES.items():
        if func_name not in genes_of_interest:
            continue
        
        genera_list = func_info.get("genera", [])
        
        # Find OTUs with matching genera
        mask = taxonomy_df["genus"].isin(genera_list)
        function_matrix.loc[mask, func_name] = confidence_threshold
        
        n_assigned = mask.sum()
        logger.debug(f"  ✓ {func_name}: {n_assigned} OTUs (genera: {', '.join(genera_list)})")
    
    logger.info(f"✓ Function assignment complete: {function_matrix.shape[0]} OTUs × {function_matrix.shape[1]} functions")
    
    return function_matrix


def load_rast_annotations(
    rast_file: Path,
    taxonomy_df: pd.DataFrame,
    genes_of_interest: List[str]
) -> pd.DataFrame:
    """
    Load RAST annotations from otus.97.allinfo or similar file.
    
    Args:
        rast_file: Path to RAST annotation file (tab-separated)
        taxonomy_df: DataFrame with OTU metadata
        genes_of_interest: Functional categories to extract
    
    Returns:
        OTU × Function matrix from RAST annotations
    """
    
    if not rast_file.exists():
        logger.warning(f"⚠️ RAST file not found: {rast_file}. Falling back to taxonomy-based assignment.")
        return assign_functions_from_taxonomy(taxonomy_df, genes_of_interest)
    
    logger.info(f"📖 Loading RAST annotations from {rast_file}...")
    
    try:
        rast_df = pd.read_csv(rast_file, sep="\t", header=0)
    except Exception as e:
        logger.error(f"Failed to parse RAST file: {e}")
        return assign_functions_from_taxonomy(taxonomy_df, genes_of_interest)
    
    # Initialize function matrix
    function_matrix = pd.DataFrame(
        0.0,
        index=taxonomy_df.index,
        columns=genes_of_interest
    )
    
    # Parse RAST annotation for each OTU
    for otu_id in taxonomy_df.index:
        otu_rows = rast_df[rast_df.get("OTU_ID", rast_df.columns[0]) == otu_id]
        
        if otu_rows.empty:
            continue
        
        # Extract roles from RAST (usually in 'Role' or 'Function' column)
        roles = otu_rows.get("Role", [])
        if isinstance(roles, pd.Series):
            roles = roles.tolist()
        else:
            roles = [roles]
        
        # Match roles to functional genes
        roles_str = " ".join(str(r) for r in roles).lower()
        
        for func_name, func_info in FUNCTIONAL_GENES.items():
            if func_name not in genes_of_interest:
                continue
            
            rast_roles = [r.lower() for r in func_info.get("rast_roles", [])]
            if any(role in roles_str for role in rast_roles):
                function_matrix.loc[otu_id, func_name] = 1.0  # Assign 100% confidence from RAST
    
    logger.info(f"✓ RAST annotations loaded: {(function_matrix > 0).sum().sum()} function assignments")
    
    return function_matrix


def run_functional_annotation(
    otu_table: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    rast_annotation_path: Optional[Path] = None,
    genes_of_interest: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main entry point for functional annotation.
    
    Args:
        otu_table: OTU abundance table (samples × OTUs)
        taxonomy_df: Taxonomy metadata (OTU_ID × taxonomy)
        output_dir: Output directory for results
        config: Configuration dict
        rast_annotation_path: Optional path to RAST annotation file
        genes_of_interest: Genes to annotate
    
    Returns:
        Tuple of:
        - function_matrix: OTU × Function matrix
        - results_summary: Dict with annotation statistics
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genes_of_interest = genes_of_interest or list(FUNCTIONAL_GENES.keys())
    
    logger.info("\n" + "="*80)
    logger.info("FUNCTIONAL ANNOTATION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Input: {len(otu_table.columns)} OTUs across {len(otu_table)} samples")
    logger.info(f"Genes to annotate: {', '.join(genes_of_interest)}")
    
    # Load RAST if available, otherwise use taxonomy-based assignment
    if rast_annotation_path and Path(rast_annotation_path).exists():
        function_matrix = load_rast_annotations(
            Path(rast_annotation_path),
            taxonomy_df,
            genes_of_interest
        )
    else:
        function_matrix = assign_functions_from_taxonomy(
            taxonomy_df,
            genes_of_interest
        )
    
    # Create abundance matrix for functions
    # (sum abundance of OTUs with each function per sample)
    function_abundance = pd.DataFrame(
        0.0,
        index=otu_table.index,
        columns=genes_of_interest
    )
    
    for func in genes_of_interest:
        func_otus = function_matrix[function_matrix[func] > 0].index
        if len(func_otus) > 0:
            function_abundance[func] = otu_table[func_otus].sum(axis=1)
    
    # Calculate statistics
    results_summary = {
        "n_otus": len(function_matrix),
        "n_samples": len(otu_table),
        "n_functions": len(genes_of_interest),
        "function_coverage": {
            func: (function_matrix[func] > 0).sum()
            for func in genes_of_interest
        },
        "samples_with_each_function": {
            func: (function_abundance[func] > 0).sum()
            for func in genes_of_interest
        }
    }
    
    # Save results
    function_matrix.to_csv(output_dir / "otu_function_matrix.csv")
    function_abundance.to_csv(output_dir / "sample_function_abundance.csv")
    
    logger.info(f"\n✓ Results saved to {output_dir}/")
    for func, n_otus in results_summary["function_coverage"].items():
        n_samples = results_summary["samples_with_each_function"][func]
        logger.info(f"  {func}: {n_otus} OTUs, present in {n_samples} samples")
    
    return function_matrix, results_summary
