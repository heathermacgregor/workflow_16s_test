# workflow_16s/upstream/metadata/directly_create_h5ad.py
import os
from pathlib import Path
from typing import Union
from workflow_16s.upstream.metadata.utils import (
    create_anndata_from_qiime_artifacts, 
    validate_anndata_file
)
from workflow_16s.downstream.utils.adata_utils import safe_write_h5ad
from workflow_16s.utils.logger import get_logger

def format_bytes(size_in_bytes):
    if size_in_bytes > 1_000_000:
        return f"{size_in_bytes / 1_000_000:.2f} MB"
    if size_in_bytes > 1_000:
        return f"{size_in_bytes / 1_000:.2f} KB"
    return f"{size_in_bytes} B"

def process_qiime_to_anndata(
    qiime_base_dir: Union[str, Path], 
    metadata_base_dir: Union[str, Path], 
    output_dir: Union[str, Path]
):
    """
    Finds complete sets of QIIME 2 artifacts and converts them into AnnData (.h5ad) files.
    
    It searches for:
    1. .../qiime_base_dir/<subset_path>/table/feature-table.biom
    2. .../qiime_base_dir/<subset_path>/*/taxonomy/taxonomy.tsv (flexible search)
    3. .../qiime_base_dir/<subset_path>/rep-seqs/dna-sequences.fasta
    4. .../metadata_base_dir/<subset_path>/sample-metadata.tsv
    
    And creates:
    - .../output_dir/<SUBSET.ID>.h5ad
    """
    logger = get_logger("workflow_16s")
    qiime_path = Path(qiime_base_dir)
    metadata_path = Path(metadata_base_dir)
    output_path = Path(output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f" 🔍 Starting search for QIIME artifacts in: {qiime_path}")
    
    # Use glob to find all feature tables, as this is a good anchor file
    for feature_table_biom_path in qiime_path.glob("**/table/feature-table.biom"):
        
        # Get the root directory for this specific analysis
        # (e.g., .../FWD_GTGCCAGCMGCCGCGGTAA_REV_GGACTACHVGGGTWTCTAAT)
        analysis_root = feature_table_biom_path.parent.parent
        
        try:
            # Get the relative path (e.g., PRJDB7915/illumina/paired/v4/FWD_...)
            relative_path = analysis_root.relative_to(qiime_path)
        except ValueError:
            logger.warning(f" ⚠️ Could not determine relative path for {analysis_root}. Skipping.")
            continue
            
        logger.info(f" 📥 Found potential analysis set: {relative_path}")

        # Find other required artifact paths 
        rep_seqs_fasta_path = analysis_root / "rep-seqs" / "dna-sequences.fasta"
        
        # Robustly find the taxonomy file. It could be in 'silva-...' or 'gg-...'
        taxonomy_paths = list(analysis_root.glob("*/taxonomy/taxonomy.tsv"))
        
        # Find metadata path 
        metadata_file_path = metadata_path / relative_path / "sample-metadata.tsv"

        # Validate all paths
        if not rep_seqs_fasta_path.exists():
            logger.warning(f" ⚠️ Missing rep-seqs: {rep_seqs_fasta_path}. Skipping.")
            continue
        
        if not taxonomy_paths:
            logger.warning(f" ⚠️ Missing taxonomy: No '*/taxonomy/taxonomy.tsv' found. Skipping.")
            continue
            
        if len(taxonomy_paths) > 1:
            logger.warning(f" ⚠️ Found multiple taxonomy files. Using first one: {taxonomy_paths[0]}")
        taxonomy_tsv_path = taxonomy_paths[0] # Use the first one found
            
        if not metadata_file_path.exists():
            logger.warning(f" ⚠️ Missing metadata: {metadata_file_path}. Skipping.")
            continue
            
        logger.info(f" ✅ Found all required files for: {relative_path}")

        # Construct output path and check if it exists 
        subset_id = str(relative_path).replace(os.sep, '.').upper()
        anndata_path = output_path / f"{subset_id}.h5ad"
        
        if anndata_path.exists():
            logger.info(f" ⏭️ Output file {anndata_path} already exists. Skipping.")
            continue
            
        # Create and write AnnData object
        try:
            logger.info(f" 💾 Creating final AnnData object for {subset_id}...")
            adata = create_anndata_from_qiime_artifacts(
                feature_table_biom_path,
                taxonomy_tsv_path,
                rep_seqs_fasta_path,
                rep_seqs_fasta_path.parent,  # rooted_tree_nwk_path (set to None as per "no tree file")
                metadata_file_path
            )

            logger.info(f" 💾 Writing AnnData object to {anndata_path}")
            safe_write_h5ad(adata, str(anndata_path))
            
            # Post-write validation 
            if anndata_path.exists():
                final_size = anndata_path.stat().st_size
                logger.debug(f"📦 Final AnnData file size for '{subset_id}': {format_bytes(final_size)}")
                validate_anndata_file(anndata_path, subset_id)
            else:
                raise FileNotFoundError(f" ❌ Failed to write AnnData file to {anndata_path}")
                
        except Exception as e:
            logger.error(f"Failed to create AnnData for {subset_id}: {e}")
            # Clean up partially written file if it exists
            if anndata_path.exists():
                anndata_path.unlink()
                logger.debug(f" ✨ Removed partial file: {anndata_path}")

    logger.info("🏁 AnnData processing complete.")


if __name__ == "__main__":
    # Base directory where QIIME 2 outputs are (e.g., .../per_dataset/qiime)
    QIIME_DIR = "/usr2/people/macgregor/amplicon/test/data/per_dataset/qiime"
    # Base directory where corresponding metadata is (e.g., .../per_dataset/metadata)
    METADATA_DIR = "/usr2/people/macgregor/amplicon/test/data/per_dataset/metadata"
    # Directory to save the final .h5ad files
    OUTPUT_DIR = "/usr2/people/macgregor/amplicon/project_01/03_processed_data"
    
    # Run the main processing function
    process_qiime_to_anndata(
        qiime_base_dir=QIIME_DIR,
        metadata_base_dir=METADATA_DIR,
        output_dir=OUTPUT_DIR
    )