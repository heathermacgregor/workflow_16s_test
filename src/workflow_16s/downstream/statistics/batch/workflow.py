import logging
from pathlib import Path
from typing import Dict, Optional
import anndata as ad

from .diagnostics import detect_batch_effects
from .visualization import plot_batch_pca_interactive, plot_silhouette_analysis, plot_batch_heatmap
from .correction import apply_conqur_correction, apply_combat_correction

logger = logging.getLogger("workflow_16s")

def run_batch_workflow(
    adata: ad.AnnData,
    batch_col: str = 'batch',
    biology_col: Optional[str] = None,
    output_dir: Optional[Path] = None,
    correct_method: Optional[str] = None
) -> Dict:
    """
    Complete batch effect analysis workflow.
    
    Steps:
    1. Detect batch effects (Stats)
    2. Visualize batch effects (Plots)
    3. Correction (Optional)
    4. Re-assess (if corrected)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    results = {}
    
    # 1. Detection
    logger.info(">>> STEP 1: Detecting batch effects...")
    diag_res = detect_batch_effects(adata, batch_col, biology_col)
    results['before_correction'] = diag_res
    logger.info("\n" + diag_res['interpretation'])
    
    # 2. Visualization
    logger.info(">>> STEP 2: Creating visualizations...")
    if output_dir:
        plot_batch_pca(adata, batch_col, biology_col, output_path=output_dir/"batch_pca.html")
        plot_silhouette_analysis(adata, batch_col, output_path=output_dir/"batch_silhouette.png")
        plot_batch_heatmap(adata, batch_col, biology_col, output_path=output_dir/"batch_heatmap.png")
        
    # 3. Correction
    if correct_method:
        logger.info(f">>> STEP 3: Applying {correct_method.upper()} correction...")
        if correct_method.lower() == 'conqur':
            adata_corr = apply_conqur_correction(
                adata, batch_col, 
                covariate_cols=[biology_col] if biology_col else None,
                output_dir=output_dir
            )
        elif correct_method.lower() == 'combat':
            adata_corr = apply_combat_correction(
                adata, batch_col, 
                covariate_cols=[biology_col] if biology_col else None
            )
        else:
            logger.warning(f"Unknown method {correct_method}, skipping correction.")
            adata_corr = None
            
        if adata_corr is not None:
            results['corrected_data'] = adata_corr
            
            # 4. Re-assessment
            logger.info(">>> STEP 4: Re-assessing after correction...")
            post_res = detect_batch_effects(adata_corr, batch_col, biology_col)
            results['after_correction'] = post_res
            logger.info("\n" + post_res['interpretation'])
            
            if output_dir:
                plot_batch_pca_interactive(
                    adata_corr, batch_col, biology_col, 
                    output_path=output_dir/"batch_pca_corrected.html",
                    title="PCA After Batch Correction"
                )

    return results