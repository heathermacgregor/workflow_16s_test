# downstream/ecological_insights_integration.py

"""
Ecological Insights Integration Module

Orchestrates the three new ecological analysis modules:
1. Gradient analysis (bimodal ecotypes)
2. Microgeography (spatial clustering)
3. Temporal dynamics (stability & succession)

Provides unified configuration, caching, and result aggregation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging
import json
from datetime import datetime

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream import gradient_analysis
from workflow_16s.downstream import microgeography
from workflow_16s.downstream import temporal_dynamics

logger = get_logger("workflow_16s")


def run_ecological_insights_pipeline(
    adata,
    beta_diversity_df: Optional[pd.DataFrame] = None,
    output_base_dir: Path = None,
    config: Dict = None,
    analyses_to_run: List[str] = None,
    force_recompute: bool = False
) -> Dict:
    """
    Main orchestrator for ecological insights pipeline.
    
    Args:
        adata: AnnData object with expression and metadata
        beta_diversity_df: Precomputed beta diversity matrix (optional, auto-computed if None)
        output_base_dir: Base output directory (creates subdirectories for each analysis)
        config: Configuration dict from workflow
        analyses_to_run: List of analyses to run (default: all)
            Options: ["gradient", "microgeography", "temporal"]
        force_recompute: Skip caching and recompute all results
    
    Returns:
        Aggregated results dict with all three analyses
    """
    
    if output_base_dir is None:
        output_base_dir = Path("./ecological_insights")
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    if analyses_to_run is None:
        analyses_to_run = ["gradient", "microgeography", "temporal"]
    
    logger.info("\n" + "="*80)
    logger.info("ECOLOGICAL INSIGHTS PIPELINE")
    logger.info("="*80)
    logger.info(f"Running analyses: {', '.join(analyses_to_run)}")
    logger.info(f"Output directory: {output_base_dir}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(adata),
        "n_features": len(adata.var),
        "analyses": {}
    }
    
    # Compute beta diversity if needed (for microgeography)
    if "microgeography" in analyses_to_run and beta_diversity_df is None:
        logger.info("\n  Computing beta diversity (Bray-Curtis) for microgeography...")
        from scipy.spatial.distance import pdist, squareform
        from skbio.diversity.beta import braycurtis
        
        try:
            adata_dense = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            beta_div_list = []
            
            for i in range(len(adata_dense)):
                for j in range(i + 1, len(adata_dense)):
                    bc_dist = braycurtis(adata_dense[i], adata_dense[j])
                    beta_div_list.append(bc_dist)
            
            beta_diversity_df = squareform(beta_div_list)
        except Exception as e:
            logger.warning(f"  ⚠️ Could not compute beta diversity: {e}")
            beta_diversity_df = None
    
    # 1. GRADIENT ANALYSIS
    if "gradient" in analyses_to_run:
        logger.info("\n" + "-"*80)
        logger.info("1. GRADIENT ANALYSIS (Ecotype Detection)")
        logger.info("-"*80)
        
        gradient_dir = output_base_dir / "01_gradient_analysis"
        
        try:
            # Prepare OTU table
            otu_table = pd.DataFrame(
                adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                index=adata.obs_names,
                columns=adata.var_names
            )
            
            gradient_result = gradient_analysis.run_gradient_analysis(
                otu_table=otu_table,
                metadata_df=adata.obs.copy(),
                output_dir=gradient_dir,
                config=config or {},
                auto_enrich_metadata=True,
                bimodality_threshold=0.7,
                min_samples_per_otu=10
            )
            
            results["analyses"]["gradient"] = {
                "status": "completed",
                "n_bimodal_otus": gradient_result.get("n_bimodal"),
                "output_dir": str(gradient_dir),
                "key_findings": f"{gradient_result.get('n_bimodal', 0)} OTUs show bimodal distributions along environmental gradients"
            }
        except Exception as e:
            logger.error(f"  ❌ Gradient analysis failed: {e}")
            results["analyses"]["gradient"] = {"status": "failed", "error": str(e)}
    
    # 2. MICROGEOGRAPHY ANALYSIS
    if "microgeography" in analyses_to_run:
        logger.info("\n" + "-"*80)
        logger.info("2. MICROGEOGRAPHY ANALYSIS (Spatial Clustering)")
        logger.info("-"*80)
        
        microgeography_dir = output_base_dir / "02_microgeography"
        
        try:
            if beta_diversity_df is None:
                logger.warning("  ⚠️ Beta diversity not available, skipping microgeography")
                results["analyses"]["microgeography"] = {
                    "status": "skipped",
                    "reason": "Beta diversity not provided"
                }
            else:
                microgeography_result = microgeography.run_microgeography_analysis(
                    adata=adata,
                    beta_diversity_df=beta_diversity_df,
                    output_dir=microgeography_dir,
                    config=config or {},
                    lat_col="latitude",
                    lon_col="longitude",
                    spatial_threshold_km=1.0
                )
                
                mantel_r = microgeography_result.get("mantel", {}).get("r_observed", np.nan)
                mantel_p = microgeography_result.get("mantel", {}).get("p_value", np.nan)
                
                results["analyses"]["microgeography"] = {
                    "status": "completed",
                    "mantel_r": float(mantel_r),
                    "mantel_p_value": float(mantel_p),
                    "output_dir": str(microgeography_dir),
                    "key_findings": f"Mantel correlation r={mantel_r:.3f} (p={mantel_p:.4f})" if not np.isnan(mantel_r) else "No spatial structure detected"
                }
        except Exception as e:
            logger.error(f"  ❌ Microgeography analysis failed: {e}")
            results["analyses"]["microgeography"] = {"status": "failed", "error": str(e)}
    
    # 3. TEMPORAL DYNAMICS ANALYSIS
    if "temporal" in analyses_to_run:
        logger.info("\n" + "-"*80)
        logger.info("3. TEMPORAL DYNAMICS ANALYSIS (Succession & Stability)")
        logger.info("-"*80)
        
        temporal_dir = output_base_dir / "03_temporal_dynamics"
        
        try:
            # Check for time column
            if "collection_date" in adata.obs.columns or "date_collected" in adata.obs.columns:
                time_col = "collection_date" if "collection_date" in adata.obs.columns else "date_collected"
                
                temporal_result = temporal_dynamics.run_temporal_analysis(
                    adata=adata,
                    time_col=time_col,
                    output_dir=temporal_dir,
                    config=config,
                    group_col=None
                )
                
                mean_turnover = temporal_result.get("global_turnover", {}).get("mean_turnover", np.nan)
                
                results["analyses"]["temporal"] = {
                    "status": "completed",
                    "mean_otu_turnover": float(mean_turnover),
                    "output_dir": str(temporal_dir),
                    "key_findings": f"Mean OTU turnover={mean_turnover:.3f}" if not np.isnan(mean_turnover) else "Time series analysis complete"
                }
            else:
                logger.warning("  ⚠️ No temporal metadata found, skipping temporal dynamics")
                results["analyses"]["temporal"] = {
                    "status": "skipped",
                    "reason": "No temporal metadata column found"
                }
        except Exception as e:
            logger.error(f"  ❌ Temporal dynamics analysis failed: {e}")
            results["analyses"]["temporal"] = {"status": "failed", "error": str(e)}
    
    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("ECOLOGICAL INSIGHTS SUMMARY")
    logger.info("="*80)
    
    for name, status_dict in results["analyses"].items():
        status = status_dict.get("status", "unknown")
        if status == "completed":
            key_findings = status_dict.get("key_findings", "")
            logger.info(f"✓ {name.upper()}: {key_findings}")
        elif status == "skipped":
            logger.info(f"⊘ {name.upper()}: {status_dict.get('reason', 'Skipped')}")
        else:
            logger.info(f"❌ {name.upper()}: {status_dict.get('error', 'Unknown error')}")
    
    # Save results JSON
    results_json_path = output_base_dir / "ecological_insights_summary.json"
    with open(results_json_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✓ All results saved to {output_base_dir}/")
    logger.info(f"✓ Summary saved to {results_json_path}")
    
    return results


def generate_ecological_report(
    results_dict: Dict,
    output_path: Path
) -> None:
    """
    Generate a human-readable HTML report from ecological insights.
    
    Args:
        results_dict: Results from run_ecological_insights_pipeline()
        output_path: Path to save HTML report
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ecological Insights Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #34495e; padding-bottom: 10px; }}
            .section {{ margin-bottom: 40px; }}
            .status-completed {{ color: green; }}
            .status-skipped {{ color: orange; }}
            .status-failed {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
            th {{ background-color: #ecf0f1; }}
        </style>
    </head>
    <body>
        <h1>Ecological Insights Report</h1>
        <p><strong>Generated:</strong> {results_dict.get('timestamp', 'Unknown')}</p>
        <p><strong>Samples:</strong> {results_dict.get('n_samples', 'Unknown')}</p>
        <p><strong>Features:</strong> {results_dict.get('n_features', 'Unknown')}</p>
        
        <div class="section">
            <h2>Analysis Results</h2>
            <table>
                <tr>
                    <th>Analysis</th>
                    <th>Status</th>
                    <th>Key Findings</th>
                </tr>
    """
    
    for name, status_dict in results_dict.get("analyses", {}).items():
        status = status_dict.get("status", "unknown")
        status_class = f"status-{status}"
        key_findings = status_dict.get("key_findings", status_dict.get("reason", "N/A"))
        
        html_content += f"""
                <tr>
                    <td>{name.upper()}</td>
                    <td class="{status_class}">{status.upper()}</td>
                    <td>{key_findings}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    Path(output_path).write_text(html_content)
    logger.info(f"✓ HTML report saved to {output_path}")
