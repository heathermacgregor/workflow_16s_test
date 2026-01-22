import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any
import anndata as ad
import plotly.graph_objects as go

from .inference import run_spiec_easi, run_sparcc, run_proportionality
from .visualization import plot_network

logger = logging.getLogger('workflow_16s')

def network_analysis_workflow(
    adata: ad.AnnData,
    method: str = 'spiec-easi',
    output_dir: Optional[Path] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Complete network analysis workflow: Inference -> Visualization -> Saving.
    
    Args:
        adata: AnnData object
        method: 'spiec-easi', 'sparcc', or 'proportionality'
        output_dir: Directory to save results (optional)
        **kwargs: Arguments passed to the specific inference method
    """
    logger.info("="*60)
    logger.info(f"NETWORK ANALYSIS WORKFLOW: {method.upper()}")
    logger.info("="*60)
    
    # 1. Run Inference
    if method == 'spiec-easi':
        results = run_spiec_easi(adata, **kwargs)
    elif method == 'sparcc':
        results = run_sparcc(adata, **kwargs)
    elif method.startswith('proportionality'):
        # Handle 'proportionality' or 'proportionality-rho' etc.
        metric = 'rho'
        if '-' in method: metric = method.split('-')[1]
        results = run_proportionality(adata, method=metric, **kwargs)
    else:
        raise ValueError(f"Unknown network method: {method}")
    
    # 2. Create Visualization
    logger.info("Creating network visualization...")
    fig = plot_network(results)
    
    # 3. Save Outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Edge List (CSV)
        if 'edge_list' in results:
            results['edge_list'].to_csv(output_dir / 'network_edges.csv', index=False)
            
        # Save Statistics (JSON)
        if 'stats' in results:
            with open(output_dir / 'network_stats.json', 'w') as f:
                json.dump(results['stats'], f, indent=4)
        
        # Save Plot (HTML)
        fig.write_html(str(output_dir / 'network_plot.html'))
        logger.info(f"Results saved to {output_dir}")

    return {
        'results': results,
        'figure': fig
    }