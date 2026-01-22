import logging
from pathlib import Path
from typing import Dict, Optional, Any
import anndata as ad

from .temporal import (
    check_temporal_structure, 
    calculate_temporal_stability, 
    trajectory_clustering
)
from .analysis import run_zibr, run_maaslin2_longitudinal

logger = logging.getLogger('workflow_16s')

def longitudinal_analysis_workflow(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    group_col: Optional[str] = None,
    method: str = 'zibr',
    output_dir: Optional[Path] = None,
    **kwargs: Any
) -> Dict:
    """
    Complete longitudinal analysis workflow.
    """
    logger.info("="*60)
    logger.info("LONGITUDINAL ANALYSIS WORKFLOW")
    logger.info("="*60)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Check Structure
    info = check_temporal_structure(adata, time_col, subject_col)
    if not info['is_longitudinal']:
        raise ValueError("Data lacks repeated measurements.")
        
    # 2. Run Statistics
    if method == 'zibr':
        results = run_zibr(adata, time_col, subject_col, group_col, **kwargs)
    elif method == 'maaslin2':
        if not output_dir: raise ValueError("MaAsLin2 requires output_dir")
        results = run_maaslin2_longitudinal(
            adata, time_col, subject_col,
            fixed_effects=kwargs.get('fixed_effects', [time_col]),
            random_effects=kwargs.get('random_effects', []),
            output_dir=output_dir,
            **{k:v for k,v in kwargs.items() if k not in ['fixed_effects', 'random_effects']}
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # 3. Exploratory Analysis
    stability = calculate_temporal_stability(adata, time_col, subject_col)
    clusters = trajectory_clustering(adata, time_col, subject_col)
    
    # 4. Save
    if output_dir:
        if isinstance(results, pd.DataFrame):
            results.to_csv(output_dir / 'longitudinal_stats.csv')
        stability.to_csv(output_dir / 'temporal_stability.csv')
        if 'cluster_assignments' in clusters:
            clusters['cluster_assignments'].to_csv(output_dir / 'trajectory_clusters.csv')
            
    return {
        'info': info,
        'results': results,
        'stability': stability,
        'clusters': clusters
    }