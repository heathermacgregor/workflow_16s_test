"""
Proxy Columns Enrichment Step

Calculates derived environmental features from raw environmental data enrichment
columns for improved ML model performance.

Integration Point:
- Runs AFTER preprocessing (data is clean)
- Runs BEFORE analysis (features available for ML)
- Optional: can be disabled via config

Author: GitHub Copilot
Date: March 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from workflow_16s.utils.logger import get_logger
from workflow_16s.api.environmental_data.other.tools._proxy_columns_and_documentation import (
    ProxyColumnsCalculator,
    ColumnRegistry
)

logger = get_logger("workflow_16s")


def run_proxy_columns_enrichment(workflow):
    """
    Calculate proxy columns from raw environmental enrichment data.
    
    This step:
    1. Checks if environmental enrichment data exists in adata.obs
    2. Calculates 22 proxy columns across 7 categories
    3. Adds columns to adata.obs with proper metadata
    4. Logs statistics about coverage and data quality
    
    Args:
        workflow: DownstreamWorkflow object with adata attribute
    
    Returns:
        None (modifies workflow.adata in place)
    """
    
    if workflow.adata is None:
        logger.error("No AnnData object found. Skipping proxy columns enrichment.")
        return
    
    # Check if enrichment is enabled in config
    enrichment_cfg = getattr(workflow.config, 'downstream', {})
    if isinstance(enrichment_cfg, dict):
        proxy_enabled = enrichment_cfg.get('calculate_proxy_columns', True)
    else:
        proxy_enabled = getattr(enrichment_cfg, 'calculate_proxy_columns', True)
    
    if not proxy_enabled:
        logger.info("⊘ Proxy columns enrichment disabled in config (calculate_proxy_columns=False)")
        return
    
    logger.info("\n" + "="*80)
    logger.info("PHASE: ENVIRONMENTAL PROXY COLUMNS ENRICHMENT")
    logger.info("="*80)
    
    # Check for environmental data
    registry = ColumnRegistry()
    available_env_cols = [col for col in registry.get_all_columns().keys() 
                         if col in workflow.adata.obs.columns]
    
    if len(available_env_cols) == 0:
        logger.warning("⚠️  No environmental enrichment columns found in metadata.")
        logger.info("    Skipping proxy columns calculation (requires enrichment data first).")
        return
    
    logger.info(f"✓ Found {len(available_env_cols)} environmental variables")
    logger.info(f"  Ready to calculate proxy columns...")
    
    # Log coverage before calculation
    logger.debug(f"\nAvailable columns for proxy calculation:")
    for col in sorted(available_env_cols)[:10]:  # Show first 10
        non_null = workflow.adata.obs[col].notna().sum()
        pct = 100 * non_null / len(workflow.adata)
        logger.debug(f"  - {col:50s} {non_null:6d} samples ({pct:5.1f}%)")
    if len(available_env_cols) > 10:
        logger.debug(f"  ... and {len(available_env_cols) - 10} more")
    
    # Calculate proxy columns
    try:
        calculator = ProxyColumnsCalculator()
        n_cols_before = len(workflow.adata.obs.columns)
        
        logger.info("\n⏳ Calculating proxy columns... (this may take a moment)")
        workflow.adata.obs = calculator.calculate_all_proxies(workflow.adata.obs)
        
        n_cols_after = len(workflow.adata.obs.columns)
        n_proxy_cols = n_cols_after - n_cols_before
        
        logger.info(f"✅ Proxy columns calculated successfully!")
        logger.info(f"   Added {n_proxy_cols} new proxy columns")
        logger.info(f"   Total metadata columns: {n_cols_before} → {n_cols_after}")
        
        # Report statistics
        proxy_cols = [col for col in workflow.adata.obs.columns if col.startswith('proxy_')]
        logger.info(f"\n✓ Proxy columns by category:")
        
        categories = {
            'stress': [c for c in proxy_cols if 'stress' in c],
            'productivity': [c for c in proxy_cols if 'productivity' in c or 'growing_season' in c or 'variability' in c],
            'carbon': [c for c in proxy_cols if 'carbon' in c or 'decompos' in c],
            'biodiversity': [c for c in proxy_cols if 'biodiversity' in c or 'endemism' in c or 'fragmentation' in c],
            'human_pressure': [c for c in proxy_cols if 'human' in c or 'pollution' in c or 'agricultural' in c],
            'resilience': [c for c in proxy_cols if 'resilience' in c or 'stability' in c or 'recovery' in c],
            'soil_health': [c for c in proxy_cols if 'soil' in c],
        }
        
        for category, cols in categories.items():
            if cols:
                logger.info(f"  • {category.replace('_', ' ').title()}: {len(cols)} columns")
                for col in cols[:2]:  # Show first 2 per category
                    non_null = workflow.adata.obs[col].notna().sum()
                    pct = 100 * non_null / len(workflow.adata)
                    logger.debug(f"    - {col:50s} {pct:5.1f}%")
                if len(cols) > 2:
                    logger.debug(f"    ... and {len(cols)-2} more")
        
        # Overall coverage statistics
        logger.info(f"\n✓ Proxy column coverage:")
        proxy_coverage = {}
        for col in proxy_cols:
            non_null = workflow.adata.obs[col].notna().sum()
            pct = 100 * non_null / len(workflow.adata)
            proxy_coverage[col] = pct
        
        avg_coverage = np.mean(list(proxy_coverage.values()))
        min_coverage = np.min(list(proxy_coverage.values()))
        max_coverage = np.max(list(proxy_coverage.values()))
        
        logger.info(f"  Average coverage: {avg_coverage:6.1f}%")
        logger.info(f"  Range: {min_coverage:6.1f}% → {max_coverage:6.1f}%")
        
        # Mark enrichment status
        workflow.adata.uns['proxy_columns_enriched'] = True
        workflow.adata.uns['proxy_columns_count'] = n_proxy_cols
        workflow.adata.uns['proxy_columns_list'] = proxy_cols
        
        logger.info(f"\n✅ Proxy enrichment complete!")
        
    except Exception as e:
        logger.error(f"❌ Proxy columns calculation failed: {e}")
        logger.debug(f"Traceback: {type(e).__name__}: {str(e)}")
        logger.warning("Continuing workflow without proxy columns. Analysis may have reduced feature space.")
        return


def generate_proxy_columns_report(workflow, output_dir: Optional[Path] = None):
    """
    Generate documentation of proxy columns used in analysis.
    
    Creates a CSV file listing all proxy columns with metadata.
    
    Args:
        workflow: DownstreamWorkflow object
        output_dir: Output directory (uses workflow.output_dir if None)
    
    Returns:
        Path to report file
    """
    
    if output_dir is None:
        output_dir = workflow.output_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    proxy_cols = workflow.adata.uns.get('proxy_columns_list', [])
    
    if not proxy_cols:
        logger.warning("No proxy columns in adata.uns. Skipping report generation.")
        return None
    
    # Generate metadata for each proxy column
    registry = ColumnRegistry()
    all_columns = registry.get_all_columns()
    
    report_data = []
    for col in sorted(proxy_cols):
        # Parse column info from registry where possible
        if col in all_columns:
            col_info = all_columns[col]
            meta = {
                'column_name': col,
                'source': col_info.get('source'),
                'description': col_info.get('description'),
                'unit': col_info.get('unit'),
                'requires_date': '✅' if col_info.get('requires_date') else '❌',
                'quality_tier': col_info.get('quality_tier'),
                'ml_usefulness': col_info.get('ml_usefulness'),
            }
        else:
            # For calculated proxies, infer category from name
            if 'stress' in col:
                category = 'Stress Index'
            elif 'productivity' in col or 'growing_season' in col:
                category = 'Productivity'
            elif 'carbon' in col or 'decompos' in col:
                category = 'Carbon Cycling'
            elif 'biodiversity' in col:
                category = 'Biodiversity'
            elif 'human' in col or 'pollution' in col:
                category = 'Human Pressure'
            elif 'resilience' in col or 'stability' in col or 'recovery' in col:
                category = 'Resilience'
            elif 'soil' in col:
                category = 'Soil Health'
            else:
                category = 'Calculated'
            
            meta = {
                'column_name': col,
                'source': 'Calculated (Proxy)',
                'description': col.replace('proxy_', '').replace('_', ' ').title(),
                'unit': 'Index (0-1 or computed)',
                'requires_date': 'Conditional',
                'quality_tier': 'Derived',
                'ml_usefulness': 'High',
            }
        
        # Add coverage stats
        if col in workflow.adata.obs.columns:
            non_null = workflow.adata.obs[col].notna().sum()
            pct = 100 * non_null / len(workflow.adata)
            mean_val = workflow.adata.obs[col].mean() if pd.api.types.is_numeric_dtype(workflow.adata.obs[col]) else np.nan
            std_val = workflow.adata.obs[col].std() if pd.api.types.is_numeric_dtype(workflow.adata.obs[col]) else np.nan
            
            meta['coverage_count'] = non_null
            meta['coverage_percent'] = f"{pct:.1f}%"
            meta['mean_value'] = f"{mean_val:.3f}" if not np.isnan(mean_val) else "N/A"
            meta['std_value'] = f"{std_val:.3f}" if not np.isnan(std_val) else "N/A"
        
        report_data.append(meta)
    
    # Create DataFrame and save
    report_df = pd.DataFrame(report_data)
    report_file = output_dir / "proxy_columns_metadata.csv"
    report_df.to_csv(report_file, index=False)
    
    logger.info(f"✓ Proxy columns report saved: {report_file}")
    return report_file
