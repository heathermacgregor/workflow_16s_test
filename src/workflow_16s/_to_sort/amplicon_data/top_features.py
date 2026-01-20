# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.figures.merged import create_feature_abundance_map, violin_feature
from workflow_16s.stats.tests import (
    fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest
)
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df, 
    update_table_and_meta, to_biom
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== FUNCTIONS ===================================== #

def top_features_plots(
    output_dir,
    config,
    top_features,
    tables,
    meta,
    nfc_facilities,
    verbose
):
    output_dir = output_dir / 'top_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    top_features_c = copy.deepcopy(top_features)
    n = config.get('violin_plots', {}).get('n', 30)
    
    for col, vals in top_features_c.items():
        for val, features in vals.items():
            group_key = f"{col}={val}"
            #logger.info(f"Processing top features for group: {group_key}")
            
            with get_progress_bar() as progress:
                groupval_desc = f"Processing '{col}'={val} features"
                groupval_task = progress.add_task(
                    _format_task_desc(groupval_desc), 
                    total=min(n, len(features))
                )
                
                for i, feature in enumerate(features[:n]):
                    if not isinstance(feature, dict):
                        logger.warning(f"Skipping non-dict feature entry: {feature}")
                        progress.update(groupval_task, advance=1)
                        continue
                    
                    table_type = feature.get('table_type')
                    level = feature.get('level')
                    feature_name = feature.get('feature')
                    
                    if not all([table_type, level, feature_name]):
                        logger.warning(f"Missing required keys in feature: {feature}")
                        progress.update(groupval_task, advance=1)
                        continue
                    
                    try:
                        # Get the table and convert to DataFrame
                        biom_table = tables.get(table_type, {}).get(level)
                        if not biom_table:
                            logger.warning(f"Table not found: {table_type}/{level}")
                            continue
                            
                        table_df = table_to_df(biom_table)
                        
                        # Verify feature exists
                        if feature_name not in table_df.columns:
                            logger.warning(f"Feature '{feature_name}' not found in table")
                            continue
                            
                        # Create a DataFrame with just this feature
                        feature_df = table_df[[feature_name]].copy()
                        
                        # Normalize IDs for matching
                        feature_df.index = feature_df.index.astype(str).str.strip().str.lower()
                        meta_ids = meta['#sampleid'].astype(str).str.strip().str.lower()
                        
                        # Align metadata with feature table
                        common_ids = feature_df.index.intersection(meta_ids)
                        if len(common_ids) == 0:
                            logger.warning("No matching samples between feature table and metadata")
                            continue
                            
                        # Add group column using aligned IDs
                        group_map = meta.set_index(meta_ids)[col]
                        feature_df[col] = feature_df.index.map(group_map)
                        
                        # Remove samples without group assignment
                        feature_df = feature_df.dropna(subset=[col])
                        
                        # Create output directory
                        feature_output_dir = output_dir / col / str(val) / table_type / level
                        feature_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Initialize figures storage
                        if 'figures' not in feature:
                            feature['figures'] = {}
                        
                        # Generate violin plot
                        if config.get('violin_plots', {}).get('enabled', False):
                            try:
                                fig = violin_feature(
                                    df=feature_df,
                                    feature=feature_name,
                                    output_dir=feature_output_dir,
                                    status_col=col
                                )
                                # Store figure reference in results
                                feature['figures']['violin'] = fig
                                # Also store at top level for easy access
                                feature['violin_figure'] = fig  
                            except Exception as e:
                                logger.error(f"Violin plot failed: {str(e)}")
                        
                        # Generate feature map
                        if config.get('feature_maps', {}).get('enabled', False):
                            try:
                                fig_map = create_feature_abundance_map(
                                    metadata=meta,
                                    feature_abundance=feature_df[[feature_name]],
                                    feature_name=feature_name,
                                    nfc_facilities_data=nfc_facilities,
                                    output_dir=feature_output_dir,
                                    show=False,
                                    verbose=verbose
                                )
                                feature['figures']['abundance_map'] = fig_map
                            except Exception as e:
                                logger.error(f"Feature map failed: {str(e)}")
                                
                    except Exception as e:
                        logger.error(f"Error processing feature {feature_name}: {str(e)}")
                    finally:
                        progress.update(groupval_task, advance=1)
    
    return top_features_c
