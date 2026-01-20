# ==================================================================================== #
#                          downstream/steps/backfill.py
# ==================================================================================== #

import os
import pandas as pd
import logging
from workflow_16s.api.environmental_data.other.execute import EnvironmentalDataCollector
from workflow_16s.api.environmental_data.google.arkin_env_agents import main as arkin_env_agents
from workflow_16s.utils.dir_utils import Project

# Setup logger locally if needed
logger = logging.getLogger("workflow_16s")

def get_config_val(config, section, key, default):
    """Helper to safely get config values whether dict or object."""
    try:
        # Handle Pydantic object
        if hasattr(config, section):
            sect_obj = getattr(config, section)
            if isinstance(sect_obj, dict): return sect_obj.get(key, default)
            return getattr(sect_obj, key, default)
        
        # Handle Dictionary
        if isinstance(config, dict):
            return config.get(section, {}).get(key, default)
            
        return default
    except Exception:
        return default

def run_data_backfill(workflow):
    """
    Orchestrates the multi-API data backfill for missing metadata.
    Handles Arkin Env Agents, NFC facility matching, and Environmental data collection.
    """
    if workflow.adata is None: return
    workflow.logger.info("3. Modular Backfill: Running external API enrichment...")
    
    # 1. Arkin Agents Pathway
    # Fetches missing spatial and temporal data via LLM-driven agents
    if workflow.is_arkin_enabled:
        temp_meta = workflow.output_dir / "temp_meta_for_arkin.tsv"
        try:
            workflow.adata.obs.to_csv(temp_meta, sep='\t', index_label="#SampleID")
            arkin_df = arkin_env_agents(metadata_path=temp_meta, project_dir=Project(workflow.config))
            if arkin_df is not None and not arkin_df.empty:
                arkin_df['run_accession'] = arkin_df['associated_sample_ids'].str.split(', ')
                arkin_df = arkin_df.explode('run_accession').drop_duplicates(subset=['run_accession']).set_index('run_accession')
                workflow.adata.obs = workflow.adata.obs.merge(arkin_df, left_index=True, right_index=True, how='left', suffixes=('', '_arkin'))
        except Exception as e:
            workflow.logger.error(f"Arkin Agents backfill failed: {e}")
        finally: 
            if temp_meta.exists(): os.remove(temp_meta)

    # 2. NFC GIS Facility Matching
    # Maps sample coordinates to the nearest Nuclear Fuel Cycle facility
    if workflow.is_nfc_enabled and workflow.nfc_handler and not workflow.nfc_facilities_df.empty:
        
        # Check config to see if we should overwrite existing matches
        force_redo = get_config_val(workflow.config, 'nfc_facilities', 'match_existing_samples', False)
        
        if force_redo:
            workflow.logger.info("Force-matching ALL samples (match_existing_samples=True)...")
            # Select all samples that have coordinates
            mask = (workflow.adata.obs['lat'].notna()) & (workflow.adata.obs['lon'].notna())
            rows_to_match = workflow.adata.obs[mask].copy()
        else:
            # Standard mode: only fill missing
            if 'facility_match' not in workflow.adata.obs.columns:
                workflow.adata.obs['facility_match'] = None
            rows_to_match = workflow.adata.obs[workflow.adata.obs['facility_match'].isnull()].copy()

        if not rows_to_match.empty:
            workflow.logger.info(f"NFC matching for {len(rows_to_match)} locations...")
            # Ensure coordinates are float
            rows_to_match['lat'] = pd.to_numeric(rows_to_match['lat'], errors='coerce')
            rows_to_match['lon'] = pd.to_numeric(rows_to_match['lon'], errors='coerce')
            
            matched = workflow.nfc_handler._match_facilities_with_locations(workflow.nfc_facilities_df, rows_to_match)
            
            # Update the main dataframe
            if matched is not None and not matched.empty:
                workflow.adata.obs.update(matched)
                workflow.logger.info(f"Updated {len(matched)} samples with NFC facility data.")

    # 3. External Environmental Data Collector
    # Fetches soil, climate, and elevation data from public APIs
    if workflow.is_env_data_enabled:
        # Check for missing values in environmental columns
        all_env_cols = [c for c in workflow.adata.obs.columns if c.startswith(('SoilGrids_', 'Meteostat_'))]
        rows_to_fetch_mask = workflow.adata.obs[all_env_cols].isnull().all(axis=1) if all_env_cols else pd.Series(True, index=workflow.adata.obs.index)
        rows_to_fetch = workflow.adata.obs.loc[rows_to_fetch_mask].copy()
        
        if not rows_to_fetch.empty:
            data_collector = EnvironmentalDataCollector(data=rows_to_fetch, config=workflow.config, output_file=None)
            env_df = data_collector.run_apis()
            if env_df is not None: 
                workflow.adata.obs.update(env_df)