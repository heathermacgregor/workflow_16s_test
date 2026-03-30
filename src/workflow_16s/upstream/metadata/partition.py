import asyncio
import gzip
import json
import os
import re
import traceback
import subprocess
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from Bio import Entrez

from workflow_16s.api.ena import SQLiteCacheManager as EnaCacheManager, SequenceFetcher
from workflow_16s.api.environmental_data import NFCFacilitiesHandler
from workflow_16s.api.publication.fetcher import PublicationFetcher
from workflow_16s.config import AppConfig
from workflow_16s.upstream.sequences.analysis import PrimerFinder, run_comprehensive_analysis
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import get_logger

from .cache import PartitionCacheManager
from .constants import exclusion_keywords
from .processor import PartitionProcessor

logger = get_logger('workflow_16s')

def is_empty(var) -> bool:
    if var is None: return True
    if isinstance(var, list): return len(var) == 0
    if isinstance(var, (pd.DataFrame, pd.Series)): return var.empty
    return False

class DatasetPartition:
    FWD_PRIMER_COL = 'pcr_primer_fwd_seq'
    REV_PRIMER_COL = 'pcr_primer_rev_seq'
    MIN_RUNS_THRESHOLD = 1

    def __init__(self, config, ena_client, publication_fetcher, env_collector, arkin_agents, region_to_pairs_map, nfc_handler=None, nfc_facilities_df=None, progress_obj=None):
        self.config, self.ena_client = config, ena_client
        self.logger = get_logger('workflow_16s')
        self.project_dir = Project(config)
        self.mode = self.config.sequences.pcr_primers.mode.lower()
        self.progress_obj = progress_obj
        self.cache = PartitionCacheManager(self.project_dir.cache / 'partition_cache.db')
        self.publication_fetcher = publication_fetcher
        self.region_to_pairs_map = region_to_pairs_map
        self.nfc_handler = nfc_handler
        self.nfc_facilities_df = nfc_facilities_df if nfc_facilities_df is not None else pd.DataFrame()
        self.processed_h5ad_paths, self.failed = [], []
        Entrez.email = config.credentials.ena_email

    async def run(self, datasets, ena_cache_manager, ui_stats_ref=None, metadata_task_id=None, qiime_task_id=None, ui_refresher=None):
        heavy_task_semaphore = asyncio.Semaphore(1)
        network_semaphore = asyncio.Semaphore(10)
        async def wrapped_process(ds_id, info, prog):
            try:
                success = await self._managed_process(ds_id, info, ena_cache_manager, heavy_task_semaphore, network_semaphore, prog, metadata_task_id, qiime_task_id)
                if ui_stats_ref is not None:
                    key = 'success_h5ad' if success else 'failed_h5ad'
                    ui_stats_ref[key] = ui_stats_ref.get(key, 0) + 1
                if not success:
                    already_recorded = any(f.get('dataset') == ds_id for f in self.failed)
                    if not already_recorded:
                        self.failed.append({'dataset': ds_id, 'error': 'No output partitions were produced.'})
            except Exception as e:
                self.logger.error(f'Unhandled failure: {e}', exc_info=True)
                if ui_stats_ref is not None:
                    ui_stats_ref['failed_h5ad'] = ui_stats_ref.get('failed_h5ad', 0) + 1
                self.failed.append({'dataset': ds_id, 'error': str(e)})
            finally:
                if ui_refresher: ui_refresher.refresh()
        tasks = [asyncio.create_task(wrapped_process(ds_id, info, f'{i}/{len(datasets)}')) for i, (ds_id, info) in enumerate(datasets.items(), 1)]
        await asyncio.gather(*tasks)
        return self.processed_h5ad_paths, self.failed

    async def _managed_process(self, dataset_id, info, ena_cache_manager, heavy_semaphore, network_semaphore, progress_str, metadata_task_id, qiime_task_id):
        raw_seq_dir = Path(self.project_dir.raw_data) / dataset_id / 'seqs'
        try:
            async with network_semaphore:
                self.logger.info(f'Starting Step 1: {dataset_id}')
                metadata_df = await self._fetch_metadata_logic(dataset_id, ena_cache_manager)
                if not is_empty(metadata_df):
                    processor = PartitionProcessor(self.config, self.project_dir, self.nfc_handler, self.nfc_facilities_df, self.progress_obj)
                    await processor.pre_flight_checks(dataset_id, metadata_df)
                if self.progress_obj and metadata_task_id: self.progress_obj.update(metadata_task_id, advance=1)
            async with heavy_semaphore:
                return await self.process(dataset_id, info, metadata_df, progress_str, qiime_task_id)
        except Exception as e:
            self.logger.error(f'Failure for {dataset_id}: {e}', exc_info=True)
            self.failed.append({'dataset': dataset_id, 'error': str(e)})
            return False
        finally:
            if raw_seq_dir.exists(): shutil.rmtree(raw_seq_dir, ignore_errors=True)

    async def process(self, dataset, info, ena_metadata, progress_str, qiime_task_id) -> bool:
        if is_empty(ena_metadata): return False
        if isinstance(ena_metadata, list): ena_metadata = pd.DataFrame(ena_metadata)
        ena_runs = ena_metadata.set_index('run_accession', drop=False)
        results = await self.auto(dataset, info, ena_metadata, ena_runs, qiime_task_id)
        if not results:
            self.logger.warning(f"⚠️ [DEBUG] Process for {dataset} produced ZERO results. This is why the dashboard isn't updating.")
            return False
        # If we have results and no critical errors, consider it a step forward
        success = all(isinstance(r, Path) and r.exists() for r in results) if results else False
        self.logger.info(f"📊 [DEBUG] Process for {dataset} returning success={success}")
        return success

    async def auto(self, dataset, info, metadata, ena_runs, qiime_task_id):
        self.logger.info(f"🔍 [DEBUG] Entering auto for {dataset}. Total runs in metadata: {len(metadata)}")
        consensus = info.get('target_subfragment') or 'NA'
        est_dir = self.project_dir.raw_data / dataset
        all_local_paths = {}
        if consensus in ['NA', 'Undetermined']:
            fetcher = SequenceFetcher(fastq_dir=str(est_dir / 'seqs' / 'raw'), progress_obj=self.progress_obj)
            subsampled = list(metadata['run_accession'])[:5]
            downloaded = await fetcher.download_run_fastq_async(ena_runs[ena_runs.index.isin(subsampled)])
            all_local_paths = {k: [Path(f) for f in v] for k, v in downloaded.items() if v}
            reports = await run_comprehensive_analysis(all_local_paths, est_dir, self.config.paths.vsearch_db, self.config.paths.primer_db, self.region_to_pairs_map, 8, 4, self.progress_obj)
            success_regs = [r.get('prediction') for r in reports.values() if r.get('prediction') not in [None, 'Undetermined']]
            if success_regs: consensus = Counter(success_regs).most_common(1)[0][0]
            else: return []
        processor = PartitionProcessor(self.config, self.project_dir, self.nfc_handler, self.nfc_facilities_df, self.progress_obj)
        tasks, final_metadata = [], metadata.copy()
        final_metadata['predicted_region'] = consensus
        for keys, group_df in final_metadata.groupby(['library_layout', 'instrument_platform']):
            subset_id = f'{dataset}.{keys[0]}.{keys[1]}.{consensus}'.upper()
            anndata_path = self.project_dir.processed_data / f'{subset_id}.h5ad'
            if anndata_path.exists():
                self.processed_h5ad_paths.append(anndata_path); continue
            p_paths = {r: all_local_paths[r] for r in group_df['run_accession'] if r in all_local_paths}
            self.logger.info(f'🚀 Launching QIIME for {subset_id}')
            tasks.append(processor.process_partition(group_df, dataset, {'target_subfragment': consensus}, p_paths, ena_runs, subset_id, anndata_path))
        if not tasks:
            self.logger.error(f'❌ CRITICAL: Zero tasks created for {dataset}. Possible column mismatch.')
            self.logger.info(f'Available columns: {list(final_metadata.columns)}')
        results = await asyncio.gather(*tasks, return_exceptions=False) if tasks else []
        self.processed_h5ad_paths.extend([r for r in results if isinstance(r, Path)])
        return results

    async def _fetch_metadata_logic(self, dataset_id, ena_cache_manager):
        key = f'metadata_{dataset_id}'
        df = await ena_cache_manager.get(key)
        if df is None:
            from workflow_16s.api.ena import ENAClient
            client = ENAClient(self.config)
            df = await client.get_project_metadata(dataset_id)
            if df is not None: await ena_cache_manager.set(key, df)
        return pd.DataFrame(df) if isinstance(df, list) else df
