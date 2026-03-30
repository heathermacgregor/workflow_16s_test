# workflow_16s/upstream/metadata/processor.py

import ast
import asyncio
import datetime
import gzip
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
from pandas.api.types import is_datetime64_any_dtype

from workflow_16s.api.ena import SequenceFetcher
from workflow_16s.api.environmental_data import (
    EnvironmentalDataCollector, run_arkin_enrichment
)
from workflow_16s.api.environmental_data.nuclear_fuel_cycle.main import NFCFacilitiesHandler
from workflow_16s.api.qiime import execute
from workflow_16s.config import AppConfig
from workflow_16s.metadata.manager import MetadataManager, process_metadata
from workflow_16s.utils.dir_utils import Project, QIIME, RawData, SubSet
from workflow_16s.utils.io.placeholder import write_qiime_manifest
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.io.anndata import (
    create_anndata_from_qiime_artifacts, 
    embed_provenance, format_bytes, safe_write_h5ad,
    validate_anndata_file
)

logger = get_logger("workflow_16s")

def verify_md5(file_path: Path, expected_md5: str) -> bool:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5

def run_quality_guard(project_raw_dir: Path, min_reads: int = 1000) -> bool:
    """Checks if a project has enough data to survive DADA2."""
    total_reads = 0
    fastq_files = list(project_raw_dir.glob("seqs/raw/*.fastq.gz"))
    
    if not fastq_files:
        return False
        
    for f in fastq_files[:3]:
        try:
            with gzip.open(f, "rb") as gz:
                count = sum(1 for line in gz)
                total_reads += (count // 4)
        except Exception:
            continue
            
    return total_reads >= min_reads

class AdapterScanner:
    def __init__(self):
        self.adapter_signatures = {
            'Nextera': 'CTGTCTCTTATA', 
            'TruSeq': 'AGATCGGAAGAGC'
        }

    def scan_fastq(self, filepath: str, sample_size: int = 10000, threshold: float = 0.05) -> List[str]:
        hits = {k: 0 for k in self.adapter_signatures}
        reads_processed = 0
        try:
            with gzip.open(filepath, 'rt') as f:
                for i, line in enumerate(f):
                    if i % 4 == 1:
                        seq = line.strip()
                        reads_processed += 1
                        for name, signature in self.adapter_signatures.items():
                            if signature in seq:
                                hits[name] += 1
                    if reads_processed >= sample_size:
                        break
        except Exception as e:
            get_logger("workflow_16s").warning(f"Adapter scan failed on {filepath}: {e}")
            return []

        detected = [name for name, count in hits.items() if reads_processed > 0 and (count / reads_processed) >= threshold]
        if detected:
            get_logger("workflow_16s").warning(f"🚨 Adapters detected in raw reads: {detected}")
        return detected

class PartitionProcessor:
    def __init__(self, config: AppConfig, project_dir: Project, nfc_handler: Optional[NFCFacilitiesHandler] = None, nfc_facilities_df: Optional[pd.DataFrame] = None, progress_obj: Any = None):
        self.config = config
        self.logger = get_logger("workflow_16s")
        self.project_dir = project_dir
        self.nfc_handler = nfc_handler
        self.nfc_facilities_df = nfc_facilities_df if nfc_facilities_df is not None else pd.DataFrame()
        self.progress_obj = progress_obj 
        
    async def pre_flight_checks(self, dataset_id: str, metadata: pd.DataFrame):
        self.logger.info(f" 🔍 [{dataset_id}] Verifying files and scouting environment...")
        pass

    def _unpack_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.size == 0: return np.nan
            elif value.size == 1: value = value.item()
            else: return value.tolist()
        if isinstance(value, list):
            if not value: return np.nan
            cleaned_list = [self._unpack_value(v) for v in value]
            return cleaned_list[0] if len(cleaned_list) == 1 else cleaned_list
        try:
            if pd.isna(value): return np.nan
        except (TypeError, ValueError): pass
        if isinstance(value, str):
            s = value.strip()
            if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                try: return self._unpack_value(ast.literal_eval(s))
                except (ValueError, SyntaxError): return value
        return value.get('value') if isinstance(value, dict) and 'value' in value else value

    def _clean_and_unpack_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if col not in ['run_accession', '#sampleid']: 
                cleaned_df[col] = cleaned_df[col].apply(self._unpack_value)
        return cleaned_df

    def _aggregate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'run_accession' not in df.columns or df.empty: return df
        env_prefixes = ['SoilGrids_', 'Meteostat_', 'NASA_', 'OpenMeteo_']
        def custom_list_aggregator(series: pd.Series) -> Union[List[Any], float]:
            unique_items, seen_reps, all_items = [], set(), []
            for item in series.dropna():
                if isinstance(item, list): all_items.extend(item)
                else: all_items.append(item)
            for item in all_items:
                try: rep = json.dumps(item, sort_keys=True)
                except TypeError: rep = str(item)
                if rep not in seen_reps: seen_reps.add(rep); unique_items.append(item)
            return unique_items if unique_items else np.nan
        agg_dict = {}
        for col in df.columns:
            if col == 'run_accession': continue
            if col in ['gbif_id', 'species', 'scientific_name', 'attributes']: agg_dict[col] = custom_list_aggregator
            elif any(col.startswith(prefix) for prefix in env_prefixes) and pd.api.types.is_numeric_dtype(df[col]): agg_dict[col] = 'mean'
            else: agg_dict[col] = 'first'
        return df.groupby('run_accession').agg(agg_dict).reset_index()
    
    async def _ensure_fastq_files_exist(self, run_accessions_needed, existing_run_paths, dataset_id, ena_runs_df):
        runs_to_redownload = []
        loop = asyncio.get_running_loop()
        for run_id in run_accessions_needed:
            paths = existing_run_paths.get(run_id, [])
            if not paths or not all(p.exists() for p in paths): 
                runs_to_redownload.append(run_id); continue
            try:
                expected_md5s = str(ena_runs_df.loc[run_id, 'fastq_md5']).split(';')
                if len(paths) != len(expected_md5s): runs_to_redownload.append(run_id); continue
                is_intact = True
                for file_path, expected_md5 in zip(paths, expected_md5s):
                    match = await loop.run_in_executor(None, verify_md5, file_path, expected_md5)
                    if not match: is_intact = False; break
                if not is_intact:
                    for p in paths: p.unlink(missing_ok=True)
                    runs_to_redownload.append(run_id)
            except KeyError: pass
        if runs_to_redownload:
            runs_to_fetch_df = ena_runs_df[ena_runs_df.index.isin(runs_to_redownload)]
            if not runs_to_fetch_df.empty:
                fetcher = SequenceFetcher(fastq_dir=str(self.project_dir.raw_data / dataset_id / "seqs" / "raw"), progress_obj=self.progress_obj)
                downloaded_paths = await fetcher.download_run_fastq_async(runs_to_fetch_df)
                for run_id, paths in downloaded_paths.items(): existing_run_paths[run_id] = [Path(p) for p in paths]
        return existing_run_paths

    def _write_sample_metadata(self, subset_id, partition_metadata):
        qiime_dir = self.project_dir.qiime / subset_id
        qiime_dir.mkdir(parents=True, exist_ok=True)
        metadata_to_save = partition_metadata.copy()
        for col in metadata_to_save.select_dtypes(include=['object']).columns:
            if any(isinstance(val, (list, dict)) for val in metadata_to_save[col].dropna()): metadata_to_save[col] = metadata_to_save[col].astype(str)
        str_cols = metadata_to_save.select_dtypes(include=['object']).columns
        metadata_to_save[str_cols] = metadata_to_save[str_cols].replace(r'[\r\n]+', ' ', regex=True)
        if '#SampleID' in metadata_to_save.columns: metadata_to_save.drop(columns=['#SampleID'], inplace=True)
        metadata_to_save.insert(0, '#SampleID', metadata_to_save['run_accession'])
        metadata_to_save.set_index('#SampleID', inplace=True)
        metadata_to_save.to_csv(qiime_dir / "metadata.tsv", sep='\t')
        return qiime_dir / "metadata.tsv"

    async def process_partition(self, group, dataset, params, all_run_paths, ena_runs, subset_id, anndata_path):
        old_tax_df = None

        def _first_non_empty(df: pd.DataFrame, candidates: List[str]) -> Optional[Any]:
            for col in candidates:
                if col not in df.columns:
                    continue
                series = df[col].dropna()
                if series.empty:
                    continue
                value = series.iloc[0]
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                return value
            return None
        
        if anndata_path.exists():
            try:
                import anndata as ad
                check = ad.read_h5ad(anndata_path, backed='r')
                is_dense = not sparse.issparse(check.X)
                
                target_tax = self.config.qiime2.per_dataset.taxonomy.classify_method
                used_tax = "unknown"
                
                prov_str = str(check.uns.get('provenance', {})).lower()
                if "'classify_method': 'gg2'" in prov_str or "'taxonomy_strategy': 'gg2'" in prov_str:
                    used_tax = "gg2"
                elif "'classify_method': 'sklearn'" in prov_str or "'taxonomy_strategy': 'sklearn'" in prov_str:
                    used_tax = "sklearn"
                elif "'classify_method': 'sepp'" in prov_str or "'taxonomy_strategy': 'sepp'" in prov_str:
                    used_tax = "sepp"
                else:
                    if 'taxonomy_gg2' in check.var.columns or 'lineage' in check.var.columns: used_tax = "gg2"
                    elif 'taxonomy_sklearn' in check.var.columns or 'taxonomy' in check.var.columns: used_tax = "sklearn"
                
                tax_mismatch = used_tax != target_tax
                
                if not is_dense and not tax_mismatch:
                    self.logger.info(f"✅ {subset_id} is already Sparse and matches target taxonomy ({target_tax}). Skipping.")
                    return anndata_path
                
                label = "DENSE" if is_dense else f"TAX_MISMATCH ({used_tax} -> {target_tax})"
                self.logger.warning(f"🔄 {subset_id} is {label}. Triggering update workflow.")
                
                tax_cols = [c for c in check.var.columns if 'tax' in c.lower() or 'lineage' in c.lower()]
                if tax_cols:
                    old_tax_df = check.var[tax_cols].copy()
                
                del check
            except Exception as e:
                self.logger.error(f"⚠️ Could not audit {anndata_path}: {e}. Overwriting for safety.")
        
        subset_dir = SubSet(self.project_dir, subset_id)
        qiime_dir = QIIME(subset_dir)
        
        all_run_paths = await self._ensure_fastq_files_exist(group['run_accession'].tolist(), all_run_paths, dataset, ena_runs)
        project_raw_dir = self.project_dir.raw_data / dataset
        if not run_quality_guard(project_raw_dir):
            error_msg = f"❌ [QUALITY GUARD] {subset_id} failed: Total reads below 1,000 or files missing."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        subset_metadata, raw_data_dir = group.copy(), RawData(subset_dir)
        temp_metadata_path = raw_data_dir.metadata_tsv
        MetadataManager.export_tsv(subset_metadata, temp_metadata_path)
        
        try:
            subset_metadata = await process_metadata(subset_metadata, temp_metadata_path, self.config)
        except Exception as e:
            self.logger.warning(f"Metadata preprocessing failed for {subset_id}: {e}", exc_info=True)
        
        if self.nfc_handler and not (pd.DataFrame(self.nfc_facilities_df).empty if isinstance(self.nfc_facilities_df, list) else (pd.DataFrame(self.nfc_facilities_df).empty if isinstance(self.nfc_facilities_df, list) else self.nfc_facilities_df.empty)):
            try: subset_metadata = self.nfc_handler.annotate_samples(subset_metadata)
            except Exception as e: self.logger.warning(f"NFC annotation failed for {subset_id}: {e}")
        
        subset_metadata['collection_date'] = pd.to_datetime(subset_metadata.get('collection_date'), errors='coerce')
        candidate_cols = [c for c in subset_metadata.columns if any(x in c.lower() for x in ['date', 'time', 'start', 'created'])]
        for col in candidate_cols:
            if subset_metadata['collection_date'].notna().all(): break
            subset_metadata['collection_date'] = subset_metadata['collection_date'].fillna(pd.to_datetime(subset_metadata[col], errors='coerce'))
            
        if subset_metadata['collection_date'].isna().any():
            project_fallback = self.config.upstream.get('current_fallback_date', '2020-01-01')
            missing_count = subset_metadata['collection_date'].isna().sum()
            subset_metadata['collection_date'] = subset_metadata['collection_date'].fillna(pd.to_datetime(project_fallback))

        subset_metadata['collection_date'] = pd.to_datetime(subset_metadata['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        MetadataManager.export_tsv(subset_metadata, temp_metadata_path)
        
        try:
            if "lat" in subset_metadata.columns and "lon" in subset_metadata.columns:
                valid_rows = subset_metadata.dropna(subset=["lat", "lon"])
                if not valid_rows.empty:
                    data_collector = EnvironmentalDataCollector(config=self.config, progress_obj=self.progress_obj)
                    enriched_env_df = await data_collector.collect_for_metadata(valid_rows[["collection_date", "lat", "lon", "run_accession"]])
                    if enriched_env_df is not None and not enriched_env_df.empty:
                        subset_metadata['lat_round'] = pd.to_numeric(subset_metadata['lat']).round(2)
                        subset_metadata['lon_round'] = pd.to_numeric(subset_metadata['lon']).round(2)
                        subset_metadata = subset_metadata.merge(
                            enriched_env_df, 
                            left_on=['lat_round', 'lon_round', 'collection_date'], 
                            right_on=['lat', 'lon', 'collection_date'], 
                            how='left',
                            suffixes=('', '_env_drop')
                        ).drop(columns=['lat_round', 'lon_round'])
            
            arkin_df = await run_arkin_enrichment(
                metadata_path=temp_metadata_path, project_dir=self.project_dir, config=self.config,
                progress_obj=self.progress_obj
            )
            if arkin_df is not None and not arkin_df.empty:
                arkin_df['run_accession_list'] = arkin_df['associated_sample_ids'].str.split(', ')
                arkin_df = arkin_df.explode('run_accession_list').rename(columns={'run_accession_list': 'run_accession'})
                subset_metadata = pd.merge(
                    subset_metadata, 
                    arkin_df.drop(columns=['associated_sample_ids']), 
                    on='run_accession', 
                    how='left', 
                    suffixes=('', '_arkin')
                )
        except Exception as e:
            self.logger.warning(f"Environmental enrichment failed for {subset_id}: {e}", exc_info=True)

        manager = MetadataManager(subset_metadata, self.config)
        subset_metadata = manager.harmonize(similarity_threshold=85)
        subset_metadata = self._clean_and_unpack_data(self._aggregate_rows(subset_metadata))
        
        metadata_path = self._write_sample_metadata(subset_id, subset_metadata)
        layout, manifest_path = write_qiime_manifest(self.project_dir.qiime, subset_id, subset_metadata, all_run_paths)

        library_layout = params.get('library_layout') or layout or _first_non_empty(
            subset_metadata,
            ['library_layout', 'layout', 'LibraryLayout']
        )
        if library_layout is None:
            raise KeyError(
                f"Missing required library layout for {subset_id}. "
                f"Checked params/layout and metadata columns: library_layout, layout, LibraryLayout"
            )

        fwd_primer = params.get('pcr_primer_fwd_seq') or params.get('fwd_primer_seq') or _first_non_empty(
            subset_metadata,
            ['pcr_primer_fwd_seq', 'fwd_primer_seq', 'forward_primer']
        )
        if fwd_primer is None:
            self.logger.warning(
                f"No forward primer found for {subset_id}. "
                f"Proceeding without adapter trimming. "
                f"Checked params and metadata columns: pcr_primer_fwd_seq, fwd_primer_seq, forward_primer"
            )
            fwd_primer = "NONE"

        is_paired = str(library_layout).lower() == 'paired'
        rev_primer = None
        if is_paired:
            rev_primer = params.get('pcr_primer_rev_seq') or params.get('rev_primer_seq') or _first_non_empty(
                subset_metadata,
                ['pcr_primer_rev_seq', 'rev_primer_seq', 'reverse_primer']
            )
            if rev_primer is None:
                self.logger.warning(
                    f"No reverse primer found for paired-end subset {subset_id}. "
                    f"Proceeding without adapter trimming. "
                    f"Checked params and metadata columns: pcr_primer_rev_seq, rev_primer_seq, reverse_primer"
                )
                rev_primer = "NONE"

        qiime_subset = dict(params)
        qiime_subset['library_layout'] = str(library_layout).lower()
        qiime_subset['pcr_primer_fwd_seq'] = str(fwd_primer)
        if rev_primer is not None:
            qiime_subset['pcr_primer_rev_seq'] = str(rev_primer)
        
        target_region = subset_id.split('.')[3]
        size_map = {
            "V4": 253, "V3-V4": 428, "V1-V2": 310, "V4-V5": 374, "V1-V3": 490, 
            "V5-V7": 315, "V6-V8": 430, "V7-V9": 410, "V1-V9": 1450, "FULL-LENGTH": 1450,
            "V1": 250, "V2": 250, "V3": 200, "V5": 200, "V6": 100, "V7": 100, "V8": 100, "V9": 150
        }
        expected_size = size_map.get(target_region.upper(), 253)
        
        process_start = datetime.datetime.now()
        
        detected_adapters = qiime_subset.get('detected_adapters', [])
        if not detected_adapters:
            scanner = AdapterScanner()
            for run_paths in all_run_paths.values():
                if run_paths and run_paths[0].exists():
                    detected_adapters = scanner.scan_fastq(str(run_paths[0]))
                    break
            qiime_subset['detected_adapters'] = detected_adapters
        
        artifact_paths = await execute.seqs_to_features(
            app_config=self.config, subset=qiime_subset, qiime_dir=qiime_dir.main, metadata_path=metadata_path, 
            manifest_path=manifest_path, anndata_dir=anndata_path.parent, subset_id=subset_id, 
            expected_amplicon_size=expected_size, progress_obj=self.progress_obj,
            detected_adapters=detected_adapters
        )
        
        tax_tsv_path = artifact_paths.get("taxonomy_tsv", qiime_dir.main / "taxonomy.tsv")
        if not tax_tsv_path.exists():
            raise FileNotFoundError(f"❌ Taxonomy export failed for {subset_id}. Check QIIME2 logs.")
        
        adata = create_anndata_from_qiime_artifacts(
            artifact_paths["feature_table_biom"], tax_tsv_path, 
            artifact_paths["rep_seqs_fasta"], artifact_paths.get("rooted_tree_nwk", qiime_dir.main / "tree.nwk"), metadata_path
        )
        
        if old_tax_df is not None and not (pd.DataFrame(old_tax_df).empty if isinstance(old_tax_df, list) else (pd.DataFrame(Noneold_tax_df).empty if isinstance(Noneold_tax_df, list) else (pd.DataFrame(NoneNoneold_tax_df).empty if isinstance(NoneNoneold_tax_df, list) else NoneNoneold_tax_df.empty))):
            for col in old_tax_df.columns:
                if col not in adata.var.columns:
                    new_col_name = col if "legacy" in col.lower() else f"legacy_{col}"
                    adata.var[new_col_name] = old_tax_df[col]
                    
        adata = embed_provenance(adata, subset_id, self.config, start_time=process_start, qiime2_env='qiime2-amplicon-2025.7')
        safe_write_h5ad(adata, anndata_path)
        validate_anndata_file(anndata_path, subset_id)
        return anndata_path
