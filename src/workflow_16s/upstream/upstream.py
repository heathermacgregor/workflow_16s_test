# workflow_16s/upstream/upstream.py

import argparse
import asyncio
import json
import logging
import time
import yaml
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import anndata as ad
import pandas as pd
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group, Console
from rich.layout import Layout
from rich.text import Text

console = Console()

from workflow_16s.api.ena import SQLiteCacheManager as EnaCacheManager, ENAClient, ENAFetcher, get_counts_bulk_async
from workflow_16s.api.environmental_data import ArkinEnvAgents, EnvironmentalDataCollector, NFCFacilitiesHandler
from workflow_16s.api.geospatial.universal_finder import UniversalFacilityFetcher
from workflow_16s.api.publication.fetcher import PublicationFetcher
from workflow_16s.api.qiime import execute
from workflow_16s.config import AppConfig
from workflow_16s.constants import DATASETS_TO_SKIP
from workflow_16s.upstream.metadata.partition import DatasetPartition
from workflow_16s.upstream.sequences.analysis import PrimerFinder
from workflow_16s.upstream.sequences.probebase import import_and_save_database
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.io import load_datasets_list, load_datasets_info
from workflow_16s.utils.logger import get_logger, setup_logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    
OFFICIAL_TITLES = {
    "nfc_facilities": "☢️ Nuclear Facility Geospatial Sweep",
    "sample_discovery": "🔍 Semantic Keyword Scouting",
    "sieve": "⚖️ Dataset Validation & Sieve",
    "deep_metadata": "📦 Fetching Deep Metadata (Runs, Samples, Taxa)",
    "qiime2": "🧬 QIIME 2 Core Processing",
    "meta_analysis": "🔗 Phylogenetic Meta-Analysis"
}

def get_computer_name():
    try:
        import socket
        return socket.gethostname()
    except:
        import platform
        return platform.node()

class Upstream:
    def __init__(self, config: AppConfig):
        self._last_web_export = 0.0
        self.config = config
        self.project_dir = Project(config)
        self.start_time = time.time()
        self.logger = get_logger("workflow_16s")
        self.ena_cache_manager = EnaCacheManager(cache_dir=self.project_dir.cache / "ena_metadata")
        self.ena_fetcher = ENAFetcher(email=self.config.credentials.ena_email, max_concurrent=10, log_interval=10, cache_manager=self.ena_cache_manager)
        self.ena_client = ENAClient(self.config, fetcher=self.ena_fetcher)
        
        self._initialize_datasets()
        self.log_buffer = deque(maxlen=20) 
        self._setup_ui_logger()
        
        self.ui_stats = {
            "node_name": get_computer_name(),
            "current_action": "INITIALIZING...", "sieve_total": 0, "sieve_passed": 0, 'sieve_failed': 0,
            'fail_low_count': 0, 'fail_metadata': 0, 'fail_primers': 0, "success_h5ad": 0, "failed_h5ad": 0, 
            "total_samples_found": 0, "nfc_samples": 0, "semantic_samples": 0, "osm_samples": 0,
            "base_projects": len(self.datasets)
        }
        
        self._probebase_setup()
        primer_finder = PrimerFinder(self.config.paths.primer_db)
        self.region_to_pairs_map = primer_finder.get_primer_pairs_for_regions()
        
        self.publication_fetcher = PublicationFetcher(
            self.config, 
            cache_path=str(self.project_dir.cache / "publications.db")
        )
        self.nfc_handler = NFCFacilitiesHandler(self.config, progress_obj=self.progress_obj, fetcher=self.ena_fetcher) if self.config.nfc_facilities.enabled else None
        self.nfc_facilities_df = pd.DataFrame()
        self.env_collector = EnvironmentalDataCollector(self.config)
        self.arkin_agents = ArkinEnvAgents(self.config)
        
        self.processed_subsets, self.failed_subsets, self.data_objects = [], [], []

    def _initialize_datasets(self):
        if self.config.paths.dataset_list and self.config.paths.dataset_info:
            self.datasets = load_datasets_list(self.config.paths.dataset_list)
            self.datasets_info = load_datasets_info(self.config.paths.dataset_info)
            self._lookup_dict = {}
            for _, row in self.datasets_info.iterrows():
                is_ena = str(row.get('dataset_type', 'ENA')).upper() == 'ENA'
                acc = str(row.get('ena_project_accession', '')).strip().upper()
                did = str(row.get('dataset_id', '')).strip().upper()
                if acc and (acc not in self._lookup_dict or is_ena): self._lookup_dict[acc] = row
                if did and (did not in self._lookup_dict or is_ena): self._lookup_dict[did] = row
        else:
            self.datasets, self.datasets_info, self._lookup_dict = [], pd.DataFrame(), {}

    def _setup_ui_logger(self):
        from workflow_16s.utils.progress import get_progress_bar
        
        class DequeHandler(logging.Handler):
            def __init__(self, buffer):
                super().__init__()
                self.buffer = buffer
            def emit(self, record):
                self.buffer.append(self.format(record))
                
        ui_log_handler = DequeHandler(self.log_buffer)
        ui_log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(ui_log_handler)

        self.progress_obj = get_progress_bar()
        self.progress_obj.auto_refresh = False

    def _get_dashboard_renderable(self):
        """Constructs the new Amplicon Workflow Process Monitor Layout."""
        self._export_web_state()
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=24), # Increased size to fit system stats
            Layout(name="body")
        )
        
        uptime = str(datetime.now() - datetime.fromtimestamp(self.start_time)).split('.')[0]
        
        s_total = max(1, self.ui_stats.get('sieve_total', 0))
        s_passed = self.ui_stats.get('sieve_passed', 0)
        q_success = self.ui_stats.get('success_h5ad', 0)
        
        stats_table = Table.grid(expand=True)
        stats_table.add_column(justify="left", style="cyan")
        stats_table.add_column(justify="right", style="white")

        # --- SYSTEM STATS ---
        stats_table.add_row("[bold underline]SYSTEM RESOURCES", "")
        stats_table.add_row(" 🖥️ Node:", f"[magenta]{self.ui_stats['node_name']}")
        if HAS_PSUTIL:
            main_proc = psutil.Process()
            cpu = psutil.cpu_percent(interval=None)
            ram = main_proc.memory_info().rss / (1024 * 1024 * 1024)
            threads = main_proc.num_threads()
            children = len(main_proc.children(recursive=True))
            stats_table.add_row(" ⚙️ CPU Usage:", f"[yellow]{cpu:.1f}%")
            stats_table.add_row(" 🧠 RAM (RSS):", f"[yellow]{ram:.2f} GB")
            stats_table.add_row(" 🧵 Threads/Procs:", f"[yellow]{threads} / {children} children (PID: {main_proc.pid})")
        stats_table.add_row("", "")

        # --- DISCOVERY SOURCES ---
        stats_table.add_row("[bold underline]DISCOVERY SOURCE BREAKDOWN", "")
        stats_table.add_row(" 📂 Base Projects (CSV):", f"[white]{self.ui_stats.get('base_projects', 0)}")
        if self.config.nfc_facilities.enabled:
            stats_table.add_row(" ☢️ NFC Facility Hits:", f"[yellow]{self.ui_stats.get('nfc_samples', 0)}")
        if self.config.sample_discovery.enabled:
            stats_table.add_row(" 🔍 Semantic Hits:", f"[cyan]{self.ui_stats.get('semantic_samples', 0)}")
            if self.config.sample_discovery.osm_features:
                stats_table.add_row(" 🛰️ OSM Geospatial Hits:", f"[blue]{self.ui_stats.get('osm_samples', 0)}")
        stats_table.add_row(" 🌎 Total Unique Projects:", f"[bold green]{len(self.datasets)}")
        stats_table.add_row("", "")

        # --- SIEVE & PIPELINE STATUS ---
        stats_table.add_row("[bold underline]PIPELINE STATUS", "")
        stats_table.add_row(" ✅ Passed Sieve:", f"[green]{s_passed}/{s_total} ({ (s_passed/s_total)*100:.1f}%)")
        stats_table.add_row(" 🧪 Successful .h5ad Files:", f"[bold green]{q_success}/{max(1, s_passed)}")
        stats_table.add_row(" ⏳ Engine Status:", f"[bold yellow]{self.ui_stats.get('current_action', 'IDLE')}")

        header_group = Group(
            self.progress_obj,
            Panel(stats_table, title="[bold white]Live Telemetry", border_style="blue")
        )
        layout["header"].update(header_group)

        # Log Body
        log_text = Text()
        for line in self.log_buffer:
            if "INFO" in line: color = "green"
            elif "WARNING" in line: color = "yellow"
            elif "ERROR" in line: color = "red"
            else: color = "white"
            log_text.append(line + "\n", style=color)
            
        layout["body"].update(Panel(log_text, title="[bold white]Console Feed", border_style="cyan"))
        return layout

    def _export_web_state(self):
        current_time = time.time()
        if current_time - self._last_web_export < 2.0:
            return
        self._last_web_export = current_time

        try:
            db_path = self.project_dir.cache / "env" / "cache.db"
            db_size = db_path.stat().st_size if db_path.exists() else 0
            ena_db = self.project_dir.cache / "ena_metadata" / "ena_cache.db"
            ena_size = ena_db.stat().st_size if ena_db.exists() else 0
            db_cache_size = f"{(db_size + ena_size) / (1024 * 1024):.1f} MB"
        except Exception:
            db_cache_size = "0 MB"

        process_tree = []
        if HAS_PSUTIL:
            try:
                current_process = psutil.Process()
                process_tree.append({
                    "pid": current_process.pid,
                    "name": current_process.name(),
                    "cpu": current_process.cpu_percent(interval=None),
                    "ram": round(current_process.memory_percent(), 1)
                })
                for p in current_process.children()[:5]: 
                    process_tree.append({
                        "pid": p.pid,
                        "name": p.name(),
                        "cpu": p.cpu_percent(interval=None),
                        "ram": round(p.memory_percent(), 1)
                    })
            except Exception: pass

        state = {
            "stats": {
                "datasets_processed": self.ui_stats.get('success_h5ad', 0),
                "total_samples": self.ui_stats.get('total_samples_found', 0),
                "total_asv_features": self.ui_stats.get('total_asv_features', 0),
                "db_cache_size": db_cache_size,
                "current_action": self.ui_stats.get('current_action', 'Waiting for pipeline...')
            },
            "tasks": [
                {
                    "description": t.description,
                    "total": t.total or 0,
                    "completed": t.completed or 0,
                    "percentage": round(t.percentage or 0, 1) if t.percentage else 0
                } for t in self.progress_obj.tasks
            ],
            "process_tree": process_tree,
            "nfc_hot_sites": self.ui_stats.get('hot_sites', []),
            "console_feed": list(self.log_buffer)
        }
        
        try:
            tmp_path = Path(self.project_dir.main) / "workflow_state_tmp.json"
            final_path = Path(self.project_dir.main) / "workflow_state.json"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            tmp_path.replace(final_path) 
        except Exception as e:
            self.logger.error(f" ❌ JSON Dashboard Export Failed: {e}")

    async def execute(self):
        self._append_to_live_report("STARTUP", "INITIALIZING", 0, "Pipeline starting...")
        with Live(get_renderable=self._get_dashboard_renderable, refresh_per_second=4, screen=True) as live:
            async with self.ena_fetcher as fetcher:
                self.logger.info("🎬 Session opened. Starting consolidated discovery...")

                if self.config.nfc_facilities.enabled:
                    self.ui_stats["current_action"] = "GEOSPATIAL SWEEP"
                    nfc_task = self.progress_obj.add_task(OFFICIAL_TITLES["nfc_facilities"], total=490)
                    await self.nfc(fetcher=fetcher, task_id=nfc_task)

                if self.config.sample_discovery.enabled:
                    self.ui_stats["current_action"] = "SEMANTIC SCOUTING"
                    kw_ids = await self._perform_discovery(fetcher=fetcher)
                    self.datasets.extend(kw_ids)

                self.datasets = sorted(list(set(self.datasets)))
                if not self.datasets: return []

                self.ui_stats["current_action"] = "DATASET VALIDATION (SIEVE)"
                sieve_task = self.progress_obj.add_task(OFFICIAL_TITLES["sieve"], total=len(self.datasets))
                await self.sort_datasets(sieve_task_id=sieve_task)
                
                self.ui_stats["current_action"] = "DEEP METADATA & QIIME"
                await self.process_datasets(live_ui=live)
                
                self.ui_stats["current_action"] = "META-ANALYSIS"
                self._run_metaanalysis()
                
                self.ui_stats["current_action"] = "PIPELINE COMPLETE ✅"
        return self.data_objects
    
    async def nfc(self, fetcher=None, task_id=None):
        if not self.nfc_handler: return
        self.nfc_facilities_df = await self.nfc_handler.nfc_facilities()
        if not getattr(self.config.nfc_facilities, 'fetch_nearby_samples', False) or self.nfc_facilities_df.empty: return

        nearby_df = await self.nfc_handler.get_nearby_samples(fetcher=fetcher, task_id=task_id)
        if not nearby_df.empty:
            threshold = self.config.nfc_facilities.min_samples_per_study
            counts_map = nearby_df.groupby('study_accession').size()
            valid_proximal_projects = counts_map[counts_map >= threshold].index.tolist()
            new_nfc_count = counts_map[valid_proximal_projects].sum()
            self.ui_stats["nfc_samples"] = self.ui_stats.get("nfc_samples", 0) + new_nfc_count
            self.ui_stats["total_samples_found"] += counts_map[valid_proximal_projects].sum()
            self.datasets.extend([p for p in valid_proximal_projects if p not in self.datasets])

        if task_id is not None: self.progress_obj.update(task_id, completed=490, description=f"[bold green]Sweep Complete")

    async def sort_datasets(self, sieve_task_id=None):
        if not self.datasets:
            return

        cache_file = self.project_dir.cache / "valid_datasets_sieve.json"
        
        cached_ids = []
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_ids = json.load(f)
                self.logger.info(f"📂 Loaded {len(cached_ids)} projects from existing sieve cache.")
            except Exception as e:
                self.logger.error(f"Failed to read sieve cache: {e}")

        new_ids = [ds for ds in self.datasets if ds not in cached_ids]

        if not new_ids:
            self.logger.info("📅 No new projects discovered. Proceeding with cached list.")
            self.datasets = cached_ids
            self.ui_stats["sieve_passed"] = len(self.datasets)
            if sieve_task_id:
                self.progress_obj.update(sieve_task_id, completed=len(self.datasets))
            return

        self.logger.info(f"🆕 Found {len(new_ids)} new potential projects. Running validation...")
        
        n = max(1, self.config.sample_discovery.min_samples_per_study)
        
        counts_dict = await get_counts_bulk_async(
            new_ids, 
            self.config.credentials.ena_email, 
            15, 
            50, 
            self.ena_cache_manager, 
            self.progress_obj
        )

        valid_new = [ds_id for ds_id, count in counts_dict.items() if count >= n]
        self.logger.info(f"✅ {len(valid_new)} of the new discoveries passed the sieve.")

        self.datasets = sorted(list(set(cached_ids + valid_new)))

        self.ui_stats["sieve_passed"] = len(self.datasets)
        self.ui_stats["sieve_total"] = self.ui_stats.get("sieve_total", 0) + len(new_ids)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(self.datasets, f, indent=4)
        
        self.logger.info(f"💾 Sieve cache updated. Total projects: {len(self.datasets)}")

    async def process_datasets(self, live_ui: Live):
        if not self.datasets: return []
        
        metadata_task = self.progress_obj.add_task(OFFICIAL_TITLES["deep_metadata"], total=len(self.datasets))
        qiime_task = self.progress_obj.add_task(OFFICIAL_TITLES["qiime2"], total=len(self.datasets))

        batch_payload = {ds: {'dataset_type': 'ENA', 'ena_project_accession': ds, 'dataset_id': ds} 
                         for ds in self.datasets if ds not in DATASETS_TO_SKIP}

        partitioner = DatasetPartition(
            config=self.config, ena_client=self.ena_client, publication_fetcher=self.publication_fetcher,
            env_collector=self.env_collector, arkin_agents=self.arkin_agents, region_to_pairs_map=self.region_to_pairs_map, 
            nfc_handler=self.nfc_handler, nfc_facilities_df=self.nfc_facilities_df, progress_obj=self.progress_obj
        )
        
        LOAD_THRESHOLD = 85.0
        if HAS_PSUTIL and psutil.cpu_percent(interval=None) > LOAD_THRESHOLD:
            self.logger.warning(f"⚠️ High system load detected on '{get_computer_name()}'. Throttling parallel execution...")
            await asyncio.sleep(10)
        
        successful_h5ad_paths, failed_partitions = await partitioner.run(
            batch_payload, ena_cache_manager=self.ena_cache_manager, ui_stats_ref=self.ui_stats, 
            metadata_task_id=metadata_task, qiime_task_id=qiime_task, ui_refresher=live_ui
        )

        self.ui_stats["success_h5ad"] = len(successful_h5ad_paths)
        self.ui_stats["failed_h5ad"] = len(failed_partitions)
        
        for h5ad_path in successful_h5ad_paths:
            try:
                adata = ad.read_h5ad(h5ad_path, backed='r')
                if adata.n_obs > 0:
                    self.data_objects.append(adata)
                    self.processed_subsets.append(h5ad_path.stem)
            except Exception as e:
                self.logger.error(f"Error loading final AnnData {h5ad_path.name}: {e}")

        self.failed_subsets.extend([(f['dataset'], f['error']) for f in failed_partitions])
        return self.processed_subsets
    
    def _run_metaanalysis(self):
        if len(self.processed_subsets) < 2: return
        try:
            execute.phylogenetic_metaanalysis(app_config=self.config, project_dir=self.project_dir.processed_data)
        except Exception as e:
            self.logger.warning(f"Phylogenetic meta-analysis failed: {e}", exc_info=True)

    def _report(self):
        summary = {
            "datasets_total": len(self.datasets),
            "datasets_processed": len(self.processed_subsets),
            "datasets_failed": len(self.failed_subsets),
            "h5ad_success": self.ui_stats.get("success_h5ad", 0),
            "h5ad_failed": self.ui_stats.get("failed_h5ad", 0),
        }
        self.logger.info(f"Pipeline summary: {summary}")
        return summary
    
    async def _perform_discovery(self, fetcher: ENAFetcher) -> List[str]:
        self.logger.info("🔭 Starting semantic keyword expansion and OSM discovery...")
        
        semantic_ids = []
        for env in self.config.sample_discovery.requested_environments:
            self.logger.info(f"🔍 Searching keywords for: {env.name}...")
            
            search_fields = ["description", "study_title", "project_name"]
            sub_queries = []
            
            for k in env.keywords:
                clean_k = k.strip('"')
                field_group = " OR ".join([f'{field}="{clean_k}"' for field in search_fields])
                sub_queries.append(f"({field_group})")
            
            final_query = " OR ".join(sub_queries)
            
            try:
                results = await self.ena_client.search_projects(final_query, limit=1000)
                if results:
                    semantic_ids.extend(results)
                    self.logger.info(f"   ∟ Found {len(results)} projects for {env.name}")
            except Exception as e:
                self.logger.error(f"   ∟ ENA API rejected query for {env.name}: {e}")
        
        # Log unique semantic hits
        self.ui_stats["semantic_samples"] = len(set(semantic_ids))
        
        # 2. OSM Geospatial Discovery
        osm_ids = await self._perform_osm_discovery(fetcher)
        self.ui_stats["osm_samples"] = len(set(osm_ids))
        
        all_new_ids = list(set(semantic_ids + osm_ids))
        return all_new_ids

    async def _perform_osm_discovery(self, fetcher: ENAFetcher) -> List[str]:
        if not self.config.sample_discovery.osm_features:
            return []

        self.logger.info("🛰️ Querying OpenStreetMap via UniversalFacilityFetcher...")
        osm_scout = UniversalFacilityFetcher()
        raw_coords = []

        for feature in self.config.sample_discovery.osm_features:
            target_alias = getattr(feature, 'value', None) or getattr(feature, 'key', None)
            if not target_alias and hasattr(feature, 'dict'):
                f_dict = feature.dict()
                target_alias = f_dict.get('value') or f_dict.get('key')

            if target_alias not in osm_scout.FACILITY_TAGS:
                self.logger.warning(f"  ⚠️ OSM alias '{target_alias}' not found. Skipping.")
                continue

            df_locations = await osm_scout.fetch_locations(target_alias)
            if not df_locations.empty:
                raw_coords.extend(df_locations[['latitude', 'longitude']].values.tolist())

        if not raw_coords:
            return []

        unique_sweep_targets = {}
        for lat, lon in raw_coords:
            grid_key = (round(lat, 2), round(lon, 2))
            if grid_key not in unique_sweep_targets:
                unique_sweep_targets[grid_key] = (lat, lon)

        self.logger.info(f"🎯 Filtered {len(raw_coords)} hits to {len(unique_sweep_targets)} unique sweep zones.")

        osm_project_ids = []
        for _, (lat, lon) in unique_sweep_targets.items():
            try:
                nearby = await fetcher.find_samples_near(lat=lat, lon=lon, radius_km=10)
                if not nearby.empty and 'study_accession' in nearby.columns:
                    osm_project_ids.extend(nearby['study_accession'].unique())
            except Exception as e:
                self.logger.warning(f"OSM proximity lookup failed at ({lat}, {lon}): {e}")
                continue 

        return list(set(osm_project_ids))

    def _apply_primer_restriction(self, predicted_region: str) -> bool: return True

    def _append_to_live_report(self, dataset_id, status, subsets, error=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] [{status}] {dataset_id} (subsets={subsets})"
        if error:
            msg = f"{msg} | error={error}"
        self.ui_stats["current_action"] = status
        if error:
            self.logger.error(msg)
        else:
            self.logger.info(msg)

    def _find_best_match(self, dataset_id: str) -> pd.Series:
        normalized = str(dataset_id).strip().upper()
        direct = self._lookup_dict.get(normalized)
        if direct is not None:
            return direct if isinstance(direct, pd.Series) else pd.Series(direct)

        if self.datasets_info.empty:
            return pd.Series(dtype=object)

        id_col = self.datasets_info.get("dataset_id", pd.Series(dtype=str)).astype(str).str.upper()
        ena_col = self.datasets_info.get("ena_project_accession", pd.Series(dtype=str)).astype(str).str.upper()
        exact = self.datasets_info[(id_col == normalized) | (ena_col == normalized)]
        if not exact.empty:
            return exact.iloc[0]

        contains = self.datasets_info[
            id_col.str.contains(normalized, na=False) | ena_col.str.contains(normalized, na=False)
        ]
        if not contains.empty:
            return contains.iloc[0]

        return pd.Series(dtype=object)

    def _probebase_setup(self):
        primer_db = getattr(self.config.paths, "primer_db", None)
        if not primer_db:
            self.logger.warning("No primer_db path configured; skipping probebase setup.")
            return

        try:
            primer_db_path = Path(primer_db)
        except TypeError:
            self.logger.warning("Configured primer_db path is invalid; skipping probebase setup.")
            return
        if primer_db_path.exists():
            return

        primer_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Primer DB not found at {primer_db_path}. Attempting automatic setup...")
        try:
            import_and_save_database(db_path=primer_db_path)
        except TypeError:
            # Backward compatibility for function signatures that do not accept db_path.
            import_and_save_database()
        except Exception as e:
            self.logger.warning(f"Probebase setup failed: {e}", exc_info=True)

    def _initialize_publication_fetcher(self) -> PublicationFetcher: return self.publication_fetcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/auto/sahara/namib/home/macgregor/amplicon/workflow_16s/config/config.yaml"))
    args = parser.parse_args()
    with open(args.config, 'r') as f: 
        config = AppConfig(**yaml.safe_load(f))
        
    setup_logging(log_dir_path=Project(config).logs) 
    
    asyncio.run(Upstream(config).execute())
