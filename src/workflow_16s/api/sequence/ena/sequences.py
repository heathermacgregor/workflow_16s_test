from __future__ import print_function  # For Python 2/3 compatibility
# workflow_16s/api/ena/sequences.py

import asyncio
import aiohttp
import ftplib
import gzip
import logging
import os
import shutil
import subprocess
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
import urllib3
from Bio import SeqIO
from rich.progress import TaskID
from requests.adapters import HTTPAdapter

import workflow_16s.custom_tmp_config
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger

from .cache import SQLiteCacheManager as CacheManager 

logger = get_logger("workflow_16s")
class MetadataFetcher:
    def __init__(
        self, 
        base_url: str = "https://www.ebi.ac.uk/ena/portal/api",
        cache_manager: Optional[CacheManager] = None, # ADD THIS
        progress_obj: Any = None 
    ):
        self.base_url = base_url
        self.cache_manager = cache_manager # Assign
        self.session = self._create_session(10, 1)
        self.logger = get_logger("workflow_16s")
        self.progress = progress_obj if progress_obj else get_progress_bar()
        self._standalone = progress_obj is None 
        self._auto_start = auto_start_progress
        if self._auto_start and self._standalone:
            self.progress.start()

    def _create_session(self, retries: int, backoff_factor: int) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        retry_strategy = urllib3.Retry(
            total=retries, 
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.mount("http://", adapter)
        return session

    @contextmanager
    def track(self):
        """Dashboard-safe context manager."""
        started_locally = False
        if self._standalone and not self._auto_start:
            self.progress.start()
            started_locally = True
        try: yield
        finally:
            if started_locally:
                self.progress.stop()

    def _get_data(self, endpoint: str, params: dict) -> pd.DataFrame:
        # --- NEW SQLITE CACHE CHECK ---
        cache_key = None
        if self.cache_manager:
            # Create a unique key based on the URL and params
            param_str = "-".join(f"{k}:{v}" for k, v in sorted(params.items()))
            cache_key = f"metadata_{endpoint}_{param_str}"
            
            # Use the existing async-to-sync bridge if needed, 
            # or just call it directly if in an async context
            cached_df = asyncio.run(self.cache_manager.get(cache_key))
            if cached_df is not None:
                self.logger.debug(f"Cache HIT for metadata: {endpoint}")
                return cached_df

        # --- EXISTING NETWORK FETCH ---
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, stream=True)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), sep="\t", low_memory=False)
            
            # --- SAVE TO SQLITE ---
            if self.cache_manager and not df.empty:
                asyncio.run(self.cache_manager.set(cache_key, df))
            
            return df
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def get_study_metadata(self, ena_study_accession: str) -> pd.DataFrame:
        """Fetch metadata for a study accession."""
        params = {
            "accession": ena_study_accession, "fields": "all",
            "format": "tsv", "download": "true", "limit": 0
        }
        return self._get_data("filereport", params)

    def get_sample_metadata(self, ena_sample_accession: str) -> pd.DataFrame:
        """Fetch metadata for a sample accession."""
        params = {
            "result": "sample", 
            "query": f'accession="{ena_sample_accession}"'
        }
        return self._get_data("search", params)

    def get_sample_metadata_concurrent(
        self, sample_task: TaskID, 
        ena_sample_accessions: List[str],
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        Fetch metadata for multiple samples concurrently.

        Args:
            sample_task:           Progress task ID for tracking sample
            sample_task:           Progress task ID for tracking sample downloads.
            ena_sample_accessions: List of sample accessions to fetch.
            max_workers:           Maximum number of concurrent workers.

        Returns:
            Combined DataFrame of sample metadata
        """
        dfs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_sample_metadata, acc): acc 
                for acc in ena_sample_accessions
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result.empty: dfs.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to fetch sample: {e}")
                finally:
                    self.progress.advance(sample_task)
        return pd.concat(dfs).drop_duplicates() if dfs else pd.DataFrame()

    def get_study_and_sample_metadata(
        self, 
        ena_study_accession: str, 
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        Get combined study and sample metadata.
        
        Args:
            ena_study_accession: ENA study accession to fetch metadata for.
            max_workers:         Maximum number of concurrent workers for sample metadata fetching.
        
        Returns:
            DataFrame containing combined study and sample metadata.
        """
        with self.track():
            parent_task = self.progress.add_task(
                f"Processing {ena_study_accession}", 
                total=3
            )

            try:
                # Get study metadata
                study_task = self.progress.add_task(
                    "Fetching study metadata", 
                    parent=parent_task, 
                    total=1
                )
                study_df = self.get_study_metadata(ena_study_accession)
                self.progress.update(study_task, completed=1)
                self.progress.remove_task(study_task)
                self.progress.advance(parent_task)

                # Get sample metadata
                samples = study_df["sample_accession"].dropna().unique().tolist()
                sample_task = self.progress.add_task(
                    "Fetching sample metadata", 
                    parent=parent_task,
                    total=len(samples)
                )
                sample_df = self.get_sample_metadata_concurrent(
                    sample_task, 
                    samples, 
                    max_workers
                )
                self.progress.remove_task(sample_task)
                self.progress.advance(parent_task)

                # Merge study and sample metadata
                merge_task = self.progress.add_task(
                    "Merging study and sample metadata",
                    parent=parent_task, 
                    total=1
                )
                merged_df = study_df.merge(
                    sample_df, 
                    on="sample_accession",
                    how="left", 
                    suffixes=("_study", "")
                )
                self.progress.update(merge_task, completed=1)
                self.progress.remove_task(merge_task)
                self.progress.advance(parent_task)

                return merged_df

            finally:
                self.progress.remove_task(parent_task)

    def __enter__(self):
        if not self._auto_start:
            self.progress.start()
            self._auto_start = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_start:
            self.progress.stop()
            self._auto_start = False

class SequenceFetcher:
    """
    Asynchronous ENA Sequence Fetcher with GZIP integrity verification.
    """

    def __init__(
        self, 
        fastq_dir: Union[str, Path], 
        retries: int = 5,
        max_concurrent: int = 5,
        progress_obj: Any = None 
    ):
        self.fastq_dir = Path(fastq_dir)
        self.fastq_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.max_concurrent = max_concurrent
        self.logger = get_logger("workflow_16s")
        
        # 🟢 FIX: Reference the master progress bar without taking ownership
        self.progress = progress_obj if progress_obj else get_progress_bar()
        self._standalone = progress_obj is None

    def _is_gzip_valid(self, filepath: Path) -> bool:
        """🟢 INTEGRITY SCOUT: Checks if the FASTQ is corrupted or truncated."""
        if not filepath.exists() or filepath.stat().st_size == 0:
            return False
        try:
            # -t tests the integrity of the compressed file
            subprocess.run(['gzip', '-t', str(filepath)], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def fetch_file(self, session: aiohttp.ClientSession, url: str, dest: Path, task_id: Any) -> bool:
        """Asynchronously downloads a single file with retries and validation."""
        if self._is_gzip_valid(dest):
            return True

        url = f"https://{url}" if not url.startswith('http') else url
        
        for attempt in range(self.retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(dest, "wb") as f:
                            while True:
                                chunk = await response.content.read(1024 * 1024) # 1MB chunks
                                if not chunk: break
                                f.write(chunk)
                        
                        # 🟢 VERIFY AFTER DOWNLOAD
                        if self._is_gzip_valid(dest):
                            return True
                        else:
                            self.logger.warning(f"⚠️ Corrupted download for {dest.name}. Retrying...")
                            dest.unlink(missing_ok=True)
            except Exception as e:
                self.logger.debug(f"Attempt {attempt+1} failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt) # Exponential backoff
        
        return False

    async def download_run_fastq_async(self, metadata: pd.DataFrame) -> Dict[str, List[str]]:
        """Main entry point for concurrent downloads."""
        results = {}
        
        # 🟢 FIX: Only start if we are running this as a standalone script
        if self._standalone: self.progress.start()
        
        main_task = self.progress.add_task(
            "[bold cyan]📥 Downloading FASTQs", 
            total=len(metadata)
        )

        connector = aiohttp.TCPConnector(limit_per_host=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for row in metadata.itertuples():
                urls = str(row.fastq_ftp).split(";")
                run_acc = str(row.run_accession)
                
                # Internal worker to handle a single run (1 or 2 files)
                async def process_run(acc, f_urls):
                    paths = []
                    for idx, url in enumerate(f_urls, 1):
                        fname = f"{acc}_{idx}.fastq.gz" if len(f_urls) > 1 else f"{acc}.fastq.gz"
                        dest = self.fastq_dir / fname
                        if await self.fetch_file(session, url, dest, main_task):
                            paths.append(str(dest))
                    
                    self.progress.advance(main_task)
                    return {acc: paths}

                tasks.append(process_run(run_acc, urls))

            # Execute all downloads concurrently
            download_results = await asyncio.gather(*tasks)
            for res in download_results:
                results.update(res)

        if self._standalone: 
            self.progress.stop()
        else:
            # 🟢 Clean up the task so it doesn't clutter the dashboard
            self.progress.remove_task(main_task)

        return results
