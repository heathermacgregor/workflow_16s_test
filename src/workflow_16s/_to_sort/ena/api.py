from __future__ import print_function  # For Python 2/3 compatibility

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import ftplib
import gzip
import logging
import os
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
import requests
import urllib3
from Bio import SeqIO

# Local Imports
import workflow_16s.custom_tmp_config
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ===================================== FUNCTIONS ===================================== #

class MetadataFetcher:
    """Fetches metadata from the ENA database for a given ENA project accession.

    Args:
        base_url:             Base URL for the ENA API. Defaults to
                              'https://www.ebi.ac.uk/ena/portal/api'.
        retries:              Number of retries for HTTP requests. Defaults
                              to 5.
        backoff_factor:       Backoff factor for retry delays. Defaults to 1.
        auto_start_progress:  Whether to automatically start the progress bar.
                              Defaults to False.
    """

    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/ena/portal/api",
        retries: int = 5,
        backoff_factor: int = 1,
        auto_start_progress: bool = False,
    ):
        self.base_url = base_url
        self.session = self._create_session(retries, backoff_factor)
        self.progress = get_progress_bar()
        self._auto_start = auto_start_progress
        if self._auto_start:
            self.progress.start()

    def _create_session(self, retries: int, backoff_factor: int) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        retry_strategy = urllib3.util.retry.Retry(
            total=retries,
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @contextmanager
    def track(self):
        """Context manager to handle progress display."""
        original_auto_start = self._auto_start
        try:
            if not self._auto_start:
                self.progress.start()
                self._auto_start = True
            yield
        finally:
            if not original_auto_start:
                self.progress.stop()
                self._auto_start = False

    def _get_data(
        self, ena_accession: str, endpoint: str, params: dict
    ) -> pd.DataFrame:
        """Internal method to fetch data with progress tracking."""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, stream=True)
            response.raise_for_status()

            content = []
            for chunk in response.iter_content(chunk_size=8192):
                content.append(chunk)

            return pd.read_csv(
                StringIO(b"".join(content).decode("utf-8")), sep="\t", low_memory=False
            )
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise RuntimeError(f"Request failed: {e}")

    def get_study_metadata(self, ena_study_accession: str) -> pd.DataFrame:
        """Fetch metadata for a study accession."""
        params = {
            "accession": ena_study_accession,
            "result": "read_run",
            "fields": "all",
            "format": "tsv",
            "download": "true",
            "limit": 0,
        }
        return self._get_data(ena_study_accession, "filereport", params)

    def get_sample_metadata(self, ena_sample_accession: str) -> pd.DataFrame:
        """Fetch metadata for a sample accession."""
        params = {
            "result": "sample",
            "query": f'accession="{ena_sample_accession}"',
            "fields": "all",
            "format": "tsv",
            "limit": 0,
        }
        return self._get_data(ena_sample_accession, "search", params)

    def get_sample_metadata_concurrent(
        self, sample_task: int, ena_sample_accessions: List[str], max_workers: int = 5
    ) -> pd.DataFrame:
        """Fetch metadata for multiple samples concurrently.

        Args:
            sample_task:           Progress task ID for tracking sample
                                   downloads.
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
                    if not result.empty:
                        dfs.append(result)
                except Exception as e:
                    logger.error(f"Failed to fetch sample: {e}")
                finally:
                    self.progress.advance(sample_task)

        return pd.concat(dfs).drop_duplicates() if dfs else pd.DataFrame()

    def get_study_and_sample_metadata(
        self, ena_study_accession: str, max_workers: int = 5
    ) -> pd.DataFrame:
        """Get combined study and sample metadata."""
        with self.track():
            parent_desc = f"Processing {ena_study_accession}"
            parent_task = self.progress.add_task(
                _format_task_desc(parent_desc), total=3
            )

            try:
                # Get study metadata
                study_desc = "Fetching study metadata"
                study_task = self.progress.add_task(
                    _format_task_desc(study_desc), 
                    parent=parent_task, 
                    total=1
                )
                study_df = self.get_study_metadata(ena_study_accession)
                self.progress.update(study_task, completed=1)
                self.progress.remove_task(study_task)
                self.progress.advance(parent_task)

                # Get sample metadata
                samples = study_df["sample_accession"].dropna().unique().tolist()
                sample_desc = "Fetching sample metadata"
                sample_task = self.progress.add_task(
                    _format_task_desc(sample_desc),
                    parent=parent_task,
                    total=len(samples)
                )
                sample_df = self.get_sample_metadata_concurrent(
                    sample_task, samples, max_workers
                )
                # self.progress.update(sample_task, completed=1)
                self.progress.remove_task(sample_task)
                self.progress.advance(parent_task)

                # Merge study and sample metadata
                merge_desc = "Merging study and sample metadata"
                merge_task = self.progress.add_task(
                    _format_task_desc(merge_desc), 
                    parent=parent_task, 
                    total=1
                )
                merged_df = study_df.merge(
                    sample_df,
                    on="sample_accession",
                    how="left",
                    suffixes=("_study", ""),
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
    """Fetches sequencing data from the ENA database for given accessions.

    Args:
        fastq_dir:     Directory to save downloaded FASTQ files.
        retries:       Number of retry attempts for downloading.
                       Defaults to 10.
        initial_delay: Initial delay between retries in seconds.
                       Defaults to 5.
        max_workers:   Maximum number of concurrent download threads.
                       Defaults to 8.
    """

    def __init__(
        self,
        fastq_dir: str,
        retries: int = 10,
        initial_delay: int = 5,
        max_workers: int = 8,
    ):
        self.fastq_dir = Path(fastq_dir)
        self.fastq_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.initial_delay = initial_delay
        self.max_workers = max_workers
        self.progress = get_progress_bar()

    def get_run_fastq(
        self, run_accession: str, urls: List[str]
    ) -> Dict[str, List[str]]:
        """Download FASTQ files for a single run accession."""
        file_paths = []
        for url_idx, url in enumerate(urls, 1):
            if not url or str(url).lower() == "nan":
                continue

            ftp_url = f"https://{url}"
            fastq_filename = (
                f"{run_accession}_{url_idx}.fastq.gz"
                if len(urls) > 1
                else f"{run_accession}.fastq.gz"
            )
            fastq_path = self.fastq_dir / fastq_filename

            success = False
            delay = self.initial_delay
            for attempt in range(self.retries):
                try:
                    if fastq_path.exists() and fastq_path.stat().st_size > 0:
                        success = True
                        break

                    urllib.request.urlretrieve(ftp_url, fastq_path)
                    if fastq_path.stat().st_size > 0:
                        success = True
                        break
                    fastq_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(delay)
                    delay *= 2

            if success:
                file_paths.append(str(fastq_path))

        return {run_accession: file_paths}

    def download_run_fastq_concurrent(
        self, metadata: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Download sequencing data concurrently.

        Args:
            metadata: DataFrame containing run accessions and FTP URLs.

        Returns:
            Mapping of run accessions to downloaded file paths
        """
        results = {}
        with self.progress:
            main_desc = "Downloading sequencing data"
            main_task = self.progress.add_task(
                _format_task_desc(main_desc), 
                total=len(metadata)
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_run,
                        row.run_accession,
                        str(row.fastq_ftp).split(";"),
                    ): row.run_accession for row in metadata.itertuples()
                }

                for future in as_completed(futures):
                    run_accession = futures[future]
                    try:
                        result = future.result()
                        results.update(result)
                    except Exception as e:
                        logger.error(f"Failed processing {run_accession}: {e}")
                    finally:
                        self.progress.advance(main_task)

        return results

    def process_run(self, run_accession: str, urls: List[str]) -> Dict[str, List[str]]:
        """Wrapper method for processing a single run."""
        return self.get_run_fastq(run_accession, urls)


class PooledSamplesProcessor:
    def __init__(self, metadata_df: pd.DataFrame, output_dir: Union[str, Path]):
        self.metadata = metadata_df
        self.output_dir = Path(output_dir)
        self.site_records = defaultdict(list)
        self.sample_file_map = {}
        self._create_lookup_dict()
        self.progress = get_progress_bar()

    def _create_lookup_dict(self):
        """Create internal lookup dictionary from metadata."""
        self.lookup_dict = {
            (row["run_accession"], row["barcode_sequence"]): row["#SampleID"]
            for _, row in self.metadata.iterrows()
        }
        
    def process_single_file(self, file_path: Union[str, Path]):
        """Process a single FASTQ.gz file and accumulate site records."""
        try:
            # Extract run_accession from the filename
            file_name = Path(file_path).name
            run_accession = file_name.split(".")[0]

            with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                for record in SeqIO.parse(handle, "fastq"):   
                    barcode = str(record.seq)[:10]
                    if site_id := self.lookup_dict.get((run_accession, barcode)):
                        self.site_records[site_id].append(record)

        except EOFError as e:
            logger.error(f"Corrupted file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    def write_site_files(self):
        """Write records and build sample->file mapping."""
        site_dir = self.output_dir / "site_files"
        site_dir.mkdir(parents=True, exist_ok=True)

        self.sample_file_map.clear()  # Reset mapping on each write

        # Get all unique sample IDs from metadata
        all_sample_ids = self.metadata["#SampleID"].unique()

        for site_id in all_sample_ids:
            records = self.site_records.get(site_id, [])
            output_file = site_dir / f"{site_id}.fastq.gz"
            with gzip.open(output_file, "wt") as handle:
                SeqIO.write(records, handle, "fastq")
            self.sample_file_map[site_id] = output_file  # Store mapping
            logger.info(f"Wrote {len(records)} records to {output_file}")

        return self.sample_file_map  # Return the mapping dictionary

    @staticmethod
    def merge_files(input_files: List[Union[str, Path]], output_file: Union[str, Path]):
        """Merge multiple FASTQ.gz files."""
        with gzip.open(output_file, "wb") as wfd:
            for f in input_files:
                with gzip.open(f, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)

    def organize_input_files(self, raw_dir: Union[str, Path]):
        """Organize raw input files into structured directory."""
        organized_dir = self.output_dir / "organized_inputs"
        organized_dir.mkdir(parents=True, exist_ok=True)

        file_dict = defaultdict(list)
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.endswith(".fastq.gz"):
                    file_dict[file].append(Path(root) / file)

        for file, paths in file_dict.items():
            output_path = organized_dir / file
            if len(paths) > 1:
                logger.info(f"Merging {len(paths)} copies of {file}")
                self.merge_files(paths, output_path)
            else:
                shutil.copy2(paths[0], output_path)

        return organized_dir

    def find_matching_files(self, search_dir: Union[str, Path]):
        """Find FASTQ files matching metadata run_accession entries."""
        search_path = Path(search_dir)
        paths = [p for p in search_path.glob("*.fastq.gz") if "trimmed" not in str(p)]

        file_map = {}
        for run_id in self.metadata["run_accession"].unique():
            matches = [p for p in paths if str(run_id) in str(p)]
            file_map[run_id] = matches if matches else []

        return file_map

    def process_all(self, raw_data_dir: Union[str, Path]):
        """Complete processing pipeline."""
        site_dir = self.output_dir / "site_files"
        all_sample_ids = self.metadata["#SampleID"].unique()

        # Check existing files AND populate sample_file_map
        self.sample_file_map.clear()
        all_files_exist = True
        for sample_id in all_sample_ids:
            output_file = site_dir / f"{sample_id}.fastq.gz"
            if output_file.exists():
                self.sample_file_map[sample_id] = output_file
            else:
                all_files_exist = False

        if all_files_exist:
            logger.info("All output files already exist. Skipping processing.")
            return self.sample_file_map

        # If we get here, proceed with full processing
        organized_dir = self.organize_input_files(raw_data_dir)
        fastq_files = list(organized_dir.glob("*.fastq.gz"))

        with self.progress:
            main_desc = "Processing pooled files"
            main_task = self.progress.add_task(
                _format_task_desc(main_desc),
                total=len(fastq_files),
            )

            for fastq_file in fastq_files:
                self.process_single_file(fastq_file)
                self.progress.advance(main_task)

        self.write_site_files()
        shutil.rmtree(organized_dir)
        logger.info("Processing complete")
        return self.sample_file_map
