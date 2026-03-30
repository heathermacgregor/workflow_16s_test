from __future__ import print_function  # For Python 2/3 compatibility
# workflow_16s/api/ena/pooled_samples.py

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
class PooledSamplesProcessor:
    def __init__(
        self, 
        metadata_df: pd.DataFrame, 
        output_dir: Union[str, Path], 
        progress_obj: Any = None
    ):
        self.metadata = metadata_df
        self.output_dir = Path(output_dir)
        self.site_records = defaultdict(list)
        self.sample_file_map = {}
        self._create_lookup_dict()
        
        self.logger = get_logger("workflow_16s")
        
        self.progress = progress_obj if progress_obj else get_progress_bar()
        self._standalone = progress_obj is None

    def _create_lookup_dict(self):
        """Create internal lookup dictionary from metadata."""
        self.lookup_dict = {
            (
                row["run_accession"], 
                row["barcode_sequence"]
            ): row["#SampleID"] 
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
            self.logger.error(f"Corrupted file {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")

    def write_site_files(self) -> Dict[str, Path]:
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
            self.logger.info(f"Wrote {len(records)} records to {output_file}")

        return self.sample_file_map  # Return the mapping dictionary

    @staticmethod
    def merge_files(
        input_files: List[Union[str, Path]], 
        output_file: Union[str, Path]
    ) -> None:
        """Merge multiple FASTQ.gz files."""
        with gzip.open(output_file, "wb") as wfd:
            for f in input_files:
                with gzip.open(f, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)

    def organize_input_files(
        self, 
        raw_dir: Union[str, Path]
    ) -> Path:
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
                self.logger.info(f"Merging {len(paths)} copies of {file}")
                self.merge_files(paths, output_path)
            else:
                shutil.copy2(paths[0], output_path)

        return organized_dir

    def find_matching_files(self, search_dir: Union[str, Path]):
        """Find FASTQ files matching metadata run_accession entries."""
        search_path = Path(search_dir)
        paths = [
            p for p in search_path.glob("*.fastq.gz") 
            if "trimmed" not in str(p)
        ]

        file_map = {}
        for run_id in self.metadata["run_accession"].unique():
            matches = [p for p in paths if str(run_id) in str(p)]
            file_map[run_id] = matches if matches else []

        return file_map

    def process_all(
        self, 
        raw_data_dir: Union[str, Path]
    ) -> Dict[str, List[str]]:
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
            self.logger.info("All output files already exist. Skipping processing.")
            return self.sample_file_map

        # If we get here, proceed with full processing
        organized_dir = self.organize_input_files(raw_data_dir)
        fastq_files = list(organized_dir.glob("*.fastq.gz"))

        # Dashboard-safe manual lifecycle
        if self._standalone: self.progress.start()
        
        main_task = self.progress.add_task(
            "[blue]Demultiplexing Pooled Files", 
            total=len(fastq_files)
        )

        try:
            for fastq_file in fastq_files:
                self.process_single_file(fastq_file)
                self.progress.advance(main_task)
        finally:
            if self._standalone: 
                self.progress.stop()
            else: 
                self.progress.remove_task(main_task)

        self.write_site_files()
        shutil.rmtree(organized_dir)
        return self.sample_file_map