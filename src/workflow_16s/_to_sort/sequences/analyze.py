# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import gzip
import logging
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

# Local Imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import workflow_16s.custom_tmp_config
from workflow_16s import constants
from workflow_16s.ena.api import SequenceFetcher as SeqFetcher

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== CLASSES ====================================== #

class PrimerChecker:
    """Checks primer presence with precompiled regex patterns and efficient FASTQ 
    parsing."""
    
    def __init__(
        self,
        primer_regions: Dict[str, Tuple[str, str]],
        min_match: int = 10,
        max_reads: int = 1000,
        check_region: int = 100,
        num_workers: int = 2
    ):
        self.primer_regions = primer_regions
        self.min_match = min_match
        self.max_reads = max_reads
        self.check_region = check_region
        self.num_workers = num_workers
        self.iupac_table = self._create_iupac_table()
        self.forward_patterns, self.reverse_patterns = self._precompile_primer_patterns()

    def _precompile_primer_patterns(self) -> Tuple[Dict, Dict]:
        """Precompile regex patterns for all primers during initialization."""
        forward = {}
        reverse = {}
        for region, (fwd, rev) in self.primer_regions.items():
            rev_comp = str(Seq(rev).reverse_complement())
            forward[region] = self._create_primer_pattern(fwd)
            reverse[region] = self._create_primer_pattern(rev_comp)
        return forward, reverse

    def _analyze_multiple_runs(
        self, dataset: Dict[str, List[Union[str, Path]]]
    ) -> pd.DataFrame:
        """Concurrent analysis of multiple sequencing runs."""
        logger.info(
            f"Starting primer analysis for {len(dataset)} runs "
            f"with {self.num_workers} workers..."
        )
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._analyze_single_run, run_id, files): run_id
                for run_id, files in dataset.items()
            }
            
            for future in tqdm(
                as_completed(futures), 
                total=len(futures), 
                desc="Checking common primers".ljust(35), 
                colour='green'
            ):
                run_id = futures[future]
                try:
                    run_result = future.result()
                    results.append((run_id, run_result))
                    logger.info(f"Completed primer analysis for {run_id}")
                except Exception as e:
                    logger.error(f"Error processing {run_id}: {str(e)}")
        
        return pd.DataFrame.from_dict(dict(results), orient='index')

    def _analyze_single_run(
        self, run_id: str, files: List[Union[str, Path]]
    ) -> Dict[str, float]:
        """Analyze a single sequencing run with precompiled patterns."""
        logger.info(f"Processing run {run_id}")
        primer_results = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            fwd_futures = {
                region: executor.submit(
                    self._check_primer_frequency,
                    Path(files[0]),
                    pattern
                )
                for region, pattern in self.forward_patterns.items()
            }
            
            rev_futures = {
                region: executor.submit(
                    self._check_primer_frequency,
                    Path(files[1]) if len(files) > 1 else Path(files[0]),
                    pattern
                )
                for region, pattern in self.reverse_patterns.items()
            }
            
            for region in self.primer_regions:
                fwd_rate = fwd_futures[region].result()
                rev_rate = rev_futures[region].result()
                avg_rate = (fwd_rate + rev_rate) / 2
                primer_results[f"{region}_primer_rate"] = avg_rate
                logger.debug(
                    f"{region} - Fwd: {fwd_rate*100:.1f}%, Rev: {rev_rate*100:.1f}%"
                )
                
        return primer_results

    def _check_primer_frequency(self, file_path: Path, pattern: re.Pattern) -> float:
        """Calculate primer presence using BioPython parsing."""
        logger.info(f"Checking {pattern.pattern} in {file_path.name}")
        matches = 0
        count = 0
        
        try:
            with gzip.open(file_path, 'rt') as fh:
                for i, record in enumerate(SeqIO.parse(fh, 'fastq')):
                    if i >= self.max_reads:
                        break
                    seq = str(record.seq[:self.check_region]).upper()
                    if pattern.search(seq):
                        matches += 1
                    count = i + 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return 0.0
            
        match_rate = matches / count if count > 0 else 0.0
        logger.info(f"Matched {matches}/{count} reads ({match_rate*100:.1f}%)")
        return match_rate

    def _create_primer_pattern(self, primer: str) -> re.Pattern:
        """Create optimized regex pattern for primer matching."""
        pattern = []
        for c in primer.upper():
            pattern.append(f'[{self.iupac_table.get(c, "N")}]')
        return re.compile(''.join(pattern))

    def _create_iupac_table(self) -> Dict[str, str]:
        """IUPAC ambiguity code lookup table."""
        return {
            'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G',
            'R': 'AG', 'Y': 'CT', 'S': 'GC', 'W': 'AT',
            'K': 'GT', 'M': 'AC', 'B': 'CGT', 'D': 'AGT',
            'H': 'ACT', 'V': 'ACG', 'N': 'ACGT'
        }


class BLASTAnalyzer:
    """Performs BLAST analysis with in-memory processing and optimized parsing."""
    
    DEFAULT_REGIONS = constants.DEFAULT_REGIONS

    def __init__(
        self,
        regions: Dict[str, Tuple[int, int]] = DEFAULT_REGIONS,
        tmp_dir: Union[str, Path] = "/tmp",
        blast_db: Union[str, Path] = "blast_db",
        sample_size: int = 100,
        num_threads: int = 2
    ):
        self.regions = regions
        self.tmp_dir = Path(tmp_dir)
        self.blast_db = Path(blast_db)
        self.sample_size = sample_size
        self.num_threads = num_threads

    def _process_file(self, input_path: Union[str, Path]):
        """Optimized processing pipeline with in-memory operations."""
        logger.info(f"Processing {input_path.name}")
        sampled_path = self._sample_reads(Path(input_path))
        fasta_path = self._convert_to_fasta(sampled_path)
        blast_path = self._run_blast(fasta_path)
        df = self._parse_blast_results(sampled_path, blast_path)
        df = self._calculate_region_coverage(df)

        # Cleanup
        sampled_path.unlink(missing_ok=True)
        fasta_path.unlink(missing_ok=True)
        blast_path.unlink(missing_ok=True)
        
        return df

    def _sample_reads(self, input_path: Path) -> Path:
        """Subsample using BioPython without external tools."""
        output_path = self.tmp_dir / f"{input_path.stem}_sample.fastq.gz"
        with gzip.open(input_path, 'rt') as in_fh:
            records = list(SeqIO.parse(in_fh, 'fastq'))
        
        if len(records) > self.sample_size:
            records = random.sample(records, self.sample_size)
        
        with gzip.open(output_path, 'wt') as out_fh:
            SeqIO.write(records, out_fh, 'fastq')
        
        return output_path

    def _convert_to_fasta(self, input_path: Path) -> Path:
        """Convert to FASTA using BioPython."""
        fasta_path = self.tmp_dir / f"{input_path.stem}.fasta"
        with gzip.open(input_path, 'rt') as in_fh:
            records = SeqIO.parse(in_fh, 'fastq')
            SeqIO.write(records, fasta_path, 'fasta')
        return fasta_path

    def _run_blast(self, input_path: Path) -> Path:
        """Execute BLAST with optimized parameters."""
        output_path = self.tmp_dir / f"{input_path.stem}_blast.tsv"
        blast_cmd = [
            "blastn",
            "-db", str(self.blast_db),
            "-query", str(input_path),
            "-out", str(output_path),
            "-outfmt", "6 qseqid sseqid stitle pident qcovs qlen slen qstart qend sstart send qseq sseq",
            "-num_threads", str(self.num_threads)
        ]
        
        try:
            subprocess.run(blast_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"BLAST failed: {e.stderr.decode()}")
            raise
        
        return output_path

    def _parse_blast_results(self, fastq_path: Path, blast_path: Path) -> pd.DataFrame:
        """Optimized parsing with vectorized operations."""
        columns = [
            'Query ID', 'Subject ID', 'Subject Name', 'Identity', 'Coverage',
            'Query Length', 'Subject Length', 'Query Start', 'Query End',
            'Subject Start', 'Subject End', 'Query Sequence', 'Subject Sequence'
        ]
        
        try:
            df = pd.read_csv(blast_path, sep='\t', names=columns)
            if df.empty:
                return df
            
            # Numeric conversions
            numeric_cols = ['Query Start', 'Query End', 'Subject Start', 'Subject End']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Coverage calculations
            df['Coverage Start'] = df[['Subject Start', 'Subject End']].min(axis=1)
            df['Coverage End'] = df[['Subject Start', 'Subject End']].max(axis=1)
            
            # Vectorized midline creation
            df['Alignment Midline'] = [
                ''.join('|' if q == s else ' ' for q, s in zip(qseq, sseq))
                for qseq, sseq in zip(df['Query Sequence'], df['Subject Sequence'])
            ]
            
            # Sequence mapping
            with gzip.open(fastq_path, 'rt') as f:
                seq_records = {rec.id.split()[0]: str(rec.seq) for rec in SeqIO.parse(f, 'fastq')}
            
            df['Unaligned Sequence'] = df['Query ID'].str.split().str[0].map(seq_records)
            
            return df.dropna(subset=['Query Sequence', 'Subject Sequence'])
            
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            return pd.DataFrame()

    def _calculate_region_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized region coverage calculation."""
        if df.empty:
            return df
            
        for region, (start, end) in self.regions.items():
            df[f'{region} Coverage'] = (
                np.minimum(df['Coverage End'], end) - 
                np.maximum(df['Coverage Start'], start) + 1
            ).clip(lower=0)
            
        return df


class Analyzer:
    """Coordinates analysis components with optimized parallel execution."""
    
    DEFAULT_PRIMER_REGIONS = constants.DEFAULT_PRIMER_REGIONS

    def __init__(
        self,
        blast_regions: Dict[str, Tuple[int, int]] = BLASTAnalyzer.DEFAULT_REGIONS,
        primer_regions: Dict[str, Tuple[str, str]] = DEFAULT_PRIMER_REGIONS,
        tmp_dir: Union[str, Path] = "/tmp",
        blast_db: Union[str, Path] = "blast_db",
        sample_size: int = 100,
        num_threads: int = 2
    ):
        self.blast_analyzer = BLASTAnalyzer(
            regions=blast_regions,
            tmp_dir=tmp_dir,
            blast_db=blast_db,
            sample_size=sample_size,
            num_threads=num_threads
        )
        self.primer_checker = PrimerChecker(
            primer_regions=primer_regions,
            num_workers=num_threads
        )

    def _process_multiple_runs(self, run_file_paths: Dict, output_path: Path) -> Dict[str, str]:
        """Optimized parallel processing of multiple runs."""
        logger.info(f"Analyzing {len(run_file_paths)} runs")
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.primer_checker.num_workers) as executor:
            # Primer analysis
            primer_future = executor.submit(
                self.primer_checker._analyze_multiple_runs,
                run_file_paths
            )
            
            # BLAST analysis
            blast_futures = {
                executor.submit(self._process_single_run, files): run_id
                for run_id, files in run_file_paths.items()
            }
            
            # Collect results
            primer_results = primer_future.result()
            for future in tqdm(as_completed(blast_futures), total=len(blast_futures)):
                run_id = blast_futures[future]
                try:
                    blast_df = future.result()
                    coverage = self._calculate_coverage_stats(blast_df)
                    results[run_id] = self._combine_results(
                        coverage,
                        primer_results.loc[run_id].to_dict()
                    )
                except Exception as e:
                    logger.error(f"Failed {run_id}: {e}")
                    results[run_id] = None

        self._save_results(results, output_path)
        return results

    def _process_single_run(self, files: List[Path]) -> pd.DataFrame:
        """Process single run with error handling."""
        try:
            return self.blast_analyzer._process_file(files[0])
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return pd.DataFrame()

    def _calculate_coverage_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate coverage statistics from BLAST results."""
        if df.empty:
            return {}
        return {
            region: df[f'{region} Coverage'].mean()
            for region in self.blast_analyzer.regions
        }

    def _combine_results(self, coverage: Dict, primers: Dict) -> str:
        """Combine primer and coverage results for final decision."""
        combined = []
        for region in self.blast_analyzer.regions:
            cov = coverage.get(region, 0)
            primer_rate = primers.get(f"{region}_primer_rate", 0)
            score = cov * primer_rate + cov/20
            combined.append((region, score))
        
        return max(combined, key=lambda x: x[1], default=("Unknown", 0))[0]

    def _save_results(self, results: Dict, output_path: Path):
        """Save results to output file with structured logging."""
        with open(output_path, 'w') as f:
            for run_id, region in results.items():
                f.write(f"{run_id}\t{region or 'N/A'}\n")


class Validate16S:
    """Validates 16S sequences with optimized metrics calculation."""
    
    def __init__(
        self,
        tmp_dir: Union[str, Path] = "/tmp",
        blast_db: Union[str, Path] = "blast_db",
        sample_size: int = 100,
        num_threads: int = 2,
        min_identity: float = 97.0,
        min_coverage: float = 90.0,
        min_valid_rate: float = 0.9
    ):
        self.analyzer = BLASTAnalyzer(
            tmp_dir=tmp_dir,
            blast_db=blast_db,
            sample_size=sample_size,
            num_threads=num_threads
        )
        self.min_identity = min_identity
        self.min_coverage = min_coverage
        self.min_valid_rate = min_valid_rate

    def validate(self, input_path: Path) -> Tuple[bool, dict]:
        """Validate a single sequencing file."""
        try:
            df = self.analyzer._process_file(input_path)
            metrics = self._calculate_metrics(df)
            return self._is_valid(metrics), metrics
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, {"error": str(e)}

    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate validation metrics from BLAST results."""
        if df.empty:
            return {}
            
        valid_mask = (
            (df['Identity'] >= self.min_identity) &
            (df['Coverage'] >= self.min_coverage)
        )
        
        return {
            'total_reads': len(df),
            'valid_reads': valid_mask.sum(),
            'valid_rate': valid_mask.mean(),
            'mean_identity': df['Identity'].mean(),
            'mean_coverage': df['Coverage'].mean()
        }

    def _is_valid(self, metrics: dict) -> bool:
        """Determine validation status based on metrics."""
        return metrics.get('valid_rate', 0) >= self.min_valid_rate


# ==================================== FUNCTIONS ===================================== #

def estimate_16s_subfragment(
    metadata: pd.DataFrame,
    runs: List[str],
    output_dir: Union[str, Path] = Path("results"),
    num_workers: int = 4
) -> Dict[str, str]:
    """Main analysis workflow with optimized resource management."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fetcher = SeqFetcher()
    run_files = fetcher.fetch_runs(runs)
    
    validator = Validate16S(num_threads=num_workers)
    analyzer = Analyzer(num_threads=num_workers)
    
    # Validation phase
    valid_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(validator.validate, files[0]): run_id
            for run_id, files in run_files.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            run_id = futures[future]
            try:
                is_valid, metrics = future.result()
                valid_results[run_id] = is_valid
            except Exception as e:
                logger.error(f"Validation failed for {run_id}: {e}")
                valid_results[run_id] = False
    
    # Filter invalid runs
    valid_runs = {
        run: files 
        for run, files in run_files.items() if valid_results.get(run, False)
    }
    # Region analysis
    region_results = analyzer._process_multiple_runs(
        valid_runs,
        output_dir / "region_results.tsv"
    )
    return region_results


def validate_16s(
    metadata: pd.DataFrame,
    runs: List[str],
    output_dir: Union[str, Path] = Path("results"),
    num_workers: int = 4
) -> Dict[str, str]:
    """Main analysis workflow with optimized resource management."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fetcher = SeqFetcher()
    run_files = fetcher.fetch_runs(runs)
    
    validator = Validate16S(num_threads=num_workers)
    
    # Validation phase
    valid_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(validator.validate, files[0]): run_id
            for run_id, files in run_files.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            run_id = futures[future]
            try:
                is_valid, metrics = future.result()
                valid_results[run_id] = is_valid
            except Exception as e:
                logger.error(f"Validation failed for {run_id}: {e}")
                valid_results[run_id] = False
    # Filter invalid runs
    valid_runs = {
        run: files 
        for run, files in run_files.items() if valid_results.get(run, False)
    }
    return valid_runs
    

def clean_temp_files(temp_files: Dict[str, List[Path]]):
    """Cleanup temporary files with error handling."""
    for files in temp_files.values():
        for f in files:
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")
