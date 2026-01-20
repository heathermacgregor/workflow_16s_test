# workflow_16s/upstream/sequences/analysis.py
# ==================================================================================== #
#                                       IMPORTS
# ==================================================================================== #

# Standard Library Imports
import argparse
import asyncio
import gzip
import io
import shlex
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import matplotlib.pyplot as plt
import pandas as pd
import regex
from Bio import SeqIO

# Local Imports
from workflow_16s.utils.progress import get_progress_bar

# Dependency Check for YAML
try:
    import yaml
except ImportError:
    print("PyYAML library not found. Please install it with 'pip install pyyaml'")
    exit(1)


# ==================================================================================== #
#                                     CONFIGURATION
# ==================================================================================== #

PRIMER_PRESENCE_THRESHOLD = 0.50
VSEARCH_COVERAGE_THRESHOLD = 75.0
PROCESSING_BATCH_SIZE = 150
YAML_OUTPUT_DIR_NAME = "estimations"
TSV_OUTPUT_FILENAME = "region_results.tsv"
PRIMER_DB_NAME = "primer_data.db"
REQUIRED_TOOLS = ["vsearch", "seqtk", "gzip"]

COMPREHENSIVE_V_REGIONS = {
    "V1": {"fwd_pos": 69, "rev_pos": 99, "leeway": 40},
    "V2": {"fwd_pos": 137, "rev_pos": 242, "leeway": 40},
    "V3": {"fwd_pos": 433, "rev_pos": 497, "leeway": 40},
    "V4": {"fwd_pos": 576, "rev_pos": 682, "leeway": 50},
    "V5": {"fwd_pos": 822, "rev_pos": 879, "leeway": 40},
    "V6": {"fwd_pos": 986, "rev_pos": 1043, "leeway": 40},
    "V7": {"fwd_pos": 1117, "rev_pos": 1173, "leeway": 40},
    "V8": {"fwd_pos": 1243, "rev_pos": 1294, "leeway": 40},
    "V9": {"fwd_pos": 1435, "rev_pos": 1465, "leeway": 40},
    "V1-V2": {"fwd_pos": 27, "rev_pos": 338, "leeway": 40},
    "V1-V3": {"fwd_pos": 27, "rev_pos": 534, "leeway": 50},
    "V2-V3": {"fwd_pos": 338, "rev_pos": 534, "leeway": 50},
    "V3-V4": {"fwd_pos": 341, "rev_pos": 805, "leeway": 50},
    "V4-V5": {"fwd_pos": 515, "rev_pos": 926, "leeway": 50},
    "V5-V7": {"fwd_pos": 785, "rev_pos": 1100, "leeway": 60},
    "V6-V8": {"fwd_pos": 926, "rev_pos": 1392, "leeway": 75},
    "V7-V9": {"fwd_pos": 1100, "rev_pos": 1492, "leeway": 100},
    "Full-Length": {"fwd_pos": 27, "rev_pos": 1492, "leeway": 100},
}


# ==================================================================================== #
#                                     LOGGER SETUP
# ==================================================================================== #

from workflow_16s.utils.logger import get_logger
logger = get_logger()

# ==================================================================================== #
#                             PRIMER DISCOVERY & DATABASE
# ==================================================================================== #

class PrimerFinder:
    """
    Finds primer pairs for specified 16S rRNA genomic regions from a probe database.

    Interfaces with a SQLite database to find suitable forward and reverse primer pairs 
    for amplifying predefined genomic regions. It identifies all potential primer 
    combinations for each target region based on positional data stored in the database.

    Attributes:
        db_path (Path): The file path to the SQLite primer database.
    Methods:
        _query_primers: Queries the database for primers near a target position.
        get_primer_pairs_for_regions: Identifies all potential primer pairs for predefined genomic regions.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        if not self.db_path.exists(): 
            raise FileNotFoundError(f"Primer database not found at {self.db_path}. Please run probebase.py first.")
        
    def _query_primers(self, target_position: int, leeway: int, direction: str) -> List[Dict[str, Any]]:
        query_range_start = max(0, target_position - leeway) 
        query_range_end = target_position + leeway
        query = "SELECT Primer_Name, Sequence, Position_Start, Position_End FROM primers WHERE Position_Start <= ? AND Position_End >= ? AND TRIM(Direction) = ? ORDER BY Position_Start;"
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (query_range_end, query_range_start, direction.strip()))
                col_names = [description[0] for description in cursor.description]
                return [dict(zip(col_names, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"SQLite query failed: {e}"); return []
            
    def get_primer_pairs_for_regions(self) -> Dict[str, List[Dict[str, Dict]]]:
        logger.debug(f"Discovering all potential primer pairs from database: {self.db_path}")
        region_to_pairs = defaultdict(list)
        for region, params in COMPREHENSIVE_V_REGIONS.items():
            fwd_primers = self._query_primers(params["fwd_pos"], params["leeway"], 'Forward primer')
            rev_primers = self._query_primers(params["rev_pos"], params["leeway"], 'Reverse primer')
            if not fwd_primers or not rev_primers:
                logger.warning(f"Could not find sufficient primers for region '{region}'. Skipping."); continue
            for fwd in fwd_primers:
                for rev in rev_primers:
                    if fwd['Position_Start'] < rev['Position_Start']:
                        region_to_pairs[region].append({
                            "fwd": {
                                "name": fwd['Primer_Name'], "seq": fwd['Sequence'], 
                                "position": (fwd['Position_Start'], fwd['Position_End'])
                            },
                            "rev": {
                                "name": rev['Primer_Name'], "seq": rev['Sequence'], 
                                "position": (rev['Position_Start'], rev['Position_End'])
                            }
                        })
            logger.debug(f" - Found {len(fwd_primers)} forward and {len(rev_primers)} reverse primers, creating {len(region_to_pairs[region])} pairs for {region}.")
        return dict(region_to_pairs)

# ==================================================================================== #
#                                   HELPER FUNCTIONS
# ==================================================================================== #

def check_dependencies(tools: List[str]):
    for tool in tools:
        if not shutil.which(tool): 
            raise FileNotFoundError(f"Dependency Error: '{tool}' not found in your PATH.")


async def run_tasks_with_progress(tasks: List[asyncio.Task], description: str) -> List:
    progress = get_progress_bar()
    results = []
    with progress:
        task_id = progress.add_task(description, total=len(tasks))
        for future in asyncio.as_completed(tasks):
            results.append(await future)
            progress.update(task_id, advance=1)
    return results


def create_alignment_plot(results_df: pd.DataFrame, output_path: Path):
    """Generates and saves a plot visualizing the predicted alignment regions for each run."""
    predictions = results_df[results_df['is_prediction']].copy()
    if predictions.empty:
        logger.warning("No successful predictions to plot. Skipping alignment plot generation.")
        return

    # Sort runs by their start position for a cleaner plot
    predictions.sort_values('vsearch_avg_start', inplace=True)
    run_order = predictions['run_accession'].tolist()

    # Create a dynamic figure size based on the number of runs
    fig_height = max(6, len(run_order) * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # 1. Plot the full 16S gene backbone
    ax.hlines(
        y=run_order, xmin=1, xmax=1542, color='lightgrey', alpha=0.7, linewidth=5, 
        label='16S Gene Backbone'
    )

    # 2. Plot the predicted alignment for each run
    for _, row in predictions.iterrows():
        run_id = row['run_accession']
        start, end = row['vsearch_avg_start'], row['vsearch_avg_end']
        ax.hlines(
            y=run_id, xmin=start, xmax=end, color='coral', linewidth=5, 
            label=f'Predicted Fragment ({row["region"]})'
        )
        ax.text(
            end + 10, run_id, row['region'], va='center', ha='left', 
            fontsize=9, color='black'
        )

    # 3. Overlay the canonical V-regions for reference
    for region, params in COMPREHENSIVE_V_REGIONS.items():
        ax.axvspan(
            params['fwd_pos'], params['rev_pos'], alpha=0.15, 
            color='skyblue', zorder=0
        )
        ax.text(
            (params['fwd_pos'] + params['rev_pos']) / 2, 
            ax.get_ylim()[1], region, ha='center', va='bottom', 
            fontsize=8, color='blue', alpha=0.8
        )

    # Formatting
    ax.set_xlabel("Position on 16S rRNA Gene (bp)", fontsize=12)
    ax.set_ylabel("Run Accession", fontsize=12)
    ax.set_title(
        "Predicted 16S Subfragment Alignments", fontsize=14, weight='bold'
    )
    ax.set_xlim(0, 1600)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Clean up legend (remove duplicate labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')

    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Alignment visualization saved to: {output_path}")
    except Exception as e: logger.error(f"Failed to save alignment plot: {e}")
    finally: plt.close(fig)
        
# ==================================================================================== #
#                              DATA FETCHING & VALIDATION
# ==================================================================================== #

class Validate16S:
    def __init__(self, min_len: int = 100, max_reads: int = 1000):
        self.min_len, self.max_reads = min_len, max_reads

    async def validate_run(self, run_id: str, file_path: Union[Path, List[Path]]) -> Tuple[str, bool, Optional[str]]:
        path_to_check = file_path[0] if isinstance(file_path, list) else file_path
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._validate_file, run_id, path_to_check)

    def _validate_file(self, run_id: str, file_path: Path) -> Tuple[str, bool, Optional[str]]:
        read_count = 0
        try:
            with gzip.open(file_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    if read_count >= self.max_reads: break
                    if len(record.seq) < self.min_len: 
                        return run_id, False, f"short reads (<{self.min_len}bp)"
                    read_count += 1
            if read_count == 0: return run_id, False, "file contains no reads"
        except Exception as e:
            return run_id, False, f"file processing error: {e}"
        return run_id, True, None

# ==================================================================================== #
#                                   ANALYSIS CLASSES
# ==================================================================================== #
class PrimerChecker:
    def __init__(
        self, all_primer_pairs: Dict[str, list], max_reads: int = 1000, 
        check_region: int = 100, mismatches: int = 2
    ):
        self.all_primer_pairs, self.max_reads, self.check_region, self.mismatches = all_primer_pairs, max_reads, check_region, mismatches
        self.iupac_table = self._create_iupac_table()
        self.fwd_patterns, self.rev_patterns = self._precompile_all_primer_patterns()
    def _create_iupac_table(self) -> Dict[str, str]:
        return {
            'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'T', 'R': '[AG]', 'Y': '[CT]', 
            'S': '[GC]', 'W': '[AT]', 'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]', 
            'H': '[ACT]', 'V': '[ACG]', 'N': '[ATGC]'
        }
        
    def _reverse_complement(self, seq: str) -> str:
        complement = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N', 'R': 'Y', 'Y': 'R', 'S': 'S', 
            'W': 'W', 'K': 'M', 'M': 'K', 'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D'
        }
        return "".join(complement.get(base, base) for base in reversed(seq))
    
    def _precompile_all_primer_patterns(self) -> Tuple[Dict[str, regex.Pattern], Dict[str, regex.Pattern]]:
        fwd_patterns, rev_patterns, unique_primers = {}, {}, {}
        for region, pairs in self.all_primer_pairs.items():
            for pair in pairs:
                unique_primers[(pair['fwd']['seq'], 'fwd')] = pair['fwd']['name']
                unique_primers[(pair['rev']['seq'], 'rev')] = pair['rev']['name']
        for (seq, direction), name in unique_primers.items():
            seq_iupac = "".join(self.iupac_table.get(base, base) for base in seq) # type: ignore
            if direction == 'fwd':
                if name not in fwd_patterns: fwd_patterns[name] = regex.compile(f"({seq_iupac}){{e<={self.mismatches}}}")
            elif name not in rev_patterns:
                rev_comp_iupac = "".join(self.iupac_table.get(base, base) for base in self._reverse_complement(seq))
                rev_patterns[name] = regex.compile(f"({rev_comp_iupac}){{e<={self.mismatches}}}")
        logger.debug(f"Precompiled fuzzy regex for {len(fwd_patterns)} unique forward and {len(rev_patterns)} unique reverse primers.")
        return fwd_patterns, rev_patterns

    async def analyze_run(self, run_id: str, file_path: Union[Path, List[Path]]) -> Tuple[str, Dict[str, float]]:
        path_to_check = file_path[0] if isinstance(file_path, list) else file_path
        return await asyncio.get_running_loop().run_in_executor(None, self._analyze_file, run_id, path_to_check)

    def _analyze_file(self, run_id: str, file_path: Path) -> Tuple[str, Dict[str, float]]:
        primer_counts, read_count = defaultdict(int), 0
        try:
            with gzip.open(file_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    if read_count >= self.max_reads: break
                    read_count += 1
                    seq_start = str(record.seq[:self.check_region])
                    seq_end = str(record.seq[-self.check_region:])
                    for name, pattern in self.fwd_patterns.items():
                        if pattern.search(seq_start): primer_counts[name] += 1
                    for name, pattern in self.rev_patterns.items():
                        if pattern.search(seq_end): primer_counts[name] += 1
            return run_id, {name: count / read_count 
                            for name, count in primer_counts.items()} if read_count > 0 else {}
        except Exception as e:
            logger.error(f"Error processing {file_path} for comprehensive primer check: {e}"); return run_id, {}


class VsearchAnalyzer:
    def __init__(self, db_path: Path, semaphore: asyncio.Semaphore, threads: int = 16):
        if not db_path.exists(): raise FileNotFoundError(f"VSEARCH database not found at: {db_path}")
        self.db_path, self.semaphore, self.threads = db_path, semaphore, threads
        
    async def analyze_taxonomy(
        self, run_id: str, file_path: Path, pident: float = 0.97, maxaccepts: int = 1, maxrejects: int = 16, 
        sample_size: int = 0
    ) -> Tuple[str, pd.DataFrame, Optional[str]]:
        """Runs VSEARCH alignment and returns the results as a DataFrame and an optional failure reason."""
        async with self.semaphore:
            input_file, db_file = shlex.quote(str(file_path)), shlex.quote(str(self.db_path))
            log_dir = file_path.parent / "vsearch_logs"; log_dir.mkdir(exist_ok=True)
            log_file, temp_fasta_path = log_dir / f"{run_id}_vsearch_taxonomy.log", log_dir / f"{run_id}_derep.fasta"
            
            subsample_cmd = f"seqtk sample -s100 - {sample_size} | " if sample_size > 0 else ""
            derep_cmd = (
                f"gzip -dc {input_file} | {subsample_cmd}"
                f"vsearch --fastx_filter - --fastq_maxee 1.0 --fastq_minlen 150 --fastaout - | "
                f"vsearch --derep_fulllength - --output {shlex.quote(str(temp_fasta_path))} --sizeout"
            )
            
            proc_derep = await asyncio.create_subprocess_shell(
                derep_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr_derep = await proc_derep.communicate()
            
            if proc_derep.returncode != 0:
                return run_id, pd.DataFrame(), f"VSEARCH dereplication step failed: {stderr_derep.decode().strip()}"
            if not temp_fasta_path.exists() or temp_fasta_path.stat().st_size == 0:
                return run_id, pd.DataFrame(), "Dereplication produced no unique sequences"

            len_map = {rec.id: len(rec.seq) for rec in SeqIO.parse(temp_fasta_path, "fasta")}
            userfields = "query+target+id+alnlen+mism+gaps+qilo+qihi+tilo+tihi"
            col_names = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send']
            align_cmd = (
                f"vsearch --usearch_global {shlex.quote(str(temp_fasta_path))} --db {db_file} --id {pident} "
                f"--threads {self.threads} --maxaccepts {maxaccepts} --maxrejects {maxrejects} "
                f"--log {shlex.quote(str(log_file))} "
                f"--userout - --userfields {userfields}"
            )

            proc_align = await asyncio.create_subprocess_shell(
                align_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout_align, stderr_align = await proc_align.communicate()

            if proc_align.returncode != 0:
                return run_id, pd.DataFrame(), f"VSEARCH alignment step failed: {stderr_align.decode().strip()}"
            
            output = stdout_align.decode()
            if not output.strip():
                return run_id, pd.DataFrame(), "VSEARCH alignment produced no hits"
            
            try:
                results_df = pd.read_csv(io.StringIO(output), sep='\t', names=col_names)
                results_df['pident'] /= 100.0
                results_df['size'] = results_df['qseqid'].str.extract(r';size=(\d+);?', expand=False).astype(int)
                results_df['qlen'] = results_df['qseqid'].map(len_map)
                temp_fasta_path.unlink(missing_ok=True)
                return run_id, results_df, None
            except Exception as e:
                return run_id, pd.DataFrame(), f"Failed to parse VSEARCH results: {e}"


# ==================================================================================== #
#                                  RESULTS PROCESSING
# ==================================================================================== #

def score_regions_from_primer_report(
    primer_frequencies: Dict[str, float], region_to_pairs_map: Dict[str, list]
) -> Dict[str, Dict]:
    """
    Scores each V-region based on the highest-scoring primer pair found.
    Returns both the score and the best pair.
    """
    region_scores = {}
    for region, pairs in region_to_pairs_map.items():
        max_pair_score = 0.0
        best_pair_for_region = None
        if not pairs:
            region_scores[region] = {'score': 0.0, 'best_pair': None}
            continue
        
        for pair in pairs:
            fwd_freq = primer_frequencies.get(pair['fwd']['name'], 0.0)
            rev_freq = primer_frequencies.get(pair['rev']['name'], 0.0)
            pair_score = (fwd_freq + rev_freq) / 2.0
            if pair_score > max_pair_score:
                max_pair_score = pair_score
                best_pair_for_region = pair
        
        region_scores[region] = {'score': max_pair_score, 'best_pair': best_pair_for_region}
    return region_scores


def estimate_16s_subfragment(
    df: pd.DataFrame, region_map: Dict[str, Dict], vsearch_threshold: float
) -> Tuple[Dict[str, Dict], str]:
    """
    Estimates 16S subfragment coverage from a VSEARCH results DataFrame.
    Returns the results dictionary and a summary string that acknowledges multiple 
    plausible matches.
    """
    if df.empty or 'size' not in df.columns or 'qlen' not in df.columns:
        results = {
            region: {'coverage': 0.0, 'avg_len': 0.0, 'avg_start': 0.0, 'avg_end': 0.0} 
            for region in region_map
        }
        return results, "No alignments received from VSEARCH."

    df['aln_span'] = df['send'] - df['sstart']
    plausible_alignments = df[df['aln_span'] < (df['qlen'] * 1.5)]

    if plausible_alignments.empty:
        results = {
            region: {'coverage': 0.0, 'avg_len': 0.0, 'avg_start': 0.0, 'avg_end': 0.0}
            for region in region_map
        }
        return results, "No plausible alignments found after filtering."

    total_reads = df['size'].sum()
    if total_reads == 0:
        results = {
            region: {'coverage': 0.0, 'avg_len': 0.0, 'avg_start': 0.0, 'avg_end': 0.0} 
            for region in region_map
        }
        return results, "Total read count was zero after filtering."

    results = {}
    for region, params in region_map.items():
        fwd_start, rev_end = params.get('fwd_pos'), params.get('rev_pos')
        if fwd_start is None or rev_end is None: continue
        
        region_df = plausible_alignments[(plausible_alignments['sstart'] <= rev_end) & (plausible_alignments['send'] >= fwd_start)]
        if not region_df.empty:
            covered_reads = region_df['size'].sum()
            if covered_reads > 0:
                results[region] = {
                    'coverage': (covered_reads / total_reads) * 100,
                    'avg_len': (region_df['qlen'] * region_df['size']).sum() / covered_reads,
                    'avg_start': (region_df['sstart'] * region_df['size']).sum() / covered_reads,
                    'avg_end': (region_df['send'] * region_df['size']).sum() / covered_reads
                }
            else:
                results[region] = {
                    'coverage': 0.0, 'avg_len': 0.0,
                    'avg_start': 0.0, 'avg_end': 0.0
                }
        else:
            results[region] = {
                'coverage': 0.0, 'avg_len': 0.0, 
                'avg_start': 0.0, 'avg_end': 0.0
            }

    # Identify and summarize plausible matches based on the threshold
    plausible_matches = sorted(
        [(r, v['coverage']) for r, v in results.items() if v['coverage'] >= vsearch_threshold],
        key=lambda item: item[1],
        reverse=True
    )

    summary_msg = ""
    if not plausible_matches:
        # Check if there was any signal at all, even below the threshold
        best_region, best_result = max(
            results.items(), 
            key=lambda item: item[1]['coverage']
        ) if results else (None, {'coverage': 0.0})
        if best_result['coverage'] > 0:
            summary_msg = f"Weak signal for region '{best_region}' ({best_result['coverage']:.1f}% coverage), below threshold."
        else:
            summary_msg = "No alignments covered any defined V-regions."
    elif len(plausible_matches) == 1:
        region, coverage = plausible_matches[0]
        details = results[region]
        summary_msg = f"Strong match for region '{region}' ({coverage:.1f}% coverage; avg aln: {details['avg_start']:.0f}-{details['avg_end']:.0f} bp)."
    else:
        # Acknowledge multiple matches
        match_summary = ", ".join([f"{r} ({c:.1f}%)" for r, c in plausible_matches[:2]])
        best_match = plausible_matches[0][0]
        summary_msg = f"Multiple plausible matches found: {match_summary}. Best coverage for '{best_match}'."

    return results, summary_msg

def predict_best_region(run_results: Dict, all_regions: List[str], 
                        primer_threshold: float, vsearch_threshold: float) -> Dict:
    all_evidence, has_any_primer_signal = [], False
    for region in all_regions:
        primer_score = run_results.get(f'primer_{region}', 0.0)
        vsearch_info = run_results.get(f'vsearch_{region}', {'coverage': 0.0})
        vsearch_score = vsearch_info.get('coverage', 0.0)
        if primer_score > 0.01: has_any_primer_signal = True
        passes_thresholds = (primer_score >= primer_threshold and vsearch_score >= vsearch_threshold)
        region_span = COMPREHENSIVE_V_REGIONS[region]['rev_pos'] - COMPREHENSIVE_V_REGIONS[region]['fwd_pos']
        all_evidence.append({
            'region': region, 'primer_score': float(round(primer_score, 4)), 
            'vsearch_score': float(round(vsearch_score, 2)), 
            'passed_thresholds': bool(passes_thresholds), 'span': region_span
        })
    final_prediction, reasoning = "Undetermined", "Prediction failed: No region met the analysis criteria."
    primary_candidates = [e for e in all_evidence if e['passed_thresholds']]
    if primary_candidates:
        # Sort by VSEARCH score (desc), then by span (asc) for specificity
        best_candidate = sorted(primary_candidates, key=lambda x: (x['vsearch_score'], -x['span']), reverse=True)[0]
        final_prediction = best_candidate['region']
        reasoning = f"High-confidence match: Region '{final_prediction}' passed both primer ({best_candidate['primer_score']:.2%}) and alignment ({best_candidate['vsearch_score']:.1f}%) thresholds."
    elif has_any_primer_signal:
        high_primer_candidates = [e for e in all_evidence if e['primer_score'] >= primer_threshold]
        if high_primer_candidates:
            # Sort by primer score (desc), then by span (asc) for specificity
            best_candidate = sorted(high_primer_candidates, key=lambda x: (x['primer_score'], -x['span']), reverse=True)[0]
            final_prediction = best_candidate['region']
            reasoning = f"Primer-driven match: Region '{final_prediction}' had a strong primer signal ({best_candidate['primer_score']:.2%}) but did not meet the VSEARCH coverage threshold ({best_candidate['vsearch_score']:.1f}%)."
        else: reasoning = "Prediction failed: Low primer signals detected, but no region met thresholds."
    else:
        high_vsearch_candidates = [e for e in all_evidence if e['vsearch_score'] >= vsearch_threshold]
        if high_vsearch_candidates:
            # Sort by VSEARCH score (desc), then by span (asc) for specificity
            best_candidate = sorted(high_vsearch_candidates, key=lambda x: (x['vsearch_score'], -x['span']), reverse=True)[0]
            final_prediction = best_candidate['region']
            reasoning = f"Alignment-driven match: No primer signal detected. Region '{final_prediction}' was chosen as the most specific high-coverage match (coverage: {best_candidate['vsearch_score']:.1f}%, span: {best_candidate['span']} bp)."
    sorted_evidence = sorted(all_evidence, key=lambda x: (x['primer_score'], x['vsearch_score']), reverse=True)
    return {
        'prediction': final_prediction, 'reasoning': reasoning, 
        'evidence': {
            'parameters': {
                'primer_threshold': primer_threshold, 'vsearch_threshold': vsearch_threshold}, 
                'checked_regions': sorted_evidence
        }
    }


def combine_results(vsearch_results: Dict, primer_scores: Dict, all_regions: List[str]) -> Dict:
    """Combines VSEARCH results and primer scores into a flat dictionary."""
    combined = {}
    for region in all_regions:
        combined[f'primer_{region}'] = primer_scores.get(region, {}).get('score', 0.0)
        combined[f'vsearch_{region}'] = vsearch_results.get(region, {'coverage': 0.0})
    return combined


async def get_valid_runs(
    run_file_paths: Dict[str, Union[Path, List[Path]]], validator: Validate16S
) -> Tuple[Dict[str, Union[Path, List[Path]]], Dict[str, bool]]:
    tasks = [
        asyncio.create_task(validator.validate_run(run_id, f)) 
        for run_id, f in run_file_paths.items()
    ]
    valid_runs, status, failed_runs = {}, {}, defaultdict(list)
    validation_results = await run_tasks_with_progress(tasks, "Validating runs")
    for run_id, is_valid, reason in validation_results:
        status[run_id] = is_valid
        if is_valid:
            valid_runs[run_id] = run_file_paths[run_id]
        else:
            failed_runs[reason].append(run_id)
    if failed_runs:
        logger.warning("Some runs failed validation:")
        for reason, runs in failed_runs.items():
            logger.warning(f" - {len(runs)} failed: '{reason}'.")
    return valid_runs, status

# ==================================================================================== #
#                                     MAIN WORKFLOW
# ==================================================================================== #

async def merge_paired_reads(run_id: str, file_paths: List[Path]) -> Optional[Path]:
    if len(file_paths) != 2:
        logger.warning(f"[{run_id}] Expected 2 files for merging, but found {len(file_paths)}. Skipping merge.")
        return None
    r1_path, r2_path = file_paths[0], file_paths[1]
    output_dir = r1_path.parent.parent / "merged"
    output_dir.mkdir(exist_ok=True)
    merged_path = output_dir / f"{run_id}.merged.fastq.gz"
    report_path = output_dir / f"{run_id}.merge.log"
    cmd = (
        f"vsearch --fastq_mergepairs {shlex.quote(str(r1_path))} "
        f"--reverse {shlex.quote(str(r2_path))} "
        f"--fastqout - | gzip > {shlex.quote(str(merged_path))}"
    )
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    with open(report_path, 'w') as f: f.write(stderr.decode())
    if proc.returncode != 0:
        logger.error(f"[{run_id}] Failed to merge reads. See log: {report_path}"); return None
    return merged_path


async def run_comprehensive_analysis(
    run_file_paths: Dict[str, Union[Path, List[Path]]],
    output_dir: Path, 
    vsearch_db: Path,
    primer_db_path: Path, 
    region_to_pairs_map: Dict[str, list],
    threads: int = 4, 
    max_concurrency: int = 10
) -> Dict[str, Dict]:
    check_dependencies(REQUIRED_TOOLS)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_output_dir = output_dir / YAML_OUTPUT_DIR_NAME; yaml_output_dir.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency)
    all_regions = list(region_to_pairs_map.keys())
    
    primer_checker = PrimerChecker(region_to_pairs_map, max_reads=100)
    vsearch_analyzer = VsearchAnalyzer(vsearch_db, semaphore, threads)
    validator = Validate16S()
    
    all_run_reports, tidy_results = {}, []
    
    logger.info(f"\n--- Processing Batch 1/1 ({len(run_file_paths)} runs) ---")
    
    analysis_paths, merge_tasks, single_end_runs = {}, [], {}
    for run_id, files in run_file_paths.items():
        if len(files) == 2: # type: ignore
            merge_tasks.append(asyncio.create_task(merge_paired_reads(run_id, files))) # type: ignore
        else:
            single_end_runs[run_id] = files[0] # type: ignore

    if merge_tasks:
        merged_results = await run_tasks_with_progress(merge_tasks, "Merging paired-end reads")
        paired_run_ids = [run_id for run_id, files in run_file_paths.items() if len(files) == 2] # type: ignore
        for run_id, merged_path in zip(paired_run_ids, merged_results):
            if merged_path: analysis_paths[run_id] = merged_path
            else: logger.warning(f"Excluding run {run_id} from analysis due to merge failure.")
    analysis_paths.update(single_end_runs)

    if not analysis_paths:
        logger.error("No runs available for analysis after merging/validation setup."); return {}
    
    valid_runs, _ = await get_valid_runs(analysis_paths, validator)
    if not valid_runs: logger.warning("No runs passed validation."); return {}
    
    primer_tasks = [
        asyncio.create_task(primer_checker.analyze_run(run_id, path)) 
        for run_id, path in valid_runs.items()
    ]
    vsearch_tasks = [
        asyncio.create_task(vsearch_analyzer.analyze_taxonomy(run_id, path, sample_size=100)) # type: ignore
        for run_id, path in valid_runs.items()
    ] 
    
    all_primer_results, all_vsearch_results = await run_tasks_with_progress(primer_tasks, "Analyzing primers"), await run_tasks_with_progress(vsearch_tasks, "Analyzing coverage")
    
    primer_freq_map = dict(all_primer_results)
    vsearch_df_map = {}
    for run_id_res, df, reason in all_vsearch_results:
        if reason:
            logger.warning(f"[{run_id_res}] VSEARCH analysis failed: {reason}")
            vsearch_df_map[run_id_res] = pd.DataFrame() # Provide empty df for failed runs
        else:
            vsearch_df_map[run_id_res] = df

    batch_has_primer_signal, batch_consensus_prediction = any(bool(freq) for freq in primer_freq_map.values()), None
    
    if not batch_has_primer_signal:
        logger.warning("No primer signals detected in the entire batch. Determining a consensus region based on VSEARCH alignment.")
        batch_predictions = []
        for run_id in valid_runs:
            primer_scores = score_regions_from_primer_report(primer_freq_map.get(run_id, {}), region_to_pairs_map)
            vsearch_coverage, _ = estimate_16s_subfragment(
                vsearch_df_map.get(run_id, pd.DataFrame()), COMPREHENSIVE_V_REGIONS, VSEARCH_COVERAGE_THRESHOLD
            )
            report = predict_best_region(
                combine_results(vsearch_coverage, primer_scores, all_regions), all_regions, 
                PRIMER_PRESENCE_THRESHOLD, VSEARCH_COVERAGE_THRESHOLD
            )
            if report['prediction'] != "Undetermined": batch_predictions.append(report['prediction'])
        
        if batch_predictions:
            batch_consensus_prediction = Counter(batch_predictions).most_common(1)[0][0]
            logger.info(f"Batch consensus region determined to be '{batch_consensus_prediction}'. This will be applied to all runs.")

    for run_id in valid_runs:
        primer_scores_and_pairs = score_regions_from_primer_report(primer_freq_map.get(run_id, {}), region_to_pairs_map)
        vsearch_coverage_map, _ = estimate_16s_subfragment(vsearch_df_map.get(run_id, pd.DataFrame()), COMPREHENSIVE_V_REGIONS, VSEARCH_COVERAGE_THRESHOLD)
        combined_scores = combine_results(vsearch_coverage_map, primer_scores_and_pairs, all_regions)
        prediction_report = predict_best_region(combined_scores, all_regions, PRIMER_PRESENCE_THRESHOLD, VSEARCH_COVERAGE_THRESHOLD)

        if not batch_has_primer_signal and batch_consensus_prediction:
            prediction_report['prediction'] = batch_consensus_prediction
            prediction_report['reasoning'] = f"Alignment-driven batch consensus: No primer signal detected. Region '{batch_consensus_prediction}' was the most common alignment-based prediction for the batch."
        
        # Add the best primer pair to the report for downstream use
        predicted_region = prediction_report['prediction']
        if predicted_region != "Undetermined":
            best_pair = primer_scores_and_pairs.get(predicted_region, {}).get('best_pair')
            prediction_report['best_primer_pair'] = best_pair

        all_run_reports[run_id] = prediction_report
        
        yaml_data = {'run_accession': run_id, 'predicted_subfragment': prediction_report['prediction'], 'prediction_details': prediction_report}
        with open(yaml_output_dir / f"{run_id}.estimated.yaml", 'w') as f: yaml.dump(yaml_data, f, sort_keys=False, indent=2)
        
        final_prediction = prediction_report['prediction']
        for region_evidence in prediction_report['evidence']['checked_regions']:
            region_name = region_evidence['region']
            vsearch_details = vsearch_coverage_map.get(region_name, {})
            tidy_results.append({ 
                'run_accession': run_id, 'region': region_name, 'primer_score': region_evidence['primer_score'], 
                'vsearch_coverage': region_evidence['vsearch_score'], 
                'vsearch_avg_len': vsearch_details.get('avg_len', 0.0), 
                'vsearch_avg_start': vsearch_details.get('avg_start', 0.0), 
                'vsearch_avg_end': vsearch_details.get('avg_end', 0.0), 
                'is_prediction': (region_name == final_prediction)
            })
    
    if not tidy_results:
        logger.warning("Analysis completed, but no successful results were generated."); return {}
    
    results_df = pd.DataFrame(tidy_results)
    results_df.sort_values(by=['run_accession', 'vsearch_coverage'], ascending=[True, False], inplace=True)
    output_path = output_dir / TSV_OUTPUT_FILENAME
    results_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    
    # Generate and save the alignment visualization
    plot_path = output_dir / "alignment_visualization.png"
    create_alignment_plot(results_df, plot_path)
    
    logger.info(f"\n✅ Analysis complete. {len(all_run_reports)} runs successfully processed.\n"
                f" - Detailed reports saved to: {yaml_output_dir}\n"
                f" - Summary table saved to:   {output_path}\n"
                f" - Alignment plot saved to:  {plot_path}")
    return all_run_reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="16S Variable Region Prediction Workflow.")
    parser.add_argument("input_dir", type=Path, help="Directory containing FASTQ.gz files.")
    parser.add_argument("output_dir", type=Path, help="Directory to save analysis results.")
    parser.add_argument("vsearch_db", type=Path, help="Path to the VSEARCH 16S database.")
    parser.add_argument(
        "--primer_db_dir", type=Path, default=Path.cwd() / "data", 
        help="Directory containing the 'primer_data.db' SQLite database."
    )
    args = parser.parse_args()

    run_files = defaultdict(list)
    for f in args.input_dir.glob("*_1.fastq.gz"):
        run_id = f.name.split('_')[0]
        r2_file = f.parent / f.name.replace("_1.fastq.gz", "_2.fastq.gz")
        if r2_file.exists():
            run_files[run_id].extend([f, r2_file])
        else:
            run_files[run_id].append(f)
            
    if not run_files:
        print(f"Error: No FASTQ files ending in '_1.fastq.gz' found in {args.input_dir}"); exit(1)

    primer_db_path = args.primer_db_dir / PRIMER_DB_NAME
    
    try:
        primer_finder = PrimerFinder(primer_db_path)
        region_to_pairs_map = primer_finder.get_primer_pairs_for_regions()
        if not region_to_pairs_map:
            logger.error("Could not find any primer pairs in the database. Aborting.")
            exit(1)
    except FileNotFoundError as e:
        logger.error(e)
        exit(1)

    asyncio.run(run_comprehensive_analysis(
        run_file_paths=dict(run_files), output_dir=args.output_dir, 
        vsearch_db=args.vsearch_db, primer_db_path=primer_db_path,
        region_to_pairs_map=region_to_pairs_map, threads=4, 
        max_concurrency=10
    ))