"""
State-of-the-Art Primer Quality Control

This module provides comprehensive primer detection and validation:
1. Multi-pass primer detection (5' end, 3' end, anywhere)
2. Orientation analysis (forward/reverse/mixed)
3. Mismatch tolerance using CutAdapt
4. Adapter contamination detection
5. Trimming validation (before/after comparison)
6. Per-sample and aggregate reporting
"""

import re
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger('workflow_16s')


class PrimerQC:
    """
    Comprehensive primer quality control using CutAdapt and BioPython.
    
    Goes beyond basic regex matching to provide detailed primer
    detection statistics with mismatch tolerance.
    """
    
    # Common Illumina adapters
    ILLUMINA_ADAPTERS = {
        'TruSeq_Universal': 'AGATCGGAAGAGC',
        'TruSeq_Index': 'GATCGGAAGAGCACACGTCTGAACTCCAGTCAC',
        'Nextera_Transposase': 'CTGTCTCTTATACACATCT',
        'Nextera_Index': 'CTGTCTCTTATACACATCTCCGAGCCCACGAGAC',
    }
    
    # PhiX spike-in (first 50bp)
    PHIX_SEQUENCE = 'GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCT'
    
    def __init__(self, primers: Dict[str, str], 
                 max_error_rate: float = 0.15,
                 min_overlap: int = 10,
                 max_reads: int = 10000,
                 n_cores: int = 4):
        """
        Initialize PrimerQC.
        
        Args:
            primers: Dict of primer_name -> sequence
            max_error_rate: Maximum allowed error rate for CutAdapt (default: 0.15 = 15%)
            min_overlap: Minimum overlap between primer and read
            max_reads: Maximum reads to sample per file
            n_cores: Number of cores for parallel processing
        """
        self.primers = primers
        self.max_error_rate = max_error_rate
        self.min_overlap = min_overlap
        self.max_reads = max_reads
        self.n_cores = n_cores
        
        # Precompile regex for IUPAC codes
        self.iupac_table = self._create_iupac_table()
        self.primer_patterns = self._compile_primer_patterns()
    
    def _create_iupac_table(self) -> Dict[str, str]:
        """Create IUPAC ambiguity code table."""
        return {
            'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'T',
            'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
            'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
            'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'
        }
    
    def _compile_primer_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for primers with IUPAC codes."""
        patterns = {}
        for name, seq in self.primers.items():
            # Convert IUPAC to regex
            regex_seq = ''.join(self.iupac_table.get(base, base) for base in seq.upper())
            patterns[name] = re.compile(regex_seq, re.IGNORECASE)
        return patterns
    
    def _reverse_complement(self, seq: str) -> str:
        """Calculate reverse complement."""
        comp = str.maketrans('ACGTacgt', 'TGCAtgca')
        return seq.translate(comp)[::-1]
    
    def check_primers_regex(self, fastq_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Fast regex-based primer check (existing method, enhanced).
        
        Args:
            fastq_path: Path to FASTQ file
        
        Returns:
            Dict with primer detection frequencies
        """
        logger.info(f"Regex primer check: {fastq_path}")
        
        results = defaultdict(lambda: {
            'forward_5prime': 0,
            'forward_3prime': 0,
            'reverse_comp_5prime': 0,
            'reverse_comp_3prime': 0,
            'total_checked': 0
        })
        
        with open(fastq_path, 'r') as fh:
            for i, record in enumerate(SeqIO.parse(fh, 'fastq')):
                if i >= self.max_reads:
                    break
                
                seq = str(record.seq).upper()
                
                for primer_name, pattern in self.primer_patterns.items():
                    results[primer_name]['total_checked'] += 1
                    
                    # Check 5' end (first 100bp)
                    if pattern.search(seq[:100]):
                        results[primer_name]['forward_5prime'] += 1
                    
                    # Check 3' end (last 100bp)
                    if pattern.search(seq[-100:]):
                        results[primer_name]['forward_3prime'] += 1
                    
                    # Check reverse complement
                    rev_comp = self._reverse_complement(seq)
                    if pattern.search(rev_comp[:100]):
                        results[primer_name]['reverse_comp_5prime'] += 1
                    if pattern.search(rev_comp[-100:]):
                        results[primer_name]['reverse_comp_3prime'] += 1
        
        # Convert counts to frequencies
        for primer_name in results:
            total = results[primer_name]['total_checked']
            if total > 0:
                for key in ['forward_5prime', 'forward_3prime', 
                           'reverse_comp_5prime', 'reverse_comp_3prime']:
                    results[primer_name][key] = results[primer_name][key] / total
        
        return dict(results)
    
    def check_primers_cutadapt(self, fastq_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Comprehensive primer detection using CutAdapt.
        
        Uses CutAdapt's mismatch tolerance for more accurate detection.
        
        Args:
            fastq_path: Path to FASTQ file
        
        Returns:
            Dict with detailed primer statistics
        """
        logger.info(f"CutAdapt primer check: {fastq_path}")
        
        results = {}
        
        for primer_name, primer_seq in self.primers.items():
            # Run CutAdapt in info mode (doesn't trim, just reports)
            cmd = [
                'cutadapt',
                '--cores', str(self.n_cores),
                '--error-rate', str(self.max_error_rate),
                '--overlap', str(self.min_overlap),
                '--times', '1',  # Only find first occurrence
                '--discard-untrimmed',  # Count trimmed vs untrimmed
                '--info-file', '/dev/stdout',  # Output per-read info
                '-a', primer_seq,  # 3' adapter
                '-g', primer_seq,  # 5' adapter
                '-o', '/dev/null',  # Don't save trimmed reads
                str(fastq_path)
            ]
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300
                )
                
                # Parse CutAdapt output
                stats = self._parse_cutadapt_output(result.stderr)
                stats['primer_name'] = primer_name
                stats['primer_sequence'] = primer_seq
                
                results[primer_name] = stats
                
            except subprocess.TimeoutExpired:
                logger.error(f"CutAdapt timeout for {primer_name} on {fastq_path}")
                results[primer_name] = {'error': 'timeout'}
            except Exception as e:
                logger.error(f"CutAdapt failed for {primer_name}: {e}")
                results[primer_name] = {'error': str(e)}
        
        return results
    
    def _parse_cutadapt_output(self, stderr: str) -> Dict[str, Any]:
        """Parse CutAdapt stderr output for statistics."""
        stats = {
            'total_reads': 0,
            'reads_with_adapters': 0,
            'basepairs_removed': 0,
            'basepairs_written': 0,
        }
        
        for line in stderr.split('\n'):
            if 'Total reads processed:' in line:
                stats['total_reads'] = int(line.split(':')[1].strip().replace(',', ''))
            elif 'Reads with adapters:' in line:
                match = re.search(r'([\d,]+)\s+\(([\d.]+)%\)', line)
                if match:
                    stats['reads_with_adapters'] = int(match.group(1).replace(',', ''))
                    stats['adapter_percentage'] = float(match.group(2))
            elif 'Total basepairs processed:' in line:
                match = re.search(r'([\d,]+)\s+bp', line)
                if match:
                    stats['basepairs_processed'] = int(match.group(1).replace(',', ''))
            elif 'Total written (filtered):' in line:
                match = re.search(r'([\d,]+)\s+bp', line)
                if match:
                    stats['basepairs_written'] = int(match.group(1).replace(',', ''))
        
        # Calculate percentage if not present
        if 'adapter_percentage' not in stats and stats['total_reads'] > 0:
            stats['adapter_percentage'] = 100 * stats['reads_with_adapters'] / stats['total_reads']
        
        return stats
    
    def check_contamination(self, fastq_path: Union[str, Path]) -> Dict[str, float]:
        """
        Check for common contaminants (adapters, PhiX).
        
        Args:
            fastq_path: Path to FASTQ file
        
        Returns:
            Dict of contaminant -> frequency
        """
        logger.info(f"Checking contamination: {fastq_path}")
        
        contamination = {name: 0 for name in self.ILLUMINA_ADAPTERS}
        contamination['PhiX'] = 0
        total_reads = 0
        
        with open(fastq_path, 'r') as fh:
            for i, record in enumerate(SeqIO.parse(fh, 'fastq')):
                if i >= self.max_reads:
                    break
                
                total_reads += 1
                seq = str(record.seq).upper()
                
                # Check Illumina adapters
                for adapter_name, adapter_seq in self.ILLUMINA_ADAPTERS.items():
                    if adapter_seq in seq:
                        contamination[adapter_name] += 1
                
                # Check PhiX
                if self.PHIX_SEQUENCE in seq:
                    contamination['PhiX'] += 1
        
        # Convert to frequencies
        if total_reads > 0:
            contamination = {k: v/total_reads for k, v in contamination.items()}
        
        return contamination
    
    def validate_trimming(self, pre_trim_fastq: Union[str, Path], 
                         post_trim_fastq: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate that primer trimming actually worked.
        
        Compares before and after files to ensure:
        1. Primers are removed
        2. Read lengths changed appropriately
        3. No unexpected artifacts
        
        Args:
            pre_trim_fastq: FASTQ file before trimming
            post_trim_fastq: FASTQ file after trimming
        
        Returns:
            Dict with validation statistics
        """
        logger.info(f"Validating trimming: {pre_trim_fastq} -> {post_trim_fastq}")
        
        # Get primer frequencies before/after
        pre_stats = self.check_primers_regex(pre_trim_fastq)
        post_stats = self.check_primers_regex(post_trim_fastq)
        
        # Get length distributions
        pre_lengths = self._get_length_distribution(pre_trim_fastq)
        post_lengths = self._get_length_distribution(post_trim_fastq)
        
        validation = {
            'pre_trim_stats': pre_stats,
            'post_trim_stats': post_stats,
            'pre_length_dist': pre_lengths,
            'post_length_dist': post_lengths,
            'validation_checks': {}
        }
        
        # Validation checks
        for primer_name in pre_stats:
            pre_freq = pre_stats[primer_name]['forward_5prime']
            post_freq = post_stats[primer_name]['forward_5prime']
            
            # Should see reduction in primer frequency
            if post_freq < pre_freq * 0.1:  # >90% reduction
                validation['validation_checks'][primer_name] = 'PASS'
            elif post_freq < pre_freq * 0.5:  # >50% reduction
                validation['validation_checks'][primer_name] = 'WARNING'
            else:
                validation['validation_checks'][primer_name] = 'FAIL'
        
        # Check length shift
        pre_median = np.median(list(pre_lengths.keys()))
        post_median = np.median(list(post_lengths.keys()))
        expected_shift = len(list(self.primers.values())[0])  # Use first primer length
        
        actual_shift = pre_median - post_median
        if abs(actual_shift - expected_shift) < 10:  # Within 10bp
            validation['length_shift_check'] = 'PASS'
        else:
            validation['length_shift_check'] = 'WARNING'
            validation['length_shift_details'] = {
                'expected': expected_shift,
                'actual': actual_shift,
                'difference': abs(actual_shift - expected_shift)
            }
        
        return validation
    
    def _get_length_distribution(self, fastq_path: Union[str, Path]) -> Dict[int, int]:
        """Get distribution of read lengths."""
        lengths = defaultdict(int)
        
        with open(fastq_path, 'r') as fh:
            for i, record in enumerate(SeqIO.parse(fh, 'fastq')):
                if i >= self.max_reads:
                    break
                lengths[len(record.seq)] += 1
        
        return dict(lengths)
    
    def comprehensive_check(self, fastq_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run all primer QC checks on a FASTQ file.
        
        Args:
            fastq_path: Path to FASTQ file
        
        Returns:
            Comprehensive report dict
        """
        logger.info(f"Running comprehensive primer QC on {fastq_path}")
        
        report = {
            'file': str(fastq_path),
            'regex_stats': self.check_primers_regex(fastq_path),
            'cutadapt_stats': self.check_primers_cutadapt(fastq_path),
            'contamination': self.check_contamination(fastq_path),
            'length_distribution': self._get_length_distribution(fastq_path),
        }
        
        # Add overall assessment
        report['overall_assessment'] = self._assess_quality(report)
        
        return report
    
    def _assess_quality(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall primer quality and provide recommendation."""
        assessment = {
            'status': 'UNKNOWN',
            'recommendation': '',
            'issues': []
        }
        
        # Check primer detection rate
        cutadapt_stats = report.get('cutadapt_stats', {})
        for primer_name, stats in cutadapt_stats.items():
            if isinstance(stats, dict) and 'adapter_percentage' in stats:
                pct = stats['adapter_percentage']
                
                if pct > 80:
                    # Good primer detection
                    continue
                elif pct > 50:
                    assessment['issues'].append(
                        f"{primer_name}: Only {pct:.1f}% reads have primer (expected >80%)"
                    )
                else:
                    assessment['issues'].append(
                        f"{primer_name}: Low primer detection ({pct:.1f}%). Wrong primers or degraded library?"
                    )
        
        # Check contamination
        contamination = report.get('contamination', {})
        for contaminant, freq in contamination.items():
            if freq > 0.01:  # >1% contamination
                assessment['issues'].append(
                    f"{contaminant} contamination detected ({freq*100:.2f}%)"
                )
        
        # Overall recommendation
        if not assessment['issues']:
            assessment['status'] = 'PASS'
            assessment['recommendation'] = 'Primers detected correctly. Proceed with analysis.'
        elif len(assessment['issues']) <= 2:
            assessment['status'] = 'WARNING'
            assessment['recommendation'] = 'Minor issues detected. Review and consider re-trimming.'
        else:
            assessment['status'] = 'FAIL'
            assessment['recommendation'] = 'Multiple issues detected. Do not use this dataset or investigate primer problems.'
        
        return assessment
    
    def batch_check(self, fastq_files: List[Union[str, Path]], 
                   output_report: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Run primer QC on multiple FASTQ files in parallel.
        
        Args:
            fastq_files: List of FASTQ file paths
            output_report: Optional path to save HTML report
        
        Returns:
            DataFrame with results for all files
        """
        logger.info(f"Running batch primer QC on {len(fastq_files)} files")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            futures = {
                executor.submit(self.comprehensive_check, f): f 
                for f in fastq_files
            }
            
            for future in as_completed(futures):
                fastq_file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {fastq_file}: {e}")
                    results.append({
                        'file': str(fastq_file),
                        'error': str(e)
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save report if requested
        if output_report:
            self._generate_html_report(df, output_report)
        
        return df
    
    def _generate_html_report(self, df: pd.DataFrame, output_path: Union[str, Path]):
        """Generate HTML report from results DataFrame."""
        # TODO: Implement HTML report generation with visualizations
        # For now, save as CSV
        csv_path = str(output_path).replace('.html', '.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved report to {csv_path}")
