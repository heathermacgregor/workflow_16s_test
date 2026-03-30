"""VSEARCH Wrapper for ASV-to-MAG Alignment

Wraps VSEARCH command-line tool for high-throughput ASV alignment against MAG 16S databases.

VSEARCH Parameters:
- --id: Sequence identity threshold (default: 0.99 = 99%)
- --query_cov: Query coverage threshold (default: 0.99)
- --db_cov: Database coverage threshold (optional)
- --top_hits_only: Return only top hit per query
- --blast6out: BLAST-6 format output (9 fields)

Reference: https://github.com/torognes/vsearch
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import tempfile
import pandas as pd
import numpy as np
from scipy import sparse

logger = logging.getLogger("workflow_16s")


class VSEARCHWrapper:
    """
    Wrapper for VSEARCH command-line tool.
    
    Handles:
    - Query/database preparation
    - VSEARCH subprocess execution
    - Output parsing (BLAST-6 format)
    - Error handling
    """
    
    def __init__(self, vsearch_bin: str = "vsearch", logger_obj=None):
        """
        Initialize VSEARCH wrapper.
        
        Args:
            vsearch_bin: Path to vsearch binary (default: $PATH)
            logger_obj: Logger instance
        """
        self.vsearch_bin = vsearch_bin
        self.logger = logger_obj or logger
        self._verify_binary()
    
    def _verify_binary(self):
        """Check if vsearch binary is available"""
        try:
            result = subprocess.run(
                [self.vsearch_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.logger.debug(f"✓ VSEARCH available: {result.stdout.strip()}")
            else:
                raise RuntimeError(f"VSEARCH check failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.logger.error(f"❌ VSEARCH binary not found: {self.vsearch_bin}")
            raise RuntimeError(f"VSEARCH not available: {e}")
    
    def run_search(
        self,
        query_fasta: Path,
        db_fasta: Path,
        identity: float = 0.99,
        query_cov: float = 0.99,
        db_cov: Optional[float] = None,
        top_hits_only: bool = True,
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Run VSEARCH alignment query against database.
        
        Args:
            query_fasta: Path to query FASTA (ASVs)
            db_fasta: Path to database FASTA (MAG 16S)
            identity: Identity threshold (0.0-1.0), default 0.99
            query_cov: Query coverage threshold (0.0-1.0), default 0.99
            db_cov: Database coverage threshold, optional
            top_hits_only: Return only top hit per query
            threads: Number of threads
            
        Returns:
            DataFrame with BLAST-6 format columns:
            [qseqid, sseqid, pident, alnlen, mismatch, gapopen, qstart, qend, sstart, send, evalue, bitscore]
        """
        query_fasta = Path(query_fasta)
        db_fasta = Path(db_fasta)
        
        if not query_fasta.exists():
            raise FileNotFoundError(f"Query FASTA not found: {query_fasta}")
        if not db_fasta.exists():
            raise FileNotFoundError(f"Database FASTA not found: {db_fasta}")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            output_file = tmp.name
        
        try:
            # Build VSEARCH command
            cmd = [
                self.vsearch_bin,
                "--usearch_global", str(query_fasta),
                "--db", str(db_fasta),
                "--id", str(identity),
                "--query_cov", str(query_cov),
                "--blast6out", output_file,
                "--threads", str(threads),
            ]
            
            if db_cov is not None:
                cmd.extend(["--db_cov", str(db_cov)])
            
            if top_hits_only:
                cmd.append("--top_hits_only")
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=True
            )
            
            self.logger.info(f"✓ VSEARCH search complete")
            
            # Parse output
            if Path(output_file).stat().st_size == 0:
                self.logger.warning("⚠ VSEARCH returned no hits")
                return pd.DataFrame()
            
            df = pd.read_csv(
                output_file,
                sep='\t',
                header=None,
                names=['qseqid', 'sseqid', 'pident', 'alnlen', 'mismatch',
                       'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
            )
            
            self.logger.info(f"✓ Parsed {len(df)} alignments from VSEARCH output")
            return df
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ VSEARCH failed: {e.stderr}")
            raise
        finally:
            # Cleanup
            Path(output_file).unlink(missing_ok=True)
    
    def filter_alignments(
        self,
        alignments: pd.DataFrame,
        identity_threshold: float = 0.99,
        length_threshold: int = 50,
    ) -> pd.DataFrame:
        """
        Filter alignment results by quality criteria.
        
        Args:
            alignments: BLAST-6 format DataFrame
            identity_threshold: Minimum percent identity (0-100)
            length_threshold: Minimum alignment length
            
        Returns:
            Filtered DataFrame
        """
        if alignments.empty:
            return alignments
        
        # Convert percent identity (0-100) to fraction (0-1)
        thresh_frac = identity_threshold if identity_threshold <= 1.0 else identity_threshold / 100.0
        
        filtered = alignments[
            (alignments['pident'] >= thresh_frac * 100) &
            (alignments['alnlen'] >= length_threshold)
        ].copy()
        
        self.logger.info(
            f"Filtered alignments: {len(alignments)} → {len(filtered)} "
            f"({len(filtered)/len(alignments)*100:.1f}%)"
        )
        
        return filtered


def run_vsearch_alignment(
    query_fasta: Path,
    db_fasta: Path,
    identity: float = 0.99,
    query_cov: float = 0.99,
    threads: int = 4,
    logger_obj=None,
) -> pd.DataFrame:
    """
    Convenience function to run VSEARCH alignment.
    
    Args:
        query_fasta: Path to query FASTA
        db_fasta: Path to database FASTA
        identity: Identity threshold (0.0-1.0)
        query_cov: Query coverage threshold
        threads: Number of threads
        logger_obj: Logger instance
        
    Returns:
        Alignment results DataFrame
    """
    wrapper = VSEARCHWrapper(logger_obj=logger_obj)
    return wrapper.run_search(
        query_fasta=query_fasta,
        db_fasta=db_fasta,
        identity=identity,
        query_cov=query_cov,
        threads=threads,
    )
