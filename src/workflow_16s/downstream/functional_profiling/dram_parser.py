"""DRAM Output Parser

Parse DRAM (Distilled and Refined Annotation of Metabolism) gene annotations
and build MAG × KEGG ortholog (KO) assignment matrices.

DRAM outputs multiple annotation formats; this parser unifies them.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("workflow_16s")


class AnnotationSource(Enum):
    """Supported annotation sources in DRAM"""
    KEGG = "ko_id"
    PFAM = "pfam_id"
    DBCAN = "dbcan_id"
    CAZY = "cazy_id"
    EC = "ec_number"


@dataclass
class DRAMGene:
    """Single gene annotation from DRAM"""
    gene_id: str
    mag_id: str
    ko_id: Optional[str] = None
    ec_number: Optional[str] = None
    pfam_ids: List[str] = None
    dbcan_id: Optional[str] = None
    locus_tag: Optional[str] = None
    product: Optional[str] = None
    
    def __post_init__(self):
        if self.pfam_ids is None:
            self.pfam_ids = []


class DRAMParser:
    """Parse DRAM output files"""
    
    def __init__(self, dram_dir: Path, logger_obj=None):
        """
        Initialize parser.
        
        Args:
            dram_dir: Path to DRAM output directory
            logger_obj: Logger instance
        """
        self.dram_dir = Path(dram_dir)
        self.logger_obj = logger_obj or logger
        
        if not self.dram_dir.exists():
            raise FileNotFoundError(f"DRAM directory not found: {self.dram_dir}")
        
        self.annotations_file = self.dram_dir / "annotations.tsv"
        self.genes_file = self.dram_dir / "genes.gff"
        self.genes = []
        self.mags = set()
    
    def load_annotations(self) -> List[DRAMGene]:
        """
        Load gene annotations from DRAM tsv.
        
        DRAM annotations.tsv columns:
        - gene_id: Unique gene identifier
        - mag_id: Source MAG identifier
        - ko_id: KEGG Ortholog ID (K#####)
        - ec_number: EC classification
        - pfam_id: Pfam domain IDs
        - dbcan_id: CAZy domain ID
        - product: Gene product/annotation
        - locus_tag: Alternative gene ID
        
        Returns:
            List of DRAMGene objects
        """
        if not self.annotations_file.exists():
            self.logger_obj.warning(f"Annotations file not found: {self.annotations_file}")
            return []
        
        self.logger_obj.info(f"📖 Loading DRAM annotations from {self.annotations_file}")
        
        try:
            df = pd.read_csv(
                self.annotations_file,
                sep='\t',
                dtype={
                    'gene_id': str,
                    'mag_id': str,
                    'ko_id': str,
                    'ec_number': str,
                    'product': str
                },
                keep_default_na=False,  # Don't convert empty strings to NaN
                na_values=['']
            )
            
            genes = []
            for _, row in df.iterrows():
                gene = DRAMGene(
                    gene_id=row.get('gene_id', ''),
                    mag_id=row.get('mag_id', '') or self._extract_mag_id(row.get('gene_id', '')),
                    ko_id=row.get('ko_id', None) if pd.notna(row.get('ko_id')) else None,
                    ec_number=row.get('ec_number', None) if pd.notna(row.get('ec_number')) else None,
                    product=row.get('product', None) if pd.notna(row.get('product')) else None,
                )
                genes.append(gene)
                self.mags.add(gene.mag_id)
            
            self.genes = genes
            self.logger_obj.info(f"✓ Loaded {len(genes)} genes from {len(self.mags)} MAGs")
            
            # Annotation coverage
            ko_count = sum(1 for g in genes if g.ko_id)
            ec_count = sum(1 for g in genes if g.ec_number)
            self.logger_obj.debug(
                f"  Coverage: {ko_count} KO ({100*ko_count//len(genes)}%), "
                f"{ec_count} EC ({100*ec_count//len(genes)}%)"
            )
            
            return genes
            
        except Exception as e:
            self.logger_obj.error(f"❌ Failed to parse DRAM annotations: {e}")
            raise
    
    def _extract_mag_id(self, gene_id: str) -> str:
        """
        Extract MAG identifier from gene_id.
        
        Common formats:
        - MAG_123_00001 → MAG_123
        - sample_binXX_geneID → sample_binXX
        """
        parts = gene_id.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:-1])  # Remove last part (gene number)
        return gene_id
    
    def build_mag_ko_matrix(
        self,
        genes: Optional[List[DRAMGene]] = None
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Build MAG × KO binary assignment matrix.
        
        Aggregates genes to MAG level, creating binary (1/0) matrix indicating
        presence/absence of KOs in each MAG.
        
        Args:
            genes: List of DRAMGene objects (uses self.genes if None)
            
        Returns:
            (DataFrame with MAG × KO, List of MAG IDs, List of KO IDs)
        """
        if genes is None:
            genes = self.genes
        
        if not genes:
            self.logger_obj.warning("No genes loaded; call load_annotations() first")
            return pd.DataFrame(), [], []
        
        self.logger_obj.info("🔧 Building MAG × KO matrix...")
        
        # Filter genes with KO assignment
        genes_with_ko = [g for g in genes if g.ko_id]
        self.logger_obj.info(f"  {len(genes_with_ko)}/{len(genes)} genes have KO annotation")
        
        # Build mapping
        mag_ko_dict = {}
        for gene in genes_with_ko:
            key = (gene.mag_id, gene.ko_id)
            mag_ko_dict[key] = 1  # Binary: presence/absence
        
        # Get unique identifiers
        mag_ids = sorted(set(g.mag_id for g in genes_with_ko))
        ko_ids = sorted(set(g.ko_id for g in genes_with_ko))
        
        # Build DataFrame
        data = []
        for mag in mag_ids:
            row = {ko: mag_ko_dict.get((mag, ko), 0) for ko in ko_ids}
            data.append(row)
        
        df = pd.DataFrame(data, index=mag_ids, columns=ko_ids).astype(np.uint8)
        
        self.logger_obj.info(f"✓ Built {df.shape[0]} MAGs × {df.shape[1]} KOs matrix")
        
        return df, mag_ids, ko_ids
    
    def get_annotation_summary(self) -> Dict[str, int]:
        """
        Get coverage statistics.
        
        Returns:
            Dictionary with "total_genes", "genes_with_ko", "unique_kos", etc.
        """
        if not self.genes:
            return {}
        
        genes_with_ko = [g for g in self.genes if g.ko_id]
        unique_kos = set(g.ko_id for g in genes_with_ko)
        
        summary = {
            'total_genes': len(self.genes),
            'genes_with_ko': len(genes_with_ko),
            'ko_coverage': 100 * len(genes_with_ko) / len(self.genes),
            'unique_kos': len(unique_kos),
            'num_mags': len(self.mags),
        }
        
        return summary
    
    @classmethod
    def from_multiple_runs(
        cls,
        run_dirs: List[Path],
        logger_obj=None
    ) -> 'DRAMParser':
        """
        Aggregate annotations from multiple DRAM runs (e.g., one per MAG).
        
        Args:
            run_dirs: List of DRAM output directories
            logger_obj: Logger instance
            
        Returns:
            Single DRAMParser with all genes
        """
        logger_obj = logger_obj or logger
        logger_obj.info(f"📁 Aggregating {len(run_dirs)} DRAM runs...")
        
        # Use first directory as base
        parser = cls(run_dirs[0], logger_obj)
        parser.load_annotations()
        
        # Load additional runs
        for run_dir in run_dirs[1:]:
            if not Path(run_dir).exists():
                logger_obj.warning(f"Skipping missing directory: {run_dir}")
                continue
            
            subparser = cls(Path(run_dir), logger_obj)
            subparser.load_annotations()
            parser.genes.extend(subparser.genes)
            parser.mags.update(subparser.mags)
        
        logger_obj.info(f"✓ Aggregated {len(parser.genes)} genes from {len(parser.mags)} MAGs")
        
        return parser


def load_dram_workspace(
    dram_dir: Path,
    logger_obj=None
) -> Dict[str, any]:
    """
    Load all DRAM output files into memory.
    
    Args:
        dram_dir: Path to DRAM output directory
        logger_obj: Logger instance
        
    Returns:
        Dictionary with parsed data: {
            'genes': List[DRAMGene],
            'mag_ko_matrix': pd.DataFrame,
            'mag_ids': List[str],
            'ko_ids': List[str],
            'summary': Dict[str, int],
        }
    """
    logger_obj = logger_obj or logger
    
    parser = DRAMParser(dram_dir, logger_obj)
    genes = parser.load_annotations()
    mag_ko, mag_ids, ko_ids = parser.build_mag_ko_matrix(genes)
    summary = parser.get_annotation_summary()
    
    return {
        'parser': parser,
        'genes': genes,
        'mag_ko_matrix': mag_ko,
        'mag_ids': mag_ids,
        'ko_ids': ko_ids,
        'summary': summary,
    }
