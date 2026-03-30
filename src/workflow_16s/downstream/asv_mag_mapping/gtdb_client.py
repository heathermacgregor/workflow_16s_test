"""GTDB 16S rRNA Sequence Retrieval Client

Fetches Genome Taxonomy Database (GTDB) 16S rRNA sequences for MAG reference alignment.
Implements caching to avoid repeated downloads.

API: https://gtdb.ecogenomic.org/
Download: 16S rRNA sequences by taxonomic rank
"""

import logging
from pathlib import Path
from typing import Optional
import requests
import hashlib
from urllib.parse import urljoin
import pickle

logger = logging.getLogger("workflow_16s")


class GTDBClient:
    """
    Client for downloading and managing GTDB 16S rRNA sequences.
    
    Attributes:
        cache_dir: Local directory for cached FASTA files
        base_url: GTDB API base URL
    """
    
    BASE_URL = "https://data.gtdb.ecogenomic.org/"
    LATEST_RELEASE = "release220"  # As of 2026, update as needed
    
    def __init__(self, cache_dir: Optional[Path] = None, logger_obj=None):
        """
        Initialize GTDB client.
        
        Args:
            cache_dir: Path to cache directory. Defaults to ./data/gtdb_cache/
            logger_obj: Logger instance
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/gtdb_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger_obj or logger
        self._metadata_cache = {}
        
    def download_16s_fasta(self, domain: str = "bacteria") -> Path:
        """
        Download GTDB 16S rRNA sequences for specified domain.
        
        Args:
            domain: "bacteria" or "archaea"
            
        Returns:
            Path to FASTA file (cached if available)
            
        Raises:
            ValueError: If domain not in ["bacteria", "archaea"]
            requests.RequestException: If download fails
        """
        if domain not in ["bacteria", "archaea"]:
            raise ValueError(f"Domain must be 'bacteria' or 'archaea', got {domain}")
        
        # Check cache
        cache_file = self.cache_dir / f"gtdb_{self.LATEST_RELEASE}_16s_{domain}.fasta.gz"
        if cache_file.exists():
            self.logger.info(f"✓ GTDB {domain} cache hit: {cache_file}")
            return cache_file
        
        # Construct download URL
        # Format: https://data.gtdb.ecogenomic.org/releases/release220/16s_rna/gtdb_r220_16s_rna.fasta.gz
        url = urljoin(
            self.BASE_URL,
            f"releases/{self.LATEST_RELEASE}/16s_rna/"
            f"gtdb_r{self.LATEST_RELEASE.replace('release', '')}_16s_rna_{domain}.fasta.gz"
        )
        
        try:
            self.logger.info(f"📥 Downloading GTDB {domain} 16S: {url}")
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            # Stream to file
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.logger.info(f"✅ Downloaded {domain} 16S to {cache_file} ({cache_file.stat().st_size / 1e9:.2f} GB)")
            return cache_file
            
        except requests.RequestException as e:
            self.logger.error(f"❌ GTDB download failed: {e}")
            raise
    
    def get_metadata(self, domain: str = "bacteria") -> dict:
        """
        Fetch GTDB metadata (taxonomy, accessions, etc.)
        
        Args:
            domain: "bacteria" or "archaea"
            
        Returns:
            Dictionary mapping genome accession → metadata
        """
        cache_key = f"metadata_{domain}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        # For now, return minimal structure
        # In production: fetch from GTDB metadata API/CSV
        self._metadata_cache[cache_key] = {}
        return self._metadata_cache[cache_key]
    
    def list_available_releases(self) -> list:
        """List available GTDB releases (for future use)"""
        # Could fetch from https://data.gtdb.ecogenomic.org/releases/
        return [self.LATEST_RELEASE]
    
    def validate_cache(self) -> bool:
        """Check if cache is present and valid"""
        fasta_files = list(self.cache_dir.glob("gtdb_*_16s_*.fasta.gz"))
        if fasta_files:
            self.logger.info(f"✓ GTDB cache valid: {len(fasta_files)} files")
            return True
        self.logger.warning("⚠ GTDB cache not found")
        return False
