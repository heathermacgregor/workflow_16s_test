import json
import requests
import logging
from typing import List, Dict, Any, Optional
from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class ZenodoAPI(BaseAPI):
    def __init__(self, config, email, logger):
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://zenodo.org/api/records"
        self.rate_limit_seconds = 1.0  # Zenodo is generous but we should be polite

    @with_http_backoff(max_retries=3, base_delay=2.0)
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = []
        # Search for the accession
        params = {"q": f'"{accession}"', "size": 5}

        try:
            self._rate_limit('zenodo')
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            hits = response.json().get('hits', {}).get('hits', [])
            
            for hit in hits:
                metadata = hit.get('metadata', {})
                record_id = hit.get('id')
                
                # 🟢 NEW: File Discovery Logic
                # Check for supplementary spreadsheets/text files
                supplementary_text = self._discovery_supplementary_content(record_id)
                
                publications.append({
                    "bioproject_accession": accession,
                    "publication_title": metadata.get('title', "Zenodo Record"),
                    "pub_year": metadata.get('publication_date', 'N/A')[:4],
                    "doi": hit.get('doi') or metadata.get('doi'),
                    "zenodo_id": record_id,
                    "supplementary_content": supplementary_text, # Passed to LLM later
                    "status": "Ready (Zenodo)"
                })
        except Exception as e:
            self.logger.warning(f"Zenodo search failed for {accession}: {e}")
        return publications

    def _discovery_supplementary_content(self, record_id: str) -> str:
        """Scans Zenodo file manifests for mapping files or metadata."""
        discovered_text = ""
        try:
            # Hit the files endpoint for this specific record
            files_url = f"https://zenodo.org/api/records/{record_id}/files"
            resp = self.session.get(files_url, timeout=10)
            if resp.status_code == 200:
                files = resp.json() # List of file objects
                for f in files:
                    fname = f.get('key', '').lower()
                    # Look for high-value targets
                    if any(k in fname for k in ['primer', 'mapping', 'metadata', 'methods', 's1']):
                        discovered_text += f" [File Found: {f.get('key')} - {f.get('links', {}).get('self')}]"
        except:
            pass
        return discovered_text