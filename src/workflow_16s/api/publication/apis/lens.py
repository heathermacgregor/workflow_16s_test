import json
import requests
import logging
from typing import List, Dict, Any, Optional
from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class LensAPI(BaseAPI):
    def __init__(self, config, email, logger):
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.lens.org/scholarly/search"
        self.api_key = getattr(config.credentials, 'lens_api_key', None)
        self.rate_limit_seconds = 1.0 

    @with_http_backoff(max_retries=3, base_delay=2.0)
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        if not self.api_key:
            self.logger.debug("Lens API key missing, skipping tier.")
            return []

        publications = []
        # Lens uses a rich JSON-based query language
        payload = {
            "query": {
                "query_string": f"\"{accession}\""
            },
            "size": 10,
            "sort": [{"year_published": "desc"}]
        }
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

        try:
            self._rate_limit('lens')
            response = self.session.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            for hit in data.get('data', []):
                # Extract year safely
                year = hit.get('year_published', 'N/A')
                
                # Extract DOI safely
                external_ids = hit.get('external_ids', [])
                doi = next((i.get('value') for i in external_ids if i.get('type') == 'doi'), None)

                publications.append({
                    "bioproject_accession": accession,
                    "publication_title": hit.get('title', "Unknown Title"),
                    "pub_year": str(year),
                    "doi": doi,
                    "lens_id": hit.get('lens_id'),
                    "status": "Ready (Lens.org)"
                })

        except Exception as e:
            self.logger.warning(f"Lens.org search failed for {accession}: {e}")

        return publications