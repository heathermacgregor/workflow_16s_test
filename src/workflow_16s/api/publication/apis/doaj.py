# workflow_16s/api/publication/apis/doaj.py

import json
import requests
import logging
from typing import List, Dict, Any, Optional
from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class DOAJAPI(BaseAPI):
    def __init__(self, config, email, logger):
        super().__init__(config, email)
        self.logger = logger
        # 🟢 FIX: The endpoint changed from /search/articles to /articles/search
        self.base_url = "https://doaj.org/api/articles/search"
        self.rate_limit_seconds = 0.5 # Slightly more conservative

    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = []
        
        # 🟢 FIX: Updated query syntax for DOAJ API v3
        # We append the query directly to the URL path or use 'q' param depending on the specific version
        search_url = f"{self.base_url}/bibjson.abstract:\"{accession}\""
        
        params = {
            "pageSize": 10,
            "sort": "created_date:asc"
        }
        
        try:
            # Use the pooled session
            response = self.session.get(search_url, params=params, timeout=15)
            
            # If the direct path search fails, fallback to the standard 'q' param
            if response.status_code == 404:
                response = self.session.get(self.base_url, params={"q": accession}, timeout=15)
                
            response.raise_for_status()
            data = response.json()

            # 🟢 BUG SHIELD: Check if we got a dict
            if not isinstance(data, dict):
                return []

            for item in data.get('results', []):
                bibjson = item.get('bibjson', {})
                
                # Safer DOI extraction
                doi = None
                for identifier in bibjson.get('identifier', []):
                    if identifier.get('type') == 'doi':
                        doi = identifier.get('id')
                        break
                
                publications.append({
                    "bioproject_accession": accession,
                    "publication_title": bibjson.get('title', "Unknown Title"),
                    "pub_year": str(bibjson.get('year', 'N/A')),
                    "doi": doi,
                    "status": "Ready (DOAJ)"
                })
        except Exception as e:
            self.logger.warning(f"DOAJ search failed for {accession}: {e}")

        return publications