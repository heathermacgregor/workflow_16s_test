# workflow_16s/api/publication/apis/crossref.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class CrossrefAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.crossref.org/works"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
        self.source_success = {}
        self.api_calls = {'failed': 0}
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str, search_terms: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches publications from Crossref API with rate limiting."""
        publications = []
        params = {
            "query": search_terms if search_terms is not None else accession,
            "rows": 10,
            "mailto": self.email,
            "sort": "published",
            "order": "asc"
        }
        
        try:
            self._rate_limit('crossref')
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            if type(response.json()) != tuple:
                for item in response.json().get('message', {}).get('items', []):
                    title = (item.get('title') or ["Unknown Title"])[0]
                    date_parts = item.get('issued', {}).get('date-parts', [])
                    year = date_parts[0][0] if date_parts and date_parts[0] else None
                    publications.append({
                        "bioproject_accession": accession,
                        "publication_title": title,
                        "pub_year": str(year) if year else "N/A",
                        "doi": item.get('DOI'),
                        "status": "Ready (Crossref)"
                    })
            else: publications = []
            self.source_success['crossref'] = self.source_success.get('crossref', 0) + len(publications)
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Crossref API error for {accession}: {e}")
            self.api_calls['failed'] += 1
        
        return publications