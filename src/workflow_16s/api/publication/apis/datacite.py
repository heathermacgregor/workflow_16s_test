# workflow_16s/api/publication/apis/datacite.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class DataciteAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.datacite.org/works"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = [] 
        params = {
            "query": accession, 
            "page[size]": 10, 
            "sort": "published"
        }
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('data', []):
            attrs = item.get('attributes', {})
            titles_list = attrs.get('titles', [])
            title = titles_list[0].get('title', "Unknown Title") if titles_list else "Unknown Title"
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": title, 
                "pub_year": str(attrs.get('publicationYear', 'N/A')), 
                "doi": attrs.get('doi'), 
                "status": "Ready (DataCite)"
            })
        return publications