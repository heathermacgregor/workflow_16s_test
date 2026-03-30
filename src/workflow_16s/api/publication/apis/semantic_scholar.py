# workflow_16s/api/publication/apis/semantic_scholar.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class SemanticScholarAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()   
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = [] 
        params = {
            "query": accession, 
            "fields": "title,year,externalIds", 
            "limit": 10
        }
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('data', []):
            doi = item.get('externalIds', {}).get('DOI')
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(item.get('year', 'N/A')), 
                "doi": doi, 
                "status": "Ready (Semantic Scholar)"
            })
        return publications