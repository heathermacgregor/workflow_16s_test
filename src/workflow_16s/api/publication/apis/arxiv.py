# workflow_16s/api/publication/apis/arxiv.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class ArxivAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = []
        params = {
            "search_query": f'all:"{accession}"', 
            "sortBy": "submittedDate", 
            "sortOrder": "ascending", 
            "max_results": 10
        }
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        root, ns = ET.fromstring(response.content), {'a': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('a:entry', ns):
            title, year = entry.find('a:title', ns).text.strip(), entry.find('a:published', ns).text[:4] # type: ignore
            doi_link = entry.find('a:link[@title="doi"]', ns)
            doi = doi_link.attrib.get('href', '').split('doi.org/')[-1] if doi_link is not None else None
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": title, 
                "pub_year": str(year), 
                "doi": doi, 
                "status": "Ready (ArXiv)"
            })
        return publications