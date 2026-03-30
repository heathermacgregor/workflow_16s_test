# workflow_16s/api/publication/apis/ncbi.py

import json
import requests
import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class NCBIAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_seconds = 0.34 # 3 requests/sec

    @with_http_backoff()
    def _make_request(self, endpoint: str, params: dict) -> dict:
        self._wait_for_rate_limit()
        url = f"{self.base_url}/{endpoint}"
        # 🟢 FIX: Use self.session (the pooled one) or requests
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        # 🟢 BUG SHIELD: Check if response is empty before .json()
        if not response.text.strip():
            return {}
        return response.json()

    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        """
        Fetches publications from NCBI PubMed using a robust 3-step eutils process.
        """
        publications = []
        
        try:
            # STEP 1: esearch to resolve the BioProject accession string into an internal NCBI UID
            self._rate_limit('ncbi')
            esearch_params = {
                "db": "bioproject",
                "term": accession,
                "retmode": "json",
                "tool": "PublicationExtractor",
                "email": self.email
            }
            esearch_data = self._make_request("esearch.fcgi", esearch_params)
            uid_list = esearch_data.get('esearchresult', {}).get('idlist', [])
            
            if not uid_list:
                self.logger.debug(f"No internal UID found for BioProject accession {accession}")
                return []
                
            bioproject_uid = uid_list[0]
            
            # STEP 2: elink to map the BioProject UID to PubMed IDs (PMIDs)
            self._rate_limit('ncbi')
            elink_params = {
                "dbfrom": "bioproject",
                "db": "pubmed", # Explicitly searching PubMed, not PMC
                "id": bioproject_uid,
                "retmode": "json",
                "tool": "PublicationExtractor",
                "email": self.email
            }
            elink_data = self._make_request("elink.fcgi", elink_params)
            linksets = elink_data.get('linksets', [])
            pmids = [
                link['Id']
                for ls in linksets
                for lsd in ls.get('linksetdbs', []) if lsd.get('dbto') == 'pubmed'
                for link in lsd.get('links', [])
            ]
            
            if not pmids:
                self.logger.debug(f"No PubMed links found for {accession} (UID: {bioproject_uid})")
                return []
            
            # Limit to top 5 to avoid excessive API calls downstream
            pmids = pmids[:5]
            self.logger.debug(f"Found {len(pmids)} PMIDs for {accession}")
            
            # STEP 3: esummary to fetch the actual publication metadata
            self._rate_limit('ncbi')
            esummary_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "tool": "PublicationExtractor",
                "email": self.email
            }
            esummary_data = self._make_request("esummary.fcgi", esummary_params)
            summary_data = esummary_data.get('result', {})
            
            for pmid in pmids:
                if pmid in summary_data:
                    article = summary_data[pmid]
                    
                    # Extract DOI safely
                    doi = ""
                    for article_id in article.get('articleids', []):
                        if article_id.get('idtype') == 'doi':
                            doi = article_id.get('value')
                            break
                    
                    # Extract year
                    pub_date = article.get('pubdate', '')
                    pub_year = pub_date.split()[0] if pub_date else "N/A"
                    
                    publications.append({
                        "bioproject_accession": accession,
                        "publication_title": article.get('title', ''),
                        "pub_year": pub_year,
                        "doi": doi,
                        "pmid": pmid,
                        "source": "NCBI PubMed",
                        "status": "Ready"
                    })
                    
        except requests.exceptions.RequestException as e:
            # If the helper exhausts all 5 retries, the final error bubbles up here safely
            self.logger.warning(f"NCBI lookup failed for '{accession}': {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse NCBI JSON response for '{accession}': {e}")
            
        return publications
    
    @with_http_backoff()
    def get_pmc_links(self, doi: str, session: Optional[requests.Session] = None) -> Tuple[Optional[str], Optional[str]]:
        """Searches NCBI for a given DOI to find links to PubMed Central (PMC) safely within a thread."""
        pdf_url, article_url = None, None
        req_method = session.get if session else requests.get
        
        try:
            converter_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            params = {
                'ids': doi, 
                'format': 'json', 
                'tool': 'publication_fetcher', 
                'email': self.email
            }
            response = req_method(converter_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'records' in data and data['records']:
                pmcid = data['records'][0].get('pmcid')
                if pmcid:
                    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    efetch_params = {
                        'db': 'pmc', 
                        'id': pmcid, 
                        'retmode': 'xml', 
                        'tool': 'publication_fetcher', 
                        'email': self.email
                    }
                    
                    efetch_response = req_method(efetch_url, params=efetch_params, timeout=15)
                    efetch_response.raise_for_status()
                    
                    root = ET.fromstring(efetch_response.content)
                    pdf_link_element = root.find(f".//link[@format='pdf'][@href]")
                    if pdf_link_element is not None:
                        pdf_url = pdf_link_element.get('href')
                        
        except (requests.RequestException, ET.ParseError, IndexError):
            # Fail silently and return None
            pass
            
        return pdf_url, article_url