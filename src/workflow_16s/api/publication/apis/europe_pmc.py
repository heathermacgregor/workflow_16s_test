# workflow_16s/api/publication/apis/crossref.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class EuropePMCAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"query": accession, "resultType": "lite", "format": "json", "pageSize": 10}
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('resultList', {}).get('result', []):
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(item.get('pubYear', 'N/A')), 
                "doi": item.get('doi'), 
                "status": "Ready (Europe PMC)"
            })
        return publications
    
    def _fetch_si_text(self, id_1: str, id_2: str = "", *args, **kwargs) -> str:
        """
        Attempts to fetch Supplementary Information (SI) using the Europe PMC REST API.
        Accepts multiple identifiers (e.g., PMID and DOI) and automatically uses the best one.
        """
        si_content = []
        # Grab the thread-safe session if provided, otherwise fallback to the default
        request_session = kwargs.get('session', self.session)
        # Intelligently grab the DOI if it exists in either argument, otherwise fallback to the first ID
        identifier = str(id_1)
        if id_2 and '/' in str(id_2):
            identifier = str(id_2)
        elif '/' in str(id_1):
            identifier = str(id_1)
            
        if not identifier or identifier.lower() == 'nan':
            return ""
        
        try:
            self.logger.debug(f"Querying Europe PMC for SI text: {identifier}")
            
            # STEP 1: Resolve the identifier (DOI or PMID) to a PMCID
            query_str = f'DOI:"{identifier}"' if '/' in identifier else f'EXT_ID:"{identifier}"'
            
            search_params = {
                "query": query_str,
                "format": "json",
                "resultType": "core"
            }
            
            search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            # Change self.http_session to request_session
            search_resp = request_session.get(search_url, params=search_params, timeout=self.timeout)
            #search_resp = self.http_session.get(search_url, params=search_params, timeout=self.timeout)
            search_resp.raise_for_status()
            
            results = search_resp.json().get("resultList", {}).get("result", [])
            if not results:
                self.logger.debug(f"No Europe PMC record found for {identifier}")
                return ""
                
            pmcid = results[0].get("pmcid")
            if not pmcid:
                self.logger.debug(f"No open-access PMCID available for {identifier}")
                return ""

            # STEP 2: Fetch the full-text XML using the PMCID
            xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            # Change self.http_session to request_session
            xml_resp = request_session.get(xml_url, timeout=self.timeout)
            #xml_resp = self.http_session.get(xml_url, timeout=self.timeout)
            
            if xml_resp.status_code != 200:
                self.logger.debug(f"Could not retrieve full text XML for {pmcid}")
                return ""
                
            # STEP 3: Parse the XML to find supplementary material sections
            root = ET.fromstring(xml_resp.content)
            
            # Find standard JATS XML supplementary tags
            # 1. Look for sections designated as supplementary material
            for supp_node in root.findall('.//sec[@sec-type="supplementary-material"]'):
                text_pieces = [text.strip() for text in supp_node.itertext() if text.strip()]
                if text_pieces:
                    si_content.append(" ".join(text_pieces))
                    
            # 2. Look for explicit supplementary-material inline tags (often contains captions)
            for supp_node in root.findall('.//supplementary-material'):
                text_pieces = [text.strip() for text in supp_node.itertext() if text.strip()]
                if text_pieces:
                    si_content.append(" ".join(text_pieces))
                    
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Network error fetching SI for {identifier}: {e}")
        except ET.ParseError as e:
            self.logger.warning(f"Failed to parse XML for {identifier}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error extracting SI for {identifier}: {e}")
            
        # Deduplicate (in case tags overlap) and join into a single text block
        unique_si = list(dict.fromkeys(si_content))
        final_text = "\n\n".join(unique_si)
        
        if final_text:
            self.logger.debug(f"Successfully extracted {len(final_text)} chars of SI for {identifier}")
            
        return final_text