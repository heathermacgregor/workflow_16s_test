# workflow_16s/utils/publication_fetcher.py
# ===================================== IMPORTS ======================================= #

# Standard Library Imports
import concurrent.futures
import io
import json
import os
import re
import requests
import pdfplumber
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Third-Party Imports
from bs4 import BeautifulSoup

# Local Imports
from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.logger import get_logger

# ===================================================================================== #

logger = get_logger()

# ===================================================================================== #

class PublicationFetcher:
    """
    Finds, fetches, and analyzes publications linked to a BioProject accession.
        - Uses a tiered search across multiple academic APIs
        - Extracts full text from PDFs or webpages
        - Scans for relevant keywords and secondary citations
    """

    def __init__(self, config: AppConfig, cache_path: Optional[str] = None):
        """
        Initializes the publication fetcher with enhanced capabilities.

        Args:
            primer_db_path (str):       Path to the primer details SQLite database.
            email (str):                User email for polite API access.
            cache_path (Optional[str]): Path to the SQLite database for caching results.
        """
        self.config = config
        self.email = self.config.credentials.ena_email
        self._setup_logging()
        
        # API keys with fallback to environment variables
        self.springer_api_key = self.config.credentials.springer_api_key or os.getenv("SPRINGER_NATURE_API_KEY")
        self.ieee_api_key = self.config.credentials.ieee_api_key or os.getenv("IEEE_XPLORE_API_KEY")
        self.mendeley_api_key = self.config.credentials.mendeley_api_key or os.getenv("MENDELEY_API_KEY")
        self.dimensions_api_key = self.config.credentials.dimensions_api_key or os.getenv("DIMENSIONS_API_KEY")
        self.unpaywall_email = self.email  # Unpaywall requires email
        
        if self.springer_api_key:
            os.environ["SPRINGER_NATURE_API_KEY"] = self.springer_api_key 
        if self.ieee_api_key:
            os.environ["IEEE_XPLORE_API_KEY"] = self.ieee_api_key
        if self.mendeley_api_key:
            os.environ["MENDELEY_API_KEY"] = self.mendeley_api_key
        if self.dimensions_api_key:
            os.environ["DIMENSIONS_API_KEY"] = self.dimensions_api_key
            
        # Enhanced session with connection pooling and timeouts
        self.http_session = requests.Session()
        self.http_session.headers.update({
            "User-Agent": f"BioProjectPublicationExtractor/2.0 (mailto:{self.email})",
            "Accept": "application/json"
        })
        # Set reasonable default timeout
        self.timeout = (10, 30)  # (connect, read) timeouts
        
        self.ncbi_eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Rate limiting for different APIs
        self.rate_limits = {
            'ncbi': 0.34,  # 3 requests/sec without API key
            'crossref': 0.05,  # ~20 requests/sec is polite
            'semantic_scholar': 1.0,  # 1 request/sec
            'europe_pmc': 0.2,  # 5 requests/sec
            'unpaywall': 1.0,  # 1 request/sec to be polite
            'default': 0.5
        }
        self.last_request_times = {}
        
        self.primer_db = self._load_primer_database(self.config.paths.primer_db)

        # Store the cache path and initialize the cache table if provided.
        self.cache_path = cache_path
        if self.cache_path:
            self._create_cache_table()
            self.logger.debug(f"Publication cache enabled at: {self.cache_path}")
        
        # Stats tracking
        self.api_calls = {'total': 0, 'cached': 0, 'failed': 0}
        self.source_success = {}  # Track which sources are finding results

    def _create_cache_table(self):
        """
        Creates enhanced cache tables for storing publication results, DOI metadata, and failed lookups.
        """
        if not self.cache_path:
            return
        try:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as conn:
                # Main publication cache
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS publication_cache (
                        bioproject_id TEXT PRIMARY KEY,
                        results_json TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        source_api TEXT,
                        success_count INTEGER DEFAULT 0
                    )
                """)
                
                # DOI metadata cache to avoid redundant lookups
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS doi_metadata_cache (
                        doi TEXT PRIMARY KEY,
                        metadata_json TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        citation_count INTEGER,
                        full_text_available BOOLEAN
                    )
                """)
                
                # Failed lookup cache to avoid retrying known failures
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS failed_lookups (
                        accession TEXT PRIMARY KEY,
                        attempted_apis TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        retry_after REAL
                    )
                """)
                
                # Create indices for faster queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pub_timestamp ON publication_cache(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_doi_timestamp ON doi_metadata_cache(timestamp)")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create cache table: {e}")

    def _setup_logging(self):
        """Sets up a logger for the module."""
        self.logger = logger
    
    def _rate_limit(self, api_name: str):
        """
        Enforce rate limiting for API calls.
        
        Args:
            api_name: Name of the API (e.g., 'ncbi', 'crossref')
        """
        import time
        current = time.time()
        last_request = self.last_request_times.get(api_name, 0)
        elapsed = current - last_request
        wait_time = self.rate_limits.get(api_name, self.rate_limits['default']) - elapsed
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        self.last_request_times[api_name] = time.time()

    # --- Core Text and Data Processing Helpers ---

    def _fix_spacing_in_text(self, text: str) -> str:
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([.,;:])([a-zA-Z\d])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(-)([a-zA-Z])', r'\1 \2 \3', text)
        text = re.sub(r'([\]\)])([a-zA-Z\d\[])', r'\1 \2', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _find_methods_section(self, text: str) -> str:
        """
        Finds the materials and methods section in a publication's full text.

        This method is designed to be robust by checking for a comprehensive list of
        potential headers for the methods section and the subsequent section. It
        extracts the text between these two points.
        """
        # A list of common headers for the methods section, in lowercase
        start_headers = [
            'materials and methods', 'methods and materials', 'methods',
            'experimental procedures', 'experimental section', 'research design',
            'experimental design', 'methodology', 'study design'
        ]
        
        # A list of common headers that mark the end of the methods section
        end_headers = [
            'results', 'discussion', 'conclusions', 'acknowledgments', 'conclusion',
            'author contributions', 'references', 'supporting information',
            'data availability', 'competing interests', 'funding'
        ]

        # Find the starting position of the methods section
        start_pos = -1
        section_start_header = ""
        for header in start_headers:
            # Use regex to find header as a whole word, surrounded by whitespace/newlines
            pattern = r'\\n\s*' + re.escape(header) + r'\s*\\n'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pos = match.start()
                # Choose the earliest occurring header if multiple are found
                if start_pos == -1 or pos < start_pos:
                    start_pos = pos
                    section_start_header = match.group(0).strip()

        if start_pos == -1:
            return "Methods section not found in text."

        # Define the area to search for the end header (i.e., after the start)
        search_area = text[start_pos + len(section_start_header):]

        # Find the ending position of the methods section
        end_pos = -1
        for header in end_headers:
            pattern = r'\\n\s*' + re.escape(header) + r'\s*\\n'
            match = re.search(pattern, search_area, re.IGNORECASE)
            if match:
                pos = match.start()
                # Choose the earliest occurring end header
                if end_pos == -1 or pos < end_pos:
                    end_pos = pos
        
        # Extract the text slice
        if end_pos != -1:
            # Adjust end_pos to be relative to the start of 'search_area'
            section_text = search_area[:end_pos]
        else:
            # If no end header is found, take everything to the end
            section_text = search_area

        # Combine the header with the content and clean it up
        return (section_start_header + "\\n" + section_text.strip()).strip()

    def _isolate_reference_section(self, full_text: str) -> str:
        ref_start_pattern = re.compile(r'\b(references|bibliography|works\s+cited|literature\s+cited)\b', re.IGNORECASE)
        search_start_index = len(full_text) * 2 // 3
        ref_match = ref_start_pattern.search(full_text, pos=search_start_index)
        return full_text[ref_match.start():].strip() if ref_match else ""

    def _get_year(self, pub: Dict[str, Any]) -> int:
        try:
            return int(str(pub.get('pub_year'))[:4])
        except (TypeError, ValueError):
            return 9999

    def _deduplicate_publications(self, pub_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_keys = set()
        unique_pubs = []
        for pub in pub_list:
            doi = pub.get('doi')
            if doi and doi not in seen_keys:
                seen_keys.add(doi)
                unique_pubs.append(pub)
                continue
            title = pub.get('publication_title', "").lower().strip()
            year = pub.get('pub_year')
            fallback_key = (title, year)
            if title and year and fallback_key not in seen_keys:
                seen_keys.add(fallback_key)
                unique_pubs.append(pub)
        return unique_pubs

    def _get_pmc_links(self, doi: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Searches NCBI for a given DOI to find links to PubMed Central (PMC).
        """
        pdf_url, article_url = None, None
        try:
            converter_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            response = self.http_session.get(converter_url, params={'ids': doi, 'format': 'json', 'tool': 'publication_fetcher', 'email': self.email})
            response.raise_for_status()
            data = response.json()

            if 'records' in data and data['records']:
                pmcid = data['records'][0].get('pmcid')
                if pmcid:
                    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    params = {'db': 'pmc', 'id': pmcid, 'retmode': 'xml', 'tool': 'publication_fetcher', 'email': self.email}
                    efetch_response = self.http_session.get(efetch_url, params=params, timeout=15)
                    efetch_response.raise_for_status()
                    root = ET.fromstring(efetch_response.content)
                    pdf_link_element = root.find(f".//link[@format='pdf'][@href]")
                    if pdf_link_element is not None:
                        pdf_url = pdf_link_element.get('href')
        except (requests.RequestException, ET.ParseError, IndexError):
            # Fail silently and return None
            pass
        return pdf_url, article_url

    def _extract_text_from_webpage(self, url: str) -> Optional[str]:
        """
        Extracts the main article text from a given URL.
        """
        try:
            response = self.http_session.get(url, timeout=25)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find('article') or soup.find('main') or soup.body
            if not main_content: return None
            for tag in main_content.find_all(['script', 'style', 'header', 'footer', 'nav', 'aside', 'figure', 'figcaption', 'a']): # type: ignore
                tag.decompose()
            text = main_content.get_text(separator=' ', strip=True)
            boilerplate_patterns = [
                r"skip to main content", r"an official website of the united states government",
                r"here's how you know", r"search log in dashboard", r"publications account settings",
                r"search in pmc", r"search in pubmed", r"view in nlm catalog",
                r"add to search", r"user guide", r"permalink copy", r"pmc disclaimer",
                r"pmc copyright notice", r"the author\\(s\\)", r"find articles by",
                r"author information article notes copyright and license information"
            ]
            for pattern in boilerplate_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except requests.RequestException:
            # Fail silently and return None
            return None

    def _load_primer_database(self, path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
        """Loads primer sequences and names from the SQLite primer database."""
        primer_db = {}
        db_path = Path(path)
        if not db_path.exists():
            self.logger.warning(f"Primer database not found at '{db_path}'. Cannot perform primer validation.")
            return primer_db

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT Primer_Name, Sequence, ProbeBase_ID FROM primers")
                for row in cursor.fetchall():
                    raw_sequence = row['Sequence']
                    if not raw_sequence: continue
                    sequence = re.sub(r"5'-|'|-3'|\s", "", raw_sequence).strip().upper()
                    if sequence:
                        primer_details = {
                            "name": row['Primer_Name'],
                            "probebase_id": (row['ProbeBase_ID'] or '').split()[0] if row['ProbeBase_ID'] else ''
                        }
                        primer_db[sequence] = primer_details
            self.logger.debug(f"Loaded {len(primer_db)} primers into memory from '{db_path}'.")
        except (sqlite3.Error, Exception) as e:
            self.logger.error(f"Failed to load or parse primer SQLite database: {e}", exc_info=True)
        return primer_db

    def _validate_primers(self, found_sequences: List[str]) -> List[Dict[str, Any]]:
        validated = []
        if not self.primer_db: return validated
        for seq in found_sequences:
            match = self.primer_db.get(seq.upper())
            if match: validated.append({"sequence_found": seq, "probebase_match": match})
        return validated

    def _find_citations_near_accession(self, full_text: str, accession: str, context_chars: int = 250) -> Tuple[List[Dict[str, Any]], int]:
        normalized_text = re.sub(r'\s+', ' ', full_text)
        clean_accession = accession.replace(" ", "")
        search_text = re.sub(r'(\w)\s+(\d+)', r'\1\2', normalized_text, flags=re.IGNORECASE)
        citations_found, matches = [], list(re.finditer(re.escape(clean_accession), search_text, re.IGNORECASE))
        total_mentions = len(matches)
        for match in matches:
            context_snippet = search_text[max(0, match.start()):min(len(search_text), match.end() + context_chars)]
            search_zone = search_text[max(0, match.start()):min(len(search_text), match.end() + 40)]
            author_year_pattern = re.compile(r'\(?((?:[\w-]+\s?){1,3} et al\.?,? \d{4}|(?:[\w-]+\s?){1,3}, \d{4})\)?', re.IGNORECASE)
            author_year_matches = author_year_pattern.findall(search_zone)
            numbered_pattern = re.compile(r'[\[\(]\s*(\d+)\s*(?:[–-]\s*\d+)?(?:\s*,\s*\d+)*\s*[\]\)]')
            numbered_matches = numbered_pattern.findall(search_zone)
            clues = []
            if author_year_matches: clues.extend([f"Author-Year clue: {c.strip()}" for c in set(author_year_matches)])
            if numbered_matches: clues.extend([f"Numbered clue: {c}" for c in set(numbered_matches)])
            if clues:
                citations_found.append({"context_snippet": context_snippet.strip(), "citation_clues": clues})
        return citations_found, total_mentions

    def _find_citation_entry_by_number(self, reference_section: str, number: str) -> Optional[str]:
        try: next_number = str(int(number) + 1)
        except ValueError: return None
        current_ref_start_pattern = r'(\s*|^)(\[' + re.escape(number) + r'\]|\b' + re.escape(number) + r'\.)\s*'
        next_ref_start_pattern = r'(\s*|^)(\[' + re.escape(next_number) + r'\]|\b' + re.escape(next_number) + r'\.)\s*'
        full_entry_pattern = re.compile(current_ref_start_pattern + r'(.*?)' + r'(?=' + next_ref_start_pattern + r'|$)', re.IGNORECASE | re.DOTALL)
        match = full_entry_pattern.search(reference_section)
        if match: return f"{match.group(2).strip()} {match.group(3).strip()}"
        return None

    def _search_citation_details_via_crossref(self, title_or_author_year: str, accession: str) -> List[Dict[str, Any]]:
        publications = []
        params = {"query": title_or_author_year, "rows": 3, "mailto": self.email, "sort": "relevance"}
        try:
            response = self.http_session.get("https://api.crossref.org/works", params=params, timeout=15)
            response.raise_for_status()
            for item in response.json().get('message', {}).get('items', []):
                title = (item.get('title') or ["Unknown Title"])[0]
                date_parts = item.get('issued', {}).get('date-parts')
                year = "N/A"
                if date_parts and date_parts[0] and date_parts[0][0] is not None: year = date_parts[0][0]
                publications.append({"bioproject_accession": accession, "publication_title": title, "pub_year": str(year), "doi": item.get('DOI'), "status": "Ready (Cited)"})
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Secondary Crossref lookup failed for '{title_or_author_year}': {e}")
        return publications

    # --- Tiered API Search Functions ---
    def _get_publications_from_ncbi(self, accession: str) -> List[Dict[str, Any]]:
        """
        Fetches publications from NCBI PubMed with enhanced metadata extraction and error handling.
        
        Args:
            accession: BioProject or SRA accession ID
            
        Returns:
            List of publication dictionaries with enhanced metadata
        """
        publications = []
        
        try:
            # Rate limit NCBI requests
            self._rate_limit('ncbi')
            
            # Link BioProject to PubMed articles
            elink_params = {
                "dbfrom": "bioproject",
                "db": "pubmed",
                "id": accession,
                "retmode": "json",
                "tool": "PublicationExtractor",
                "email": self.email
            }
            
            elink_resp = self.http_session.get(
                f"{self.ncbi_eutils_base}/elink.fcgi",
                params=elink_params,
                timeout=self.timeout
            )
            elink_resp.raise_for_status()
            linksets = elink_resp.json().get('linksets', [])
            pmids = [
                link['Id']
                for ls in linksets
                for lsd in ls.get('linksetdbs', [])
                for link in lsd.get('links', [])
            ]
            
            if not pmids:
                self.logger.debug(f"No PubMed links found for {accession}")
                return []
            
            # Limit to top 5 to avoid excessive API calls
            pmids = pmids[:5]
            self.logger.debug(f"Found {len(pmids)} PMIDs for {accession}")
            
            # Rate limit again for summary fetch
            self._rate_limit('ncbi')
            
            # Fetch detailed metadata
            esummary_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "tool": "PublicationExtractor",
                "email": self.email
            }
            
            esummary_resp = self.http_session.get(
                f"{self.ncbi_eutils_base}/esummary.fcgi",
                params=esummary_params,
                timeout=self.timeout
            )
            esummary_resp.raise_for_status()
            result = esummary_resp.json()['result']
            
            # Extract enhanced metadata
            for pmid in pmids:
                if pmid not in result:
                    continue
                    
                pub_data = result[pmid]
                
                # Extract DOI
                doi = next(
                    (aid['value'] for aid in pub_data.get('articleids', [])
                     if aid.get('idtype') == 'doi'),
                    None
                )
                
                # Extract authors (limit to first 3)
                authors = pub_data.get('authors', [])
                author_list = [a.get('name', '') for a in authors[:3]]
                if len(authors) > 3:
                    author_list.append('et al.')
                authors_str = ', '.join(author_list) if author_list else 'Unknown'
                
                # Extract journal
                journal = pub_data.get('fulljournalname', pub_data.get('source', 'Unknown'))
                
                # Extract abstract (truncate if too long)
                abstract = pub_data.get('abstract', '')
                if len(abstract) > 500:
                    abstract = abstract[:497] + '...'
                
                publications.append({
                    "bioproject_accession": accession,
                    "pmid": pmid,
                    "publication_title": pub_data.get('title', 'Unknown Title'),
                    "authors": authors_str,
                    "journal": journal,
                    "pub_year": pub_data.get('pubdate', '')[:4],
                    "doi": doi,
                    "abstract": abstract,
                    "status": "Ready (NCBI)"
                })
            
            # Track success
            self.source_success['ncbi'] = self.source_success.get('ncbi', 0) + len(publications)
            self.logger.debug(f"NCBI returned {len(publications)} publications for {accession}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"NCBI API error for {accession}: {e}")
            self.api_calls['failed'] += 1
        except (KeyError, ValueError) as e:
            self.logger.error(f"NCBI response parsing error for {accession}: {e}")
            self.api_calls['failed'] += 1
        
        return publications

    def _get_publications_from_crossref(self, accession: str) -> List[Dict[str, Any]]:
        """Fetches publications from Crossref API with rate limiting."""
        publications = []
        params = {
            "query": accession,
            "rows": 10,
            "mailto": self.email,
            "sort": "published",
            "order": "asc"
        }
        
        try:
            self._rate_limit('crossref')
            response = self.http_session.get(
                "https://api.crossref.org/works",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for item in response.json().get('message', {}).get('items', []):
                title = (item.get('title') or ["Unknown Title"])[0]
                year = item.get('issued', {}).get('date-parts', [[None]])[0][0]
                publications.append({
                    "bioproject_accession": accession,
                    "publication_title": title,
                    "pub_year": str(year) if year else "N/A",
                    "doi": item.get('DOI'),
                    "status": "Ready (Crossref)"
                })
            
            self.source_success['crossref'] = self.source_success.get('crossref', 0) + len(publications)
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Crossref API error for {accession}: {e}")
            self.api_calls['failed'] += 1
        
        return publications

    def _get_publications_from_datacite(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"query": accession, "page[size]": 10, "sort": "published"}
        response = self.http_session.get("https://api.datacite.org/works", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('data', []):
            attrs = item.get('attributes', {})
            title = (attrs.get('titles', [{}])[0].get('title', "Unknown Title"))
            publications.append({"bioproject_accession": accession, "publication_title": title, "pub_year": str(attrs.get('publicationYear', 'N/A')), "doi": attrs.get('doi'), "status": "Ready (DataCite)"})
        return publications

    def _get_publications_from_semantic_scholar(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"query": accession, "fields": "title,year,externalIds", "limit": 10}
        response = self.http_session.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('data', []):
            doi = item.get('externalIds', {}).get('DOI')
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('year', 'N/A')), "doi": doi, "status": "Ready (Semantic Scholar)"})
        return publications

    def _get_publications_from_europe_pmc(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"query": accession, "resultType": "lite", "format": "json", "pageSize": 10}
        response = self.http_session.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('resultList', {}).get('result', []):
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('pubYear', 'N/A')), "doi": item.get('doi'), "status": "Ready (Europe PMC)"})
        return publications

    def _get_publications_from_plos(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"q": f'"{accession}"', "fl": "id,publication_date,title", "wt": "json", "rows": 10, "sort": "publication_date asc"}
        response = self.http_session.get("http://api.plos.org/search", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('response', {}).get('docs', []):
            year = item.get('publication_date', 'N/A')[:4]
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(year), "doi": item.get('id'), "status": "Ready (PLOS)"})
        return publications

    def _get_publications_from_springer_nature(self, accession: str) -> List[Dict[str, Any]]:
        if not self.springer_api_key: return []
        publications, params = [], {"q": f'fulltext:"{accession}"', "api_key": self.springer_api_key, "p": 10}
        response = self.http_session.get("http://api.springernature.com/openaccess/json", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('records', []):
            year = item.get('publicationDate', 'N/A')[:4]
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(year), "doi": item.get('doi'), "status": "Ready (Springer)"})
        return publications

    def _get_publications_from_base_search(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"q": accession, "format": "json", "sort": "date:asc", "limit": 10}
        response = self.http_session.get("https://api.base-search.net/v2/search", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('response', {}).get('docs', []):
            year, doi = item.get('year') or (item.get('date', 'N/A')[:4]), item.get('doi')
            if isinstance(doi, list): doi = doi[0]
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(year), "doi": doi, "status": "Ready (BASE)"})
        return publications

    def _get_publications_from_doaj(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"q": f'bibjson.abstract:"{accession}"', "sort": "created_date:asc", "pageSize": 10}
        response = self.http_session.get("https://doaj.org/api/search/articles", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('results', []):
            bibjson = item.get('bibjson', {})
            doi = next((i['id'] for i in bibjson.get('identifier', []) if i.get('type') == 'doi'), None)
            publications.append({"bioproject_accession": accession, "publication_title": bibjson.get('title', "Unknown Title"), "pub_year": str(bibjson.get('year', 'N/A')), "doi": doi, "status": "Ready (DOAJ)"})
        return publications

    def _get_publications_from_arxiv(self, accession: str) -> List[Dict[str, Any]]:
        publications, params = [], {"search_query": f'all:"{accession}"', "sortBy": "submittedDate", "sortOrder": "ascending", "max_results": 10}
        response = self.http_session.get("http://export.arxiv.org/api/query", params=params, timeout=15)
        response.raise_for_status()
        root, ns = ET.fromstring(response.content), {'a': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('a:entry', ns):
            title, year = entry.find('a:title', ns).text.strip(), entry.find('a:published', ns).text[:4] # type: ignore
            doi_link = entry.find('a:link[@title="doi"]', ns)
            doi = doi_link.attrib.get('href', '').split('doi.org/')[-1] if doi_link is not None else None
            publications.append({"bioproject_accession": accession, "publication_title": title, "pub_year": str(year), "doi": doi, "status": "Ready (ArXiv)"})
        return publications
    
    def _get_open_access_pdf_url(self, doi: str) -> Optional[str]:
        """
        Fetches open access PDF URL from Unpaywall API.
        
        Args:
            doi: DOI of the publication
            
        Returns:
            URL to open access PDF if available, None otherwise
        """
        if not doi:
            return None
        
        try:
            self._rate_limit('unpaywall')
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": self.unpaywall_email}
            
            response = self.http_session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 404:
                self.logger.debug(f"DOI not found in Unpaywall: {doi}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Check for best_oa_location which has the free PDF
            best_oa = data.get('best_oa_location')
            if best_oa and best_oa.get('url_for_pdf'):
                self.logger.debug(f"Found open access PDF for {doi}")
                return best_oa['url_for_pdf']
            
            # Fallback to any OA location
            oa_locations = data.get('oa_locations', [])
            for location in oa_locations:
                if location.get('url_for_pdf'):
                    return location['url_for_pdf']
            
            self.logger.debug(f"No open access PDF found for {doi}")
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Unpaywall API error for DOI {doi}: {e}")
            return None

    def _get_publications_from_ieee_xplore(self, accession: str) -> List[Dict[str, Any]]:
        if not self.ieee_api_key: return []
        publications, params = [], {"querytext": f'"{accession}"', "apikey": self.ieee_api_key, "max_records": 10, "sortfield": "publication_year", "sortorder": "asc"}
        response = self.http_session.get("https://ieeexploreapi.ieee.org/api/v1/search/articles", params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('articles', []):
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('publication_year', 'N/A')), "doi": item.get('doi'), "status": "Ready (IEEE)"})
        return publications

    def _get_publications_from_mendeley(self, accession: str) -> List[Dict[str, Any]]:
        if not self.mendeley_api_key: return []
        publications, headers, params = [], {"Authorization": f"Bearer {self.mendeley_api_key}"}, {"query": f'"{accession}"', "view": "all", "limit": 10, "sort": "year", "direction": "asc"}
        response = self.http_session.get("https://api.mendeley.com/catalog", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json():
            doi = item.get('identifiers', {}).get('doi')
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('year', 'N/A')), "doi": doi, "status": "Ready (Mendeley)"})
        return publications

    def _get_publications_from_core(self, accession: str) -> List[Dict[str, Any]]:
        publications, query_data = [], {"q": accession, "limit": 10, "sort": "yearPublished:asc"}
        response = self.http_session.post("https://api.core.ac.uk/v3/search/works", json=query_data, timeout=15)
        response.raise_for_status()
        for item in response.json().get('results', []):
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('yearPublished', 'N/A')), "doi": item.get('doi'), "status": "Ready (CORE)"})
        return publications

    def _get_publications_from_dimensions(self, accession: str) -> List[Dict[str, Any]]:
        if not self.dimensions_api_key: return []
        publications, query = [], f'search publications for "{accession}" return publications[title,year,doi] sort by year asc limit 10'
        response = self.http_session.post("https://app.dimensions.ai/api/dsl.json", data=query.encode('utf-8'), timeout=15)
        response.raise_for_status()
        for item in response.json().get('publications', []):
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('year', 'N/A')), "doi": item.get('doi'), "status": "Ready (Dimensions)"})
        return publications

    def _get_publications_from_biorxiv(self, accession: str) -> List[Dict[str, Any]]: return [] # Placeholder
    
    # --- METHODOLOGY EXTRACTION ---
    def _extract_methodology_details(self, text_to_scan: str) -> Dict[str, Any]:
        """
        Scans text for specific methodology details using a comprehensive set of regex patterns.
        """
        patterns = {
            'gene_mentions': r'16S\s*(?:rRNA|rDNA|gene)?',
            'variable_regions': r'\b(V[1-9](?:\s*-\s*V[1-9])?)\b',
            'primer_names': r'\b([a-zA-Z0-9_-]*?(?:515F|806R|341F|907R|Eub338|Arch915|Pro341)[a-zA-Z0-9_-]*?|[FR]\s?\d{3,})\b',
            'primer_sequences': r'\b([ACGTUNRYSWKMBDHV]{15,})\b',
            'extraction_kits': r'\b((?:DNeasy|Power(?:Soil|Fecal)|FastDNA|FastSpin|QIAamp|Mag-Bind|ZymoBIOMICS)[\w\s-]*?Kit)\b|((?:QIAGEN|Mo\s?Bio|Zymo|Promega|NEB)[\w\s-]*?(?:extraction|isolation|DNA)\s(?:Kit|System|Reagent))',
            'purification_kits': r'\b((?:QIAquick|Monarch|Wizard|ExoSAP-IT|AMPure)[\w\s-]*?(?:Kit|Reagent))\b|((?:purified with|cleaned using)\s(?:the\s)?([\w\s-]+?Kit))',
            'sequencing_instruments': r'\b(Illumina|PacBio|Pacific Biosciences|Oxford Nanopore|Thermo Fisher|Ion Torrent)[\s,()]*(MiSeq|HiSeq|NovaSeq|iSeq|Sequel|RS\s?II|MinION|GridION|PromethION|Ion\sS5|PGM)\b',
            'other_variables_measured': r'\b(pH|temperature|salinity|conductivity|dissolved oxygen|DOC|total nitrogen|phosphate|chlorophyll-a|CHL-a)\b'
        }

        extracted_info = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text_to_scan, re.IGNORECASE)
            if matches and isinstance(matches[0], tuple):
                cleaned_matches = [' '.join(part for part in m if part) for m in matches]
                matches = [m.strip() for m in cleaned_matches if m]
            if matches:
                unique_matches = sorted(list(set([re.sub(r'\s+', ' ', m).strip() for m in matches])))
                if unique_matches:
                    extracted_info[key] = unique_matches
        return extracted_info

    def _analyze_single_publication(self, pub_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Silently analyzes a single publication, returning status in the result dictionary.
        """
        doi = pub_data.get("doi")
        accession = pub_data['bioproject_accession']
        pub_data["accession_mentions_in_text"] = 0
        
        if not doi:
            pub_data["status"] = "⚠️ No DOI available."
            return pub_data, []

        full_text = None
        try:
            # --- Tier 1: Unpaywall for direct PDF ---
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
            unpaywall_resp = self.http_session.get(unpaywall_url, timeout=10)
            if unpaywall_resp.status_code == 200:
                unpaywall_data = unpaywall_resp.json()
                best_oa_location = unpaywall_data.get("best_oa_location")
                if best_oa_location and (pdf_url := best_oa_location.get("url_for_pdf")):
                    pub_data["pdf_url"] = pdf_url
                    pdf_response = self.http_session.get(pdf_url, timeout=45)
                    pdf_response.raise_for_status()
                    with pdfplumber.open(io.BytesIO(pdf_response.content)) as pdf:
                        full_text = "\\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
        except Exception:
            full_text = None

        # --- Tier 2 & 3: PubMed Central (PDF then Webpage) ---
        if not full_text:
            pmc_pdf_url, pmc_article_url = self._get_pmc_links(doi)
            if pmc_pdf_url:
                try:
                    pdf_response = self.http_session.get(pmc_pdf_url, timeout=45)
                    pdf_response.raise_for_status()
                    with pdfplumber.open(io.BytesIO(pdf_response.content)) as pdf:
                        full_text = "\\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
                    if full_text: pub_data["pdf_url"] = pmc_pdf_url
                except Exception:
                    full_text = None
            
            if not full_text and pmc_article_url:
                full_text = self._extract_text_from_webpage(pmc_article_url)

        # --- Tier 4: Publisher page via DOI link ---
        if not full_text:
            full_text = self._extract_text_from_webpage(f"https://doi.org/{doi}")

        if not full_text:
            pub_data["status"] = "❌ Failed to retrieve full text."
            return pub_data, []

        # --- Analysis on Retrieved Text ---
        full_text_clean = self._fix_spacing_in_text(full_text)
        newly_found_pubs, (secondary_citations, total_mentions) = [], self._find_citations_near_accession(full_text_clean, accession)
        pub_data["accession_mentions_in_text"] = total_mentions
        pub_data["secondary_citations_found"] = secondary_citations
        
        if secondary_citations:
            reference_section = self._isolate_reference_section(full_text_clean)
            for citation_info in secondary_citations:
                for clue in citation_info['citation_clues']:
                    search_term = None
                    if clue.startswith("Author-Year clue:"): search_term = clue.replace("Author-Year clue:", "").strip()
                    elif clue.startswith("Numbered clue:") and reference_section:
                        if number_match := re.search(r'\d+', clue):
                            search_term = self._find_citation_entry_by_number(reference_section, number_match.group(0))
                    if search_term: newly_found_pubs.extend(self._search_citation_details_via_crossref(search_term, accession))
        
        methods_text = self._find_methods_section(full_text_clean)
        text_to_scan = methods_text if "not found" not in methods_text else full_text_clean
        pub_data["materials_and_methods_section_found"] = "not found" not in methods_text
        
        methodology_details = self._extract_methodology_details(text_to_scan)
        pub_data["methodology_details"] = methodology_details

        if self.primer_db and "primer_sequences" in methodology_details:
            validated_primers = self._validate_primers(methodology_details["primer_sequences"])
            if validated_primers:
                pub_data["methodology_details"]["validated_primers"] = validated_primers

        pub_data["status"] = "✅ Extraction complete."
        return pub_data, newly_found_pubs

    def extract_bioproject_sequencing_info(self, bioproject_accession: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        clean_accession = bioproject_accession.strip()
        
        if use_cache and self.cache_path:
            try:
                with sqlite3.connect(self.cache_path) as conn:
                    cursor = conn.execute("SELECT results_json FROM publication_cache WHERE bioproject_id = ?", (clean_accession,))
                    if row := cursor.fetchone():
                        self.logger.info(f"✅ Found and returned cached results for '{clean_accession}'.")
                        return json.loads(row[0])
            except sqlite3.Error as e: 
                self.logger.error(f"Cache read failed for '{clean_accession}': {e}")
        
        self.logger.info(f"🔍  Starting publication search for BioProject: {clean_accession}")
        
        tier_functions = [
            self._get_publications_from_ncbi, self._get_publications_from_crossref, self._get_publications_from_datacite,
            self._get_publications_from_semantic_scholar, self._get_publications_from_europe_pmc, self._get_publications_from_plos,
            self._get_publications_from_springer_nature, self._get_publications_from_base_search, self._get_publications_from_doaj,
            self._get_publications_from_arxiv, self._get_publications_from_ieee_xplore, self._get_publications_from_mendeley,
            self._get_publications_from_core, self._get_publications_from_dimensions, self._get_publications_from_biorxiv
        ]
        
        initial_publications, failed_tiers = [], []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tier_functions)) as executor:
            future_to_tier = {
                executor.submit(func, clean_accession): func.__name__.replace("_get_publications_from_", "").replace("_", " ").title()
                for func in tier_functions
            }
            for future in concurrent.futures.as_completed(future_to_tier):
                tier_name = future_to_tier[future]
                try:
                    initial_publications.extend(future.result())
                except Exception as e:
                    failed_tiers.append(f"      - '{tier_name}' \n          [ERROR] {type(e).__name__} - {e}")

        unique_initial_pubs = self._deduplicate_publications(initial_publications)
        unique_initial_pubs.sort(key=self._get_year)
        
        search_summary = [f"\n  ↪ Publication search for '{clean_accession}' complete."]
        search_summary.append(f"    - Final Result: Found {len(unique_initial_pubs)} unique publications to analyze.")
        if failed_tiers:
            search_summary.append("    - Worker Status: The following workers failed during search:")
            search_summary.extend(sorted(failed_tiers))
        else:
            search_summary.append("    - Worker Status: All workers completed successfully.")
        self.logger.info("\n".join(search_summary))
        
        if not initial_publications: return []
        
        processed_dois, publications_queue = set(), []
        for pub in unique_initial_pubs:
            if doi := pub.get('doi'):
                publications_queue.append(pub)
                processed_dois.add(doi)
        
        all_final_results, round_count = [], 0
        while publications_queue and round_count < 3:
            round_count += 1
            newly_discovered_this_round, pubs_for_this_round = [], publications_queue[:]
            publications_queue.clear()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_result = {executor.submit(self._analyze_single_publication, pub): pub for pub in pubs_for_this_round}
                for future in concurrent.futures.as_completed(future_to_result):
                    try:
                        result, secondary_pubs = future.result()
                        all_final_results.append(result)
                        if secondary_pubs: newly_discovered_this_round.extend(secondary_pubs)
                    except Exception as e: 
                        original_pub = future_to_result[future]
                        self.logger.error(f"Critical error analyzing DOI {original_pub.get('doi')}: {e}", exc_info=True)

            if newly_discovered_this_round:
                unique_secondary = self._deduplicate_publications(newly_discovered_this_round)
                for pub in unique_secondary:
                    if (doi := pub.get('doi')) and doi not in processed_dois:
                        publications_queue.append(pub)
                        processed_dois.add(doi)
        
        all_final_results.sort(key=self._get_year)

        # --- MODIFICATION: FINAL ANALYSIS SUMMARY ---
        analysis_summary = [f"\n  ↪ Analysis of {len(all_final_results)} publication(s) complete."]
        success_count = 0
        failed_pubs = []
        for result in all_final_results:
            if result.get("status") == "✅ Extraction complete.":
                success_count += 1
            else:
                doi = result.get('doi', 'N/A')
                title = result.get('publication_title', 'Unknown Title')
                reason = result.get('status', 'Unknown Error')
                failed_pubs.append(f"        - DOI: {doi} ('{title[:40]}...') failed: {reason}")
        
        analysis_summary.append(f"    - ✅ Success: {success_count}")
        analysis_summary.append(f"    - ❌ Failures: {len(failed_pubs)}")
        if failed_pubs:
            analysis_summary.extend(failed_pubs)
        self.logger.info("\n".join(analysis_summary))
        
        if use_cache and self.cache_path:
            try:
                results_str = json.dumps(all_final_results)
                with sqlite3.connect(self.cache_path) as conn:
                    conn.execute("INSERT OR REPLACE INTO publication_cache (bioproject_id, results_json) VALUES (?, ?)", (clean_accession, results_str))
                self.logger.debug(f"Cached publication results for '{clean_accession}'.")
            except sqlite3.Error as e: 
                self.logger.error(f"Cache write failed for '{clean_accession}': {e}")
        
        return all_final_results