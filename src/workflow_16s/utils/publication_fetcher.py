# workflow_16s/utils/publication_fetcher.py

import xml.etree.ElementTree as ET
import concurrent.futures
import io
import json
import os
import time
import random
import re
import requests
import pdfplumber
import sqlite3
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import pandas as pd
from bs4 import BeautifulSoup

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger


def with_ncbi_backoff(max_retries=5, base_delay=1.0, max_delay=32.0):
    """
    Decorator that applies exponential backoff with jitter specifically for HTTP 429 errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    # Attempt to execute the NCBI request
                    return func(*args, **kwargs)
                
                except requests.exceptions.HTTPError as e:
                    # Check if the error is specifically a 429 Too Many Requests
                    if e.response is not None and e.response.status_code == 429:
                        if retries >= max_retries:
                            # If we've exhausted our retries, finally give up and crash/log
                            raise e
                        
                        # Calculate exponential sleep: 1s, 2s, 4s, 8s, 16s... up to max_delay
                        sleep_time = min(max_delay, base_delay * (2 ** retries))
                        
                        # Add up to 10% randomness (jitter) to prevent thread pile-ups
                        jitter = random.uniform(0, 0.1 * sleep_time)
                        total_sleep = sleep_time + jitter
                        
                        print(f"[WARNING] NCBI 429 Rate Limit Hit. Retrying in {total_sleep:.2f}s... (Attempt {retries + 1}/{max_retries})")
                        time.sleep(total_sleep)
                        retries += 1
                    else:
                        # If it's a 404, 500, etc., don't retry—just raise the error immediately
                        raise e
        return wrapper
    return decorator

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
        # STANDARDIZED: Config first, then environment variable fallback
        self.springer_api_key = self.config.credentials.springer_api_key or os.getenv("SPRINGER_NATURE_API_KEY")
        self.ieee_api_key = self.config.credentials.ieee_api_key or os.getenv("IEEE_XPLORE_API_KEY")
        self.mendeley_api_key = self.config.credentials.mendeley_api_key or os.getenv("MENDELEY_API_KEY")
        self.dimensions_api_key = self.config.credentials.dimensions_api_key or os.getenv("DIMENSIONS_API_KEY")
        self.llm_api_key = getattr(self.config.credentials, 'llm_api_key', None) or os.getenv("LLM_API_KEY")
        self.unpaywall_email = self.email  # Unpaywall requires email

        # REMOVED: os.environ writes (security risk, creates confusion)
        # Credentials are now accessed directly from self.* attributes
            
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
        self.logger = get_logger("workflow_16s")
    
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
        
    def _build_smart_queries(self, ena_metadata: pd.DataFrame) -> List[str]:
        """Generates fuzzy/Boolean queries from metadata using dynamic column matching."""
        if ena_metadata is None or ena_metadata.empty:
            return []

        smart_queries = []
        # Expanded stopwords to keep the query focused on unique biological/location concepts
        stopwords = {
            "the", "and", "of", "to", "in", "a", "is", "for", "from", "with", "by", "on", 
            "as", "an", "this", "that", "at", "16s", "rrna", "amplicon", "sequencing", 
            "microbiome", "microbiota", "community", "analysis", "data", "study", 
            "samples", "using", "bacterial", "bacterium", "bacteria", "based", "high", 
            "throughput", "environmental", "project", "gene", "diversity"
        }

        # 1. Dynamically find Author/Center columns
        author_keywords = ['center', 'broker', 'investigator', 'author', 'submitter', 'institute']
        author_cols = [col for col in ena_metadata.columns if any(k in col.lower() for k in author_keywords)]
        
        authors = []
        for col in author_cols:
            vals = ena_metadata[col].dropna().unique()
            if len(vals) > 0:  # <-- Safe check for numpy arrays and lists!
                first_val_str = str(vals[0]).strip()
                if first_val_str and first_val_str.lower() != 'nan':
                    authors.append(first_val_str.split()[0])

        # 2. Dynamically find Title/Description columns
        text_keywords = ['title', 'description', 'abstract', 'summary', 'objective', 'name']
        text_cols = [col for col in ena_metadata.columns if any(k in col.lower() for k in text_keywords)]
        
        text_corpus = ""
        for col in text_cols:
            # Combine all unique text from these columns
            text_corpus += " " + " ".join(str(v) for v in ena_metadata[col].dropna().unique())

        if text_corpus.strip():
            # Clean and split the text into words (require at least 5 chars to filter out junk)
            words = re.findall(r'\b[a-zA-Z]{5,}\b', text_corpus.lower())
            
            # Count frequencies to find the most "defining" words for this specific dataset
            from collections import Counter
            word_counts = Counter([w for w in words if w not in stopwords])
            
            # Keep the top 4 most frequent, meaningful words
            keywords = [word for word, count in word_counts.most_common(4)]
            
            if keywords:
                # Build an AND query: (keyword1 AND keyword2 AND keyword3)
                base_query = " AND ".join(keywords)
                
                # Pair it with the author/center if we found one
                if authors:
                    smart_queries.append(f"({authors[0]}) AND ({base_query})")
                else:
                    smart_queries.append(f"({base_query})")

        return smart_queries

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
            #pattern = r'\\n\s*' + re.escape(header) + r'\s*\\n'
            pattern = r'\n\s*' + re.escape(header) + r'\s*\n'
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
            #pattern = r'\\n\s*' + re.escape(header) + r'\s*\\n'
            pattern = r'\n\s*' + re.escape(header) + r'\s*\n'
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
        #ref_match = ref_start_pattern.search(full_text, pos=search_start_index)
        ref_match = ref_start_pattern.search(full_text, search_start_index)
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
    
    @with_ncbi_backoff(max_retries=5)
    def _get_pmc_links(self, doi: str, session: Optional[requests.Session] = None) -> Tuple[Optional[str], Optional[str]]:
        """Searches NCBI for a given DOI to find links to PubMed Central (PMC) safely within a thread."""
        pdf_url, article_url = None, None
        req_method = session.get if session else requests.get
        
        try:
            converter_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            params = {'ids': doi, 'format': 'json', 'tool': 'publication_fetcher', 'email': self.email}
            response = req_method(converter_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'records' in data and data['records']:
                pmcid = data['records'][0].get('pmcid')
                if pmcid:
                    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    efetch_params = {'db': 'pmc', 'id': pmcid, 'retmode': 'xml', 'tool': 'publication_fetcher', 'email': self.email}
                    
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
    
    @with_ncbi_backoff(max_retries=5)
    def _extract_text_from_webpage(self, url: str, session: Optional[requests.Session] = None) -> Optional[str]:
        """Extracts the main article text from a given URL safely within a thread."""
        req_method = session.get if session else requests.get
        
        try:
            response = req_method(url, timeout=25)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find('article') or soup.find('main') or soup.body
            
            if not main_content: 
                return None
                
            for tag in main_content.find_all(['script', 'style', 'header', 'footer', 'nav', 'aside', 'figure', 'figcaption', 'a']): 
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

    def _verify_against_source(self, extracted_items: List[str], source_text: str, is_dna: bool = False) -> List[str]:
        """
        Anti-hallucination shield: Verifies that extracted items actually exist in the source text.
        """
        if not extracted_items or not source_text:
            return []

        verified_items = []
        
        # Normalize the source text for standard text comparison
        norm_source = re.sub(r'\s+', ' ', source_text).lower()
        
        # Strip all formatting from source text for pure DNA sequence matching
        if is_dna:
            dna_source = re.sub(r'[^a-zA-Z]', '', source_text).upper()

        for item in extracted_items:
            # 1. Verification for DNA Sequences (Primers)
            if is_dna:
                clean_seq = re.sub(r'[^a-zA-Z]', '', item).upper()
                # Must be at least 10 bases to be considered a valid primer sequence to check
                if len(clean_seq) >= 10 and clean_seq in dna_source:
                    verified_items.append(clean_seq)
                else:
                    self.logger.warning(f"Hallucination caught! Dropped primer '{item}' (Not found in source).")
            
            # 2. Verification for Text (Kits, Regions, Models)
            else:
                norm_item = re.sub(r'\s+', ' ', str(item)).lower()
                # We require the extracted item (or a highly similar subset of it) to be in the text
                if norm_item in norm_source:
                    verified_items.append(item)
                else:
                    # Try a fuzzy fallback: Check if the longest word of the item is in the text
                    words = [w for w in norm_item.split() if len(w) > 4]
                    if words and any(w in norm_source for w in words):
                        verified_items.append(item) # Partial match accepted
                    else:
                        self.logger.warning(f"Hallucination caught! Dropped text '{item}' (Not found in source).")

        return verified_items
    
    def _extract_methodology_details_llm(self, text_to_scan: str) -> Dict[str, Any]:
        """
        Extracts comprehensive methodology details using an LLM.
        Strictly enforces JSON schema and runs an anti-hallucination verification pass.
        """
        if not self.llm_api_key:
            return self._extract_methodology_details(text_to_scan)
            
        text_chunk = text_to_scan[:20000] 
        
        # 1. UPGRADED PROMPT: Explicitly forbid guessing
        system_prompt = (
            "You are an expert bioinformatician data extractor. Read the provided materials and methods text "
            "and extract the experimental details. Return ONLY a raw JSON object with absolutely no markdown formatting. "
            "CRITICAL: DO NOT GUESS OR INFER. EXTRACT EXACT STRINGS FROM THE TEXT. IF IT IS NOT EXPLICITLY WRITTEN, RETURN AN EMPTY LIST.\n"
            "The JSON must contain exactly these keys:\n"
            "- 'sample_storage' (list of strings)\n"
            "- 'extraction_protocol_and_kits' (list of strings)\n"
            "- 'pcr_conditions_and_kits' (list of strings)\n"
            "- 'primer_names' (list of strings)\n"
            "- 'primer_sequences' (list of strings)\n"
            "- 'variable_regions' (list of strings)\n"
            "- 'sequencing_details' (list of strings)\n"
            "- 'unextracted_flag' (boolean): Set to true ONLY if the text explicitly references methodology "
            "(e.g., 'primers are listed in Table S1') that is MISSING from this text.\n"
            "- 'unextracted_reason' (string): If unextracted_flag is true, briefly explain what is referenced. Else, empty string."
        )
        
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama-3.1-70b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to analyze:\n{text_chunk}"}
            ],
            "temperature": 0.0 # Force maximum determinism
        }
        
        try:
            response = requests.post(
                "https://models.inference.ai.azure.com/chat/completions",
                headers=headers, 
                json=payload, 
                timeout=45
            )
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            content = content.replace("```json", "").replace("```", "").strip()
            llm_results = json.loads(content)
            
            # --- 2. ANTI-HALLUCINATION SHIELD ---
            # Define which keys get which validation treatment
            text_keys = ['sample_storage', 'extraction_protocol_and_kits', 'pcr_conditions_and_kits', 
                         'primer_names', 'variable_regions', 'sequencing_details']
                         
            for k in text_keys:
                if k in llm_results and isinstance(llm_results[k], list):
                    # Verify text items
                    llm_results[k] = self._verify_against_source(llm_results[k], text_chunk, is_dna=False)
                else:
                    llm_results[k] = []
                    
            if 'primer_sequences' in llm_results and isinstance(llm_results['primer_sequences'], list):
                # Verify DNA sequences strictly
                llm_results['primer_sequences'] = self._verify_against_source(llm_results['primer_sequences'], text_chunk, is_dna=True)
            else:
                llm_results['primer_sequences'] = []
            # ------------------------------------
                    
            if 'unextracted_flag' not in llm_results:
                llm_results['unextracted_flag'] = False
            if 'unextracted_reason' not in llm_results:
                llm_results['unextracted_reason'] = ""
                    
            return llm_results
            
        except Exception as e:
            self.logger.debug(f"LLM extraction failed ({type(e).__name__}: {e}). Falling back to regex.")
            return self._extract_methodology_details(text_to_scan)
        
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
    
    @with_ncbi_backoff(max_retries=5)
    def _search_citation_details_via_crossref(
        self, 
        title_or_author_year: str, 
        accession: str, 
        session: Optional[requests.Session] = None
    ) -> List[Dict[str, Any]]:
        
        publications = []
        params = {"query": title_or_author_year, "rows": 3, "mailto": self.email, "sort": "relevance"}
        
        # Use the passed session, or fallback to a standard requests.get if none provided
        req_method = session.get if session else requests.get
        
        try:
            response = req_method("https://api.crossref.org/works", params=params, timeout=15)
            response.raise_for_status()
            
            for item in response.json().get('message', {}).get('items', []):
                title = (item.get('title') or ["Unknown Title"])[0]
                date_parts = item.get('issued', {}).get('date-parts')
                year = "N/A"
                if date_parts and date_parts[0] and date_parts[0][0] is not None: 
                    year = date_parts[0][0]
                    
                publications.append({
                    "bioproject_accession": accession, 
                    "publication_title": title, 
                    "pub_year": str(year), 
                    "doi": item.get('DOI'), 
                    "status": "Ready (Cited)"
                })
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Secondary Crossref lookup failed for '{title_or_author_year}': {e}")
            
        return publications

    @with_ncbi_backoff(max_retries=5)
    def _make_ncbi_request(self, endpoint: str, params: dict) -> dict:
        """
        Helper function to make a single NCBI API request with built-in rate-limit retries.
        """
        # 1. Enforce our baseline wait time
        self._rate_limit('ncbi') 
        
        # 2. Make the HTTP request
        url = f"{self.ncbi_eutils_base}/{endpoint}"
        response = self.http_session.get(url, params=params, timeout=self.timeout)
        
        # 3. Raise an error if it's a 429 (this triggers the decorator to retry!)
        response.raise_for_status() 
        
        # 4. Return the parsed JSON if successful
        return response.json()
    
    # --- Tiered API Search Functions ---
    def _get_publications_from_ncbi(self, accession: str) -> List[Dict[str, Any]]:
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
            esearch_data = self._make_ncbi_request("esearch.fcgi", esearch_params)
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
            elink_data = self._make_ncbi_request("elink.fcgi", elink_params)
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
            esummary_data = self._make_ncbi_request("esummary.fcgi", esummary_params)
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
                date_parts = item.get('issued', {}).get('date-parts', [])
                year = date_parts[0][0] if date_parts and date_parts[0] else None
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
            titles_list = attrs.get('titles', [])
            title = titles_list[0].get('title', "Unknown Title") if titles_list else "Unknown Title"
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
            if isinstance(doi, list): 
                doi = doi[0] if doi else None
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
        Regex fallback matching the new expanded LLM schema.
        """
        patterns = {
            'variable_regions': r'\b(V[1-9](?:\s*-\s*V[1-9])?)\b',
            'primer_names': r'\b([a-zA-Z0-9_-]*?(?:515F|806R|341F|907R|Eub338|Arch915|Pro341)[a-zA-Z0-9_-]*?|[FR]\s?\d{3,})\b',
            'primer_sequences': r'\b([ACGTUNRYSWKMBDHV]{15,})\b',
            'extraction_protocol_and_kits': r'\b((?:DNeasy|Power(?:Soil|Fecal)|FastDNA|FastSpin|QIAamp|Mag-Bind|ZymoBIOMICS)[\w\s-]*?Kit)\b|((?:QIAGEN|Mo\s?Bio|Zymo|Promega|NEB)[\w\s-]*?(?:extraction|isolation|DNA)\s(?:Kit|System|Reagent))|\b(bead[-\s]beating|CTAB|phenol[-\s]chloroform)\b',
            'pcr_conditions_and_kits': r'\b((?:Q5|Taq|Phusion|KAPA)[\w\s-]*?(?:Polymerase|Master Mix))\b|(\d{2}\s*cycles|\d{2}\s*°C\s*for\s*\d+\s*(?:s|min))',
            'sequencing_details': r'\b(Illumina|PacBio|Pacific Biosciences|Oxford Nanopore|Thermo Fisher|Ion Torrent)[\s,()]*(MiSeq|HiSeq|NovaSeq|iSeq|Sequel|RS\s?II|MinION|GridION|PromethION|Ion\sS5|PGM)\b',
            'sample_storage': r'\b(-20\s*°?C|-80\s*°?C|liquid nitrogen|RNAlater|DNA/RNA Shield)\b'
        }

        extracted_info = {
            "unextracted_flag": False, # Regex is not smart enough to know what it missed
            "unextracted_reason": ""
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text_to_scan, re.IGNORECASE)
            if matches and isinstance(matches[0], tuple):
                cleaned_matches = [' '.join(part for part in m if part) for m in matches]
                matches = [m.strip() for m in cleaned_matches if m]
            if matches:
                unique_matches = sorted(list(set([re.sub(r'\s+', ' ', m).strip() for m in matches])))
                extracted_info[key] = unique_matches if unique_matches else []
            else:
                extracted_info[key] = []
                
        return extracted_info
    
    def _fetch_si_text(self, id_1: str, id_2: str = "", *args, **kwargs) -> str:
        """
        Attempts to fetch Supplementary Information (SI) using the Europe PMC REST API.
        Accepts multiple identifiers (e.g., PMID and DOI) and automatically uses the best one.
        """
        si_content = []
        # Grab the thread-safe session if provided, otherwise fallback to the default
        request_session = kwargs.get('session', self.http_session)
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
    
    def _analyze_single_publication(self, pub_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Silently analyzes a single publication, returning status in the result dictionary."""
        doi = pub_data.get("doi")
        accession = pub_data['bioproject_accession']
        pub_data["accession_mentions_in_text"] = 0
        
        if not doi:
            pub_data["status"] = "⚠️ No DOI available."
            return pub_data, []
        pmc_article_url = None
        full_text = None
        MAX_PDF_PAGES = 40       # Limit extraction to prevent CPU/memory spikes
        MAX_FILE_SIZE = 15000000 # 15 MB limit for PDFs
        
        # 1. FIX: Use a thread-local session to prevent "Closed Session" errors
        with requests.Session() as local_session:
            local_session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://scholar.google.com/", # Makes it look like we clicked a link from Google Scholar
                "Connection": "keep-alive"
            })
            
            # Helper function for safe PDF downloading and parsing
            def _safely_extract_pdf(pdf_url: str) -> str | None: # Note: removed 'self' if defined as an inner function
                try:
                    # 1. Make the request FIRST using stream=True
                    with local_session.get(pdf_url, stream=True, timeout=30) as resp:
                        
                        # 2. Check for 403/401 errors
                        if resp.status_code in [403, 401]:
                            self.logger.debug(f"Access denied (HTTP {resp.status_code}) for {resp.url}. Publisher is blocking us.")
                            return None
                            
                        resp.raise_for_status() # Catch other HTTP errors
                        
                        # 3. Check file size
                        content_length = int(resp.headers.get('Content-Length', 0))
                        if content_length > MAX_FILE_SIZE:
                            self.logger.debug(f"Skipping PDF {pdf_url}: File too large ({content_length} bytes)")
                            return None
                        
                        # 4. Download the actual content now that checks passed
                        pdf_content = resp.content
                        
                        # 5. Check if it's actually a PDF by looking at the first few bytes
                        # Real PDFs always start with '%PDF-'
                        if not pdf_content.startswith(b"%PDF-"):
                            self.logger.debug(f"URL {resp.url} returned HTML/text instead of a PDF. Skipping.")
                            return None
                            
                        # 6. Extract text with a strict page limit
                        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                            pages_to_extract = pdf.pages[:MAX_PDF_PAGES]
                            return "\\n".join(p.extract_text() for p in pages_to_extract if p.extract_text())
                
                except Exception as e:
                    self.logger.debug(f"Failed to extract PDF from {pdf_url}: {type(e).__name__} - {e}")
                    return None
                
            def fetch_and_parse_pdf(self, url):
                # 1. Dress up like a real browser
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/pdf"
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                
                # Check if the request was successful
                if response.status_code != 200:
                    self.logger.debug(f"Failed to fetch {url} - Status Code: {response.status_code}")
                    return None

                # 2. Verify we actually got a PDF back, not an HTML error page
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type:
                    self.logger.debug(f"Server returned {content_type} instead of a PDF for {url}. Skipping.")
                    return None

                # Now it is safe to pass response.content to your PDF parser!
                return self._safely_extract_pdf(response.content)
            
            # --- Tier 1: Unpaywall for direct PDF ---
            try:
                unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
                unpaywall_resp = local_session.get(unpaywall_url, timeout=10)
                if unpaywall_resp.status_code == 200:
                    best_oa = unpaywall_resp.json().get("best_oa_location")
                    if best_oa and (pdf_url := best_oa.get("url_for_pdf")):
                        pub_data["pdf_url"] = pdf_url
                        full_text = self.fetch_and_parse_pdf(pdf_url)
            except Exception as e:
                self.logger.debug(f"Unpaywall check failed for {doi}: {e}")

            # --- Tier 2 & 3: PubMed Central (PDF then Webpage) ---
            if not full_text:
                pmc_pdf_url, pmc_article_url = self._get_pmc_links(doi, session=local_session) # Ensure this helper doesn't use the old shared session!
                if pmc_pdf_url:
                    full_text = self.fetch_and_parse_pdf(pmc_pdf_url)
                    if full_text: pub_data["pdf_url"] = pmc_pdf_url
                
                if not full_text and pmc_article_url:
                    full_text = self._extract_text_from_webpage(pmc_article_url, session=local_session) # Ensure this uses local_session if possible

            # --- Tier 4: Publisher page via DOI link ---
            if not full_text:
                full_text = self._extract_text_from_webpage(f"https://doi.org/{doi}")

            # --- End of Local Session ---

            if not full_text:
                pub_data["status"] = "❌ Failed to retrieve full text."
                return pub_data, []

            # --- Analysis on Retrieved Text ---
            full_text_clean = self._fix_spacing_in_text(full_text)
            newly_found_pubs = []
            secondary_citations, total_mentions = self._find_citations_near_accession(full_text_clean, accession)
            
            pub_data["accession_mentions_in_text"] = total_mentions
            pub_data["secondary_citations_found"] = secondary_citations
            
            if secondary_citations:
                reference_section = self._isolate_reference_section(full_text_clean)
                for citation_info in secondary_citations:
                    for clue in citation_info['citation_clues']:
                        search_term = None
                        if clue.startswith("Author-Year clue:"): 
                            search_term = clue.replace("Author-Year clue:", "").strip()
                        elif clue.startswith("Numbered clue:") and reference_section:
                            if number_match := re.search(r'\d+', clue):
                                search_term = self._find_citation_entry_by_number(reference_section, number_match.group(0))
                        
                        if search_term: 
                            # Warning: Ensure this helper isn't using an async/closed session!
                            newly_found_pubs.extend(self._search_citation_details_via_crossref(search_term, accession, session=local_session))
            
            # --- SMART THREE-PASS EXTRACTION ---
            methods_text = self._find_methods_section(full_text_clean)
            has_methods = "not found" not in methods_text
            pub_data["materials_and_methods_section_found"] = has_methods
            
            # Pass 1: Try the isolated methods section first
            text_to_scan = methods_text if has_methods else full_text_clean
            methodology_details = self._extract_methodology_details_llm(text_to_scan)

            has_primers = bool(methodology_details.get("primer_sequences"))
            has_regions = bool(methodology_details.get("variable_regions"))
            is_flagged = methodology_details.get("unextracted_flag", False)

            # Pass 2: Fallback to scanning the FULL text if missing info OR if flagged
            if has_methods and (not (has_primers and has_regions) or is_flagged):
                self.logger.debug(f"Info missing or flagged in methods for DOI {doi}. Scanning full text.")
                
                full_text_details = self._extract_methodology_details_llm(full_text_clean)
                
                # Merge findings
                for key, val in full_text_details.items():
                    if val and not methodology_details.get(key):
                        methodology_details[key] = val
                        
                # Overwrite the flag with the full-text assessment
                methodology_details["unextracted_flag"] = full_text_details.get("unextracted_flag", False)
                methodology_details["unextracted_reason"] = full_text_details.get("unextracted_reason", "")

                # Update our check variables
                has_primers = bool(methodology_details.get("primer_sequences"))
                has_regions = bool(methodology_details.get("variable_regions"))
                is_flagged = methodology_details.get("unextracted_flag", False)

            # Pass 3: The Supplementary Information (SI) Hail Mary
            if (not (has_primers and has_regions) or is_flagged) and pmc_article_url:
                self.logger.info(f"Critical info missing or flagged for SI. Hunting PDFs/Excel at {pmc_article_url}...")
                
                si_text = self._fetch_si_text(doi, session=local_session)
                
                if si_text.strip():
                    si_details = self._extract_methodology_details_llm(si_text)
                    for key, val in si_details.items():
                        if val and not methodology_details.get(key):
                            methodology_details[key] = val
                            
                    # We reached the end of the line, just accept whatever flag status the SI gave us
                    methodology_details["unextracted_flag"] = si_details.get("unextracted_flag", False)
                    methodology_details["unextracted_reason"] = si_details.get("unextracted_reason", "")

            pub_data["methodology_details"] = methodology_details
            # -----------------------------------

            if self.primer_db and "primer_sequences" in methodology_details:
                validated_primers = self._validate_primers(methodology_details["primer_sequences"])
                if validated_primers:
                    pub_data["methodology_details"]["validated_primers"] = validated_primers

            pub_data["status"] = "✅ Extraction complete."
            return pub_data, newly_found_pubs

    def extract_bioproject_sequencing_info(
        self, 
        bioproject_accession: str, 
        ena_metadata: Optional[pd.DataFrame] = None,
        use_cache: bool = True,
        progress_obj: Any = None
    ) -> List[Dict[str, Any]]:
        """Finds and analyzes publications, plumbed for dashboard safety."""
        clean_accession = bioproject_accession.strip()
        
        # 1. Cache Check
        if use_cache and self.cache_path:
            try:
                with sqlite3.connect(self.cache_path) as conn:
                    cursor = conn.execute("SELECT results_json FROM publication_cache WHERE bioproject_id = ?", (clean_accession,))
                    if row := cursor.fetchone():
                        self.logger.info(f"✅ Found and returned cached results for '{clean_accession}'.")
                        return json.loads(row[0])
            except sqlite3.Error as e: 
                self.logger.error(f"Cache read failed for '{clean_accession}': {e}")
        
        self.logger.info(f"🔍 Starting publication search for BioProject: {clean_accession}")
        
        # --- BUILD ENRICHED QUERY LIST ---
        search_queries = [
            clean_accession, 
            f'DATA:"{clean_accession}"' # Targets Europe PMC's specific Data/SI index
        ]
        
        if ena_metadata is not None and not ena_metadata.empty:
            # 1. Add Secondary Accessions
            if 'study_accession' in ena_metadata.columns:
                study_accs = ena_metadata['study_accession'].dropna().unique()
                search_queries.extend([str(acc) for acc in study_accs if acc != clean_accession])
            
            # 2. Add Exact Titles
            if 'study_title' in ena_metadata.columns:
                titles = ena_metadata['study_title'].dropna().unique()
                if len(titles) > 0 and len(str(titles[0])) > 15:
                    search_queries.append(f'"{titles[0]}"')

            # 3. Add our new "Smart/Fuzzy" Queries
            search_queries.extend(self._build_smart_queries(ena_metadata))

        unique_queries = list(dict.fromkeys(search_queries))
        self.logger.info(f"Using enriched search queries: {unique_queries}")
        # ---------------------------------
        
        tier_functions = [
            self._get_publications_from_ncbi, self._get_publications_from_crossref, self._get_publications_from_datacite,
            self._get_publications_from_semantic_scholar, self._get_publications_from_europe_pmc, self._get_publications_from_plos,
            self._get_publications_from_springer_nature, self._get_publications_from_base_search, self._get_publications_from_doaj,
            self._get_publications_from_arxiv, self._get_publications_from_ieee_xplore, self._get_publications_from_mendeley,
            self._get_publications_from_core, self._get_publications_from_dimensions, self._get_publications_from_biorxiv
        ]
        
        initial_publications, failed_tiers = [], []
        
        # --- STAGE 1: SEARCHING ---
        # Only add a task if progress_obj is EXPLICITLY provided
        task_id = None
        if progress_obj:
            task_id = progress_obj.add_task(f"[cyan]Pubs: Searching tiers for {clean_accession}...", total=len(tier_functions))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tier_functions)) as executor:
            future_to_tier = {executor.submit(func, clean_accession): func.__name__ for func in tier_functions}
            for future in concurrent.futures.as_completed(future_to_tier):
                try:
                    initial_publications.extend(future.result())
                except Exception as e:
                    failed_tiers.append(f"Tier {future_to_tier[future]} failed: {e}")
                finally:
                    if progress_obj and task_id: 
                        progress_obj.update(task_id, advance=1)

        # Cleanup Stage 1 Bar
        if progress_obj and task_id: 
            progress_obj.remove_task(task_id)

        unique_initial_pubs = self._deduplicate_publications(initial_publications)
        if not unique_initial_pubs: return []
        
        # --- STAGE 2: ANALYZING ---
        # Back where you define the queue:
        processed_dois = {pub.get('doi') for pub in unique_initial_pubs if pub.get('doi')}
        publications_queue = unique_initial_pubs[:]
        all_final_results, round_count = [], 0
        
        while publications_queue and round_count < 3:
            round_count += 1
            pubs_for_this_round = publications_queue[:]
            publications_queue.clear()
            
            round_task = None
            if progress_obj:
                task_desc = f"[magenta]Pubs: Analyzing Round {round_count} ({len(pubs_for_this_round)} items)"
                round_task = progress_obj.add_task(task_desc, total=len(pubs_for_this_round))
            
            newly_discovered = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_result = {executor.submit(self._analyze_single_publication, pub): pub for pub in pubs_for_this_round}
                for future in concurrent.futures.as_completed(future_to_result):
                    try:
                        result, secondary = future.result()
                        all_final_results.append(result)
                        if secondary: newly_discovered.extend(secondary)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {future_to_result[future].get('doi')}: {e}")
                    finally:
                        if progress_obj and round_task: 
                            progress_obj.update(round_task, advance=1)

            # Cleanup Round Bar
            if progress_obj and round_task: 
                progress_obj.remove_task(round_task)

            # Add new citations to queue for next round
            for pub in self._deduplicate_publications(newly_discovered):
                if (doi := pub.get('doi')) and doi not in processed_dois:
                    publications_queue.append(pub)
                    processed_dois.add(doi)

        all_final_results.sort(key=self._get_year)

        # --- FLATTENED ANALYSIS SUMMARY ---
        success_count = sum(1 for r in all_final_results if r.get("status") == "✅ Extraction complete.")
        failed_pubs = [
            f"DOI {r.get('doi') or 'N/A'} ('{(r.get('publication_title') or 'Unknown Title')[:40]}...') failed..."
            for r in all_final_results if r.get("status") != "✅ Extraction complete."
        ]
        
        self.logger.info(
            f"Analysis Complete: {clean_accession} | "
            f"Total: {len(all_final_results)} | "
            f"Success: {success_count} | "
            f"Failures: {len(failed_pubs)}"
        )
        for failure in failed_pubs:
            self.logger.warning(failure)
        
        if use_cache and self.cache_path:
            try:
                results_str = json.dumps(all_final_results)
                with sqlite3.connect(self.cache_path) as conn:
                    conn.execute("INSERT OR REPLACE INTO publication_cache (bioproject_id, results_json) VALUES (?, ?)", (clean_accession, results_str))
                self.logger.debug(f"Cached publication results for '{clean_accession}'.")
            except sqlite3.Error as e: 
                self.logger.error(f"Cache write failed for '{clean_accession}': {e}")
        
        return all_final_results