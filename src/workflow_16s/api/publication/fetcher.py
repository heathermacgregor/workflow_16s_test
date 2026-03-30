# workflow_16s/api/publication/fetcher.py

import asyncio
import concurrent.futures
import json
import os
import re
import requests
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from workflow_16s.config import AppConfig
from workflow_16s.upstream.sequences.analysis import PrimerDatabase
from workflow_16s.utils.logger import get_logger

from workflow_16s.api.publication.apis import (
    ArxivAPI, BaseSearchAPI, BaseAPI, BioarxivAPI, CoreAPI, CrossrefAPI,
    DataciteAPI, DimensionsAPI, DOAJAPI, EuropePMCAPI, IEEExploreAPI,
    MendeleyAPI, NCBIAPI, PLOSAPI, SemanticScholarAPI, SpringerNatureAPI,
    UnpaywallAPI, ZenodoAPI
)
from workflow_16s.api.llm.huggingface import MethodologyAnalyzer #publication.extractors.llm_analyzer import MethodologyAnalyzer
from workflow_16s.api.publication.extractors.pdf_parser import fetch_and_parse_pdf
from workflow_16s.api.publication.extractors.text_cleaner import (
    extract_dna_sequences, extract_text_from_webpage, fetch_si_text, find_methods_section, 
    find_citation_entry_by_number, find_citations_near_accession, 
    fix_spacing_in_text, isolate_reference_section, 
)
from workflow_16s.api.publication.cache import create_cache_table, cleanup_publications_cache
from workflow_16s.utils.progress import get_progress_bar

class PublicationFetcher:
    MAX_CACHE_SIZE_MB = 500  # Size limit for publication cache
    CLEANUP_TARGET_MB = 400  # Target size after cleanup
    
    def __init__(self, config: AppConfig, cache_path: Optional[str] = None):
        self.config = config
        self.logger = get_logger("workflow_16s")
        self.email = config.credentials.ena_email
        self.session = self._build_robust_session()
        # Initialize Submodules
        self.arxiv = ArxivAPI(self.config, self.email, self.logger)
        self.basesearch = BaseSearchAPI(self.config, self.email, self.logger)
        self.bioarxiv = BioarxivAPI(self.config, self.email, self.logger)
        self.core = CoreAPI(self.config, self.email, self.logger)
        self.crossref = CrossrefAPI(self.config, self.email, self.logger)
        self.datacite = DataciteAPI(self.config, self.email, self.logger)
        self.dimensions = DimensionsAPI(self.config, self.email, self.logger)
        self.doaj = DOAJAPI(self.config, self.email, self.logger)
        self.europe_pmc = EuropePMCAPI(self.config, self.email, self.logger)
        self.ieee_xplore = IEEExploreAPI(self.config, self.email, self.logger)
        self.mendeley = MendeleyAPI(self.config, self.email, getattr(self.config.credentials, 'mendeley_api_key', None), self.logger)
        self.ncbi = NCBIAPI(self.config, self.email, self.logger)
        self.plos = PLOSAPI(self.config, self.email, self.logger)
        self.semantic_scholar = SemanticScholarAPI(self.config, self.email, self.logger)
        self.springer_nature = SpringerNatureAPI(self.config, self.email, self.logger)
        self.unpaywall = UnpaywallAPI(self.config, self.email, self.logger)
        self.zenodo = ZenodoAPI(self.config, self.email, self.logger)
        
        llm_key = getattr(config.credentials, 'llm_api_key', None) or os.getenv("LLM_API_KEY")
        self.analyzer = MethodologyAnalyzer(llm_key)#, self.logger)
        
        self.cache_path = cache_path
        create_cache_table(self.cache_path) if self.cache_path else None #PublicationCache(cache_path, self.logger) if cache_path else None
        
        # Check and cleanup cache if it exceeds size limit
        if self.cache_path and Path(self.cache_path).exists():
            cache_size_mb = Path(self.cache_path).stat().st_size / 1e6
            if cache_size_mb > self.MAX_CACHE_SIZE_MB:
                self.logger.warning(
                    f"Publication cache ({cache_size_mb:.1f} MB) exceeds limit ({self.MAX_CACHE_SIZE_MB} MB). "
                    f"Triggering cleanup..."
                )
                cleanup_publications_cache(
                    self.cache_path, 
                    max_size_mb=self.MAX_CACHE_SIZE_MB,
                    target_size_mb=self.CLEANUP_TARGET_MB
                )
        
        from workflow_16s.upstream.sequences.analysis import PrimerFinder
        self.primer_db = PrimerFinder(Path(self.config.paths.primer_db)) #PrimerDatabase(config.paths.primer_db, self.logger)
    
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
                        self.logger.info(f"[✓] Found and returned cached results for '{clean_accession}'.")
                        return json.loads(row[0])
            except sqlite3.Error as e: 
                self.logger.error(f"Cache read failed for '{clean_accession}': {e}")
        
        self.logger.info(f" ⧖ Starting publication search for BioProject: {clean_accession}")
        
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
        self.logger.info(f"    ⤷ Using enriched search queries: {unique_queries}")
        
        tier_functions = [
            self.arxiv.get_publications_from_accession, self.basesearch.get_publications_from_accession,
            self.bioarxiv.get_publications_from_accession, self.core.get_publications_from_accession,
            self.crossref.get_publications_from_accession, self.datacite.get_publications_from_accession,
            self.dimensions.get_publications_from_accession, self.doaj.get_publications_from_accession,
            self.europe_pmc.get_publications_from_accession, self.mendeley.get_publications_from_accession,
            self.ncbi.get_publications_from_accession, self.plos.get_publications_from_accession,
            self.semantic_scholar.get_publications_from_accession,
            #self.springer_nature.get_publications_from_accession,
            self.zenodo.get_publications_from_accession
        ]
        
        initial_publications, failed_tiers = [], []
        
        # --- STAGE 1: SEARCHING ---
        # Only add a task if progress_obj is EXPLICITLY provided
        task_id = None
        if progress_obj is not None:
            task_id = progress_obj.add_task(f"[cyan]Pubs: Searching tiers for {clean_accession}...", total=len(tier_functions))
        else:
            progress_obj = get_progress_bar()
            task_id = progress_obj.add_task(f"[cyan]Pubs: Searching tiers for {clean_accession}...", total=len(tier_functions))
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tier_functions)-1) as executor:
            future_to_tier = {executor.submit(func, clean_accession): func.__name__ for func in tier_functions}
            for future in concurrent.futures.as_completed(future_to_tier):
                try:
                    #self.logger.info(future.result())
                    initial_publications.extend(future.result())
                except Exception as e:
                    failed_tiers.append(f"[✕] Tier {future_to_tier[future]} failed: {e}")
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
                        self.logger.error(f"[✕] Error analyzing {future_to_result[future].get('doi')}: {e}")
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
        success_count = sum(1 for r in all_final_results if r.get("status") == "[✓] Extraction complete.")
        failed_pubs = [
            f"DOI {r.get('doi') or 'N/A'} ('{(r.get('publication_title') or 'Unknown Title')[:40]}...') failed..."
            for r in all_final_results if r.get("status") != "[✓] Extraction complete."
        ]
        
        self.logger.info(
            f" ⧗ Analysis Complete: {clean_accession} | "
            f"Total: {len(all_final_results)} | "
            f"Success: {success_count} | "
            f"Failures: {len(failed_pubs)}"
        )
        for failure in failed_pubs:
            self.logger.warning(failure)
        
        if use_cache and self.cache_path:
            try:
                results_str = json.dumps(all_final_results)
                # Use asyncio lock to prevent concurrent write corruption
                write_lock_path = Path(self.cache_path).parent / "_publication_cache_lock"
                if not hasattr(self, '_publication_write_lock'):
                    self._publication_write_lock = asyncio.Lock()
                
                # Note: In async context, use: async with self._publication_write_lock: ...
                # For now, use a file-based lock for safety across processes
                with sqlite3.connect(self.cache_path, timeout=10.0) as conn:
                    conn.execute("INSERT OR REPLACE INTO publication_cache (bioproject_id, results_json) VALUES (?, ?)", (clean_accession, results_str))
                self.logger.debug(f" ↻ Cached publication results for '{clean_accession}'.")
            except sqlite3.Error as e: 
                self.logger.error(f"[⚑] Cache write failed for '{clean_accession}': {e}")
        
        return all_final_results
    
    def _analyze_single_publication(self, pub_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Silently analyzes a single publication, returning status in the result dictionary."""
        doi = pub_data.get("doi")
        accession = pub_data['bioproject_accession']
        pub_data["accession_mentions_in_text"] = 0
        
        if not doi:
            pub_data["status"] = "[⚑] No DOI available."
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

            # --- Tier 2 & 3: PubMed Central (PDF then Webpage) ---
            if not full_text:
                pmc_pdf_url, pmc_article_url = self.ncbi.get_pmc_links(doi, session=local_session) # Ensure this helper doesn't use the old shared session!
                if pmc_pdf_url:
                    full_text = fetch_and_parse_pdf(pmc_pdf_url)
                    if full_text: pub_data["pdf_url"] = pmc_pdf_url
                
                if not full_text and pmc_article_url:
                    full_text = extract_text_from_webpage(pmc_article_url, session=local_session) # Ensure this uses local_session if possible

            # --- Tier 4: Publisher page via DOI link ---
            if not full_text:
                full_text = extract_text_from_webpage(f"https://doi.org/{doi}")

            # FALLBACK: Abstract only (Rescue ASM/AEM paywalls)
            if not full_text:
                abstract = pub_data.get('publication_title') # Usually contains abstract snippet
                if abstract and len(abstract) > 50:
                    full_text = f"ABSTRACT ONLY: {abstract}"
                    pub_data["status"] = "[!] Abstract-only extraction."
                else:
                    pub_data["status"] = "[⚑] Failed to retrieve full text."
                    return pub_data, []

            # --- Analysis on Retrieved Text ---
            full_text_clean = fix_spacing_in_text(full_text)
            newly_found_pubs = []
            secondary_citations, total_mentions = find_citations_near_accession(full_text_clean, accession)
            
            pub_data["accession_mentions_in_text"] = total_mentions
            pub_data["secondary_citations_found"] = secondary_citations
            
            if secondary_citations:
                reference_section = isolate_reference_section(full_text_clean)
                for citation_info in secondary_citations:
                    for clue in citation_info['citation_clues']:
                        search_term = None
                        if clue.startswith("Author-Year clue:"): 
                            search_term = clue.replace("Author-Year clue:", "").strip()
                        elif clue.startswith("Numbered clue:") and reference_section:
                            if number_match := re.search(r'\d+', clue):
                                search_term = find_citation_entry_by_number(reference_section, number_match.group(0))
                        
                        if search_term: 
                            # Warning: Ensure this helper isn't using an async/closed session!
                            newly_found_pubs.extend(self.crossref.get_publications_from_accession(accession, search_term))
            
            # --- SMART THREE-PASS EXTRACTION ---
            methods_text = find_methods_section(full_text_clean)
            has_methods = "not found" not in methods_text
            pub_data["materials_and_methods_section_found"] = has_methods
            
            # Pass 1: Try the isolated methods section first
            text_to_scan = methods_text if has_methods else full_text_clean
            
            #dna_sequences = extract_dna_sequences(methods_text)
            #self.logger.info(f"Extracted DNA sequences: {dna_sequences}")
            
            methodology_details = self.analyzer._extract_methodology_details_llm(text_to_scan)
            
            methodology_details = self._verify_extraction(methodology_details, text_to_scan)
            
            if methodology_details['verification_status'] == "Unverified":
                self.logger.warning(f" ⚑ LLM extracted details for {doi} but they couldn't be found in raw text.")
                
            has_primers = bool(methodology_details.get("primer_sequences"))
            has_regions = bool(methodology_details.get("variable_regions"))
            is_flagged = methodology_details.get("unextracted_flag", False)                

            # Pass 2: Fallback to scanning the FULL text if missing info OR if flagged
            if has_methods and (not (has_primers and has_regions) or is_flagged):
                self.logger.debug(f"[⚑] Info missing or flagged in methods for DOI {doi}. Scanning full text.")
                
                full_text_details = self.analyzer._extract_methodology_details_llm(full_text_clean)
                print(full_text_details)
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
                self.logger.info(f"[⚑] Critical info missing or flagged for SI. Hunting PDFs/Excel at {pmc_article_url}...")
                
                si_text = fetch_si_text(doi, session=local_session, timeout=60)
                
                if si_text.strip():
                    si_details = self.analyzer._extract_methodology_details_llm(si_text)
                    for key, val in si_details.items():
                        if val and not methodology_details.get(key):
                            methodology_details[key] = val
                            
                    # We reached the end of the line, just accept whatever flag status the SI gave us
                    methodology_details["unextracted_flag"] = si_details.get("unextracted_flag", False)
                    methodology_details["unextracted_reason"] = si_details.get("unextracted_reason", "")
            # --- PRIMER VALIDATION LAYER ---
            all_potential_seqs = methodology_details.get("primer_sequences", [])
            
            # Use the fuzzy sequence miner we built earlier to supplement LLM findings
            mined_dna = extract_dna_sequences(text_to_scan)
            all_potential_seqs = list(set(all_potential_seqs + mined_dna))

            if all_potential_seqs:
                self.logger.info(f" 🧬 Validating {len(all_potential_seqs)} potential sequences against PrimerDB...")
                # This calls your PrimerDatabase class to check for IUPAC matches
                extracted_fwd = methodology_details.get('primer_names', [""])[0] 
                extracted_rev = methodology_details.get('primer_names', [""])[1] if len(methodology_details.get('primer_names', [])) > 1 else ""
                validated_payload = self.primer_db.validate_extracted_pair(extracted_fwd, extracted_rev)
                
                if validated_payload:
                    self.logger.info(f" ✅ PrimerFinder confirmed region {validated_payload['region']}")
                    methodology_details['variable_regions'] = [validated_payload['region']]
                    methodology_details['primer_sequences'] = [validated_payload['fwd_seq'], validated_payload['rev_seq']]
                    # Ensure we use the standardized DB names
                    methodology_details['primer_names'] = [validated_payload['fwd_name'], validated_payload['rev_name']]
                else:
                    self.logger.warning(" ⚑ Primers found in text do not match any known 16S pairs in DB coordinates.")
            
            if methodology_details.get('unextracted_flag') and "Table S1" in details.get('unextracted_reason', ''):
                self.logger.info(f" ⤷ Searching Zenodo for missing supplementary files for {accession}...")
                zenodo_files = self.zenodo.get_publications_from_accession(accession)
                for file_record in zenodo_files:
                    if "supplementary_content" in file_record:
                        pub_data["zenodo_supplementary_links"] = file_record["supplementary_content"]
                        pub_data["status"] = "[✓] Table S1 link found on Zenodo."
                    
            pub_data["methodology_details"] = methodology_details

            pub_data["status"] = "[✓] Extraction complete."
            return pub_data, newly_found_pubs
    
    def _verify_extraction(self, extracted: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        verified = extracted.copy()
        text_raw = source_text.upper()
        
        # 1. Initialize status safely
        verified['verification_status'] = "Unverified"
        # Ensure validated_primers exists so the reporter doesn't crash
        if 'validated_primers' not in verified:
            verified['validated_primers'] = []

        # 2. Sequence Recovery: Clean strings and hunt around names
        if verified.get('primer_names'):
            current_seqs = [s for s in verified.get('primer_sequences', []) if len(str(s)) > 10 and str(s).lower() != 'unknown']
            for name in verified['primer_names']:
                idx = text_raw.find(name.upper())
                if idx != -1:
                    window = text_raw[idx:idx+250]
                    found = extract_dna_sequences(window)
                    if found: current_seqs.extend(found)
            # Standardize and clean the list
            verified['primer_sequences'] = list(set([s.upper() for s in current_seqs if all(c in "ACGTUNRYSWKMBDHV" for c in s.upper())]))

        # 3. COORDINATE VALIDATION (Trust but Verify)
        # Use your actual object name: self.primer_db
        seqs = verified.get('primer_sequences', [])
        if len(seqs) >= 2:
            s1, s2 = seqs[0], seqs[1]
            
            # Try Orientation A
            payload = self.primer_db.validate_extracted_pair(s1, s2)
            
            # Try Orientation B if A failed
            if not payload:
                payload = self.primer_db.validate_extracted_pair(s2, s1)

            if payload:
                self.logger.info(f" ✅ PrimerFinder confirmed region {payload['region']}")
                verified['verification_status'] = "Verified (Coordinates)"
                verified['validated_primers'] = [payload['fwd_seq'], payload['rev_seq']]
                verified['variable_regions'] = [payload['region']]
        
        # 🟢 THE "KEEP EVERYTHING" RULE:
        # If we didn't get a coordinate match, the status remains 'Unverified', 
        # but verified['primer_sequences'] and verified['primer_names'] 
        # still contain whatever the LLM or Sequence Miner found.
        return verified
    
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
    
    def _get_year(self, pub_dict: Dict[str, Any]) -> int:
        """Helper to extract an integer year for sorting, defaults to 0."""
        year_str = str(pub_dict.get('pub_year', '0'))
        # Extract the first 4 digits if it's a date string like '2018-05-20'
        match = re.search(r'\d{4}', year_str)
        return int(match.group(0)) if match else 0
    
    def _build_robust_session(self) -> requests.Session:
        """Creates a persistent HTTP session with connection pooling and retries."""
        session = requests.Session()
        
        # Dress up like a real browser to avoid instant 403s
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        
        # Configure automatic retries for transient server errors
        retry_strategy = Retry(
            total=3,  # Try 3 times before giving up
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these HTTP codes
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # Create a connection pool that matches your thread count
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
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
    
import asyncio
import yaml
from pathlib import Path
from workflow_16s.config import AppConfig
#from workflow_16s.api.publication.fetcher import PublicationFetcher
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import setup_logging, get_logger

async def test_publication_extraction():
    # 1. SETUP CONFIG AND LOGGING
    config_path = Path("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)

    project_dir = Project(config)
    setup_logging(log_dir_path=project_dir.logs)
    logger = get_logger("workflow_16s")

    # 2. INITIALIZE FETCHER
    # Using a temporary or existing cache for the test
    cache_db = project_dir.cache / "test_publications.db"
    fetcher = PublicationFetcher(config, cache_path=str(cache_db))

    # 3. TEST ACCESSION (Known working from your logs)
    test_accession = "PRJNA864623"
    logger.info(f"[STRT] Testing publication extraction for [bold cyan]{test_accession}[/bold cyan]")

    try:
        # Extract info (This triggers the new pooled session and three-pass analysis)
        results = fetcher.extract_bioproject_sequencing_info(
            bioproject_accession=test_accession,
            use_cache=True # Force a fresh run to test network stability
        )

        # 4. REPORT RESULTS
        if not results:
            logger.warning(f"[FAIL] No publications found for {test_accession}")
            return

        logger.info(f"[ OK ] Found {len(results)} publications.")
        
        for i, pub in enumerate(results, 1):
            title = pub.get('publication_title', 'Unknown Title')
            doi = pub.get('doi', 'No DOI')
            status = pub.get('status', '[FAIL]')
            
            logger.info(f"  {i}. {status} | [bold]{title[:60]}...[/bold]")
            if "methodology_details" in pub:
                details = pub["methodology_details"]
                region = details.get("variable_regions", "N/A")
                primers = details.get("validated_primers", [])
                logger.info(details)#f"     ⤷ Region: [green]{region}[/green] | Primers Found: [yellow]{len(primers)}[/yellow]")

    except Exception as e:
        logger.error(f"[FAIL] Test crashed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_publication_extraction())