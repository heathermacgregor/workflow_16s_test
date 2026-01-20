import re
import requests
import pdfplumber
import io
import json
import concurrent.futures
import time
import os
import logging
import xml.etree.ElementTree as ET
import csv
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup

# --- Configuration & Setup ---

# Set up basic logging for better control over output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load email for API politeness from an environment variable
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "default.email@example.com")
if UNPAYWALL_EMAIL == "default.email@example.com":
    logging.warning("UNPAYWALL_EMAIL not set. Using a default. Please set this environment variable for API politeness.")

# Load API keys from environment variables to keep secrets out of the code
SPRINGER_NATURE_API_KEY = os.getenv("SPRINGER_NATURE_API_KEY")
IEEE_XPLORE_API_KEY = os.getenv("IEEE_XPLORE_API_KEY")
MENDELEY_API_KEY = os.getenv("MENDELEY_API_KEY")
DIMENSIONS_API_KEY = os.getenv("DIMENSIONS_API_KEY")

# Base URL for NCBI E-utilities
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Use a requests.Session for connection pooling and setting a common User-Agent
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({
    "User-Agent": f"BioProjectPublicationExtractor/1.0 (mailto:{UNPAYWALL_EMAIL})"
})


# --- Core Helper Functions ---

def fix_spacing_in_text(text: str) -> str:
    """Adds missing spaces often lost during PDF parsing using common heuristics."""
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([.,;:])([a-zA-Z\d])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(-)([a-zA-Z])', r'\1 \2 \3', text)
    text = re.sub(r'([\]\)])([a-zA-Z\d\[])', r'\1 \2', text)
    return re.sub(r'\s+', ' ', text).strip()

def find_methods_section(full_text: str, n_chars: int = 25000) -> str:
    """Finds and isolates the Materials and Methods section from full text."""
    methods_start_pattern = re.compile(
        r'^\b(materials and methods|methods|experimental|experimental procedures|methodology)\b\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    next_section_pattern = re.compile(
        r'^\b(results|discussion|conclusions|acknowledgements|author contributions|references)\b\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    methods_match = methods_start_pattern.search(full_text)
    if not methods_match:
        return "Materials and Methods section not found."
    text_after_methods_start = full_text[methods_match.end():]
    next_section_match = next_section_pattern.search(text_after_methods_start)
    if next_section_match:
        return text_after_methods_start[:next_section_match.start()].strip()
    return text_after_methods_start[:n_chars].strip()

def isolate_reference_section(full_text: str) -> str:
    """Finds and isolates the References/Bibliography section."""
    ref_start_pattern = re.compile(
        r'\b(references|bibliography|works\s+cited|literature\s+cited)\b',
        re.IGNORECASE
    )
    search_start_index = len(full_text) * 2 // 3
    ref_match = ref_start_pattern.search(full_text, pos=search_start_index)
    return full_text[ref_match.start():].strip() if ref_match else ""

def get_year(pub: Dict[str, Any]) -> int:
    """Retrieves the publication year as an integer, defaulting to 9999 for sorting."""
    try:
        return int(str(pub.get('pub_year'))[:4])
    except (TypeError, ValueError):
        return 9999

def deduplicate_publications(pub_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate publications based on DOI, or title and year as a fallback."""
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

def get_pmc_article_url(doi: str) -> Optional[str]:
    """Uses NCBI ESearch to get the PMCID and returns the corresponding PMC article URL."""
    params = {"db": "pmc", "term": doi, "retmode": "json", "tool": "PublicationExtractor", "email": UNPAYWALL_EMAIL}
    try:
        response = HTTP_SESSION.get(f"{NCBI_EUTILS_BASE}/esearch.fcgi", params=params, timeout=10)
        response.raise_for_status()
        pmcid_list = response.json().get('esearchresult', {}).get('idlist', [])
        if pmcid_list:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_list[0]}/"
    except requests.exceptions.RequestException as e:
        logging.debug(f"Could not get PMC URL for DOI {doi}: {e}")
    return None

def extract_text_from_webpage(url: str) -> Optional[str]:
    """Fetches and extracts text from a URL using BeautifulSoup for robust parsing."""
    try:
        logging.info(f"Attempting to parse text from URL: {url}...")
        response = HTTP_SESSION.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=' ', strip=True)
        if len(text) < 500:
            logging.warning(f"Webpage parsing for {url} yielded very little text.")
            return None
        logging.info(f"Successfully parsed text from {url}.")
        return text
    except Exception as e:
        logging.error(f"Error during webpage parsing for {url}: {e}")
        return None

# --- Primer Validation Functions ---

def load_primer_database(path: str) -> Dict[str, Dict[str, str]]:
    """
    Loads the primer details TSV into a dictionary keyed by primer sequence for fast lookups.
    """
    primer_db = {}
    if not os.path.exists(path):
        logging.error(f"Primer database not found at '{path}'. Cannot perform validation.")
        return primer_db
    
    try:
        with open(path, 'r', newline='', encoding='utf-8') as tsvfile:
            # Use csv.DictReader, assuming a standard tab-separated file.
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                raw_sequence = row.get('Sequence')
                if not raw_sequence:
                    continue
                
                # Clean the sequence string: remove 5'- / -3' and any spaces
                sequence = re.sub(r"5'-|'|-3'|\s", "", raw_sequence).strip()

                if sequence:
                    # Some columns have extra info, just grab the first part.
                    probebase_id = (row.get('ProbeBase ID') or '').split()[0] if row.get('ProbeBase ID') else ''

                    primer_details = {
                        "name": row.get('Primer Name', '').strip(),
                        "probebase_id": probebase_id,
                        "length": row.get('Length [nt]', '').strip(),
                        "gc_content": row.get('G+C content [%]', '').strip(),
                        "target_rrna": row.get('Target rRNA', '').strip()
                    }
                    primer_db[sequence.upper()] = primer_details
        logging.info(f"Loaded {len(primer_db)} primers into memory from '{path}'.")
    except Exception as e:
        logging.error(f"Failed to load or parse primer database TSV: {e}", exc_info=True)
    
    return primer_db

def validate_primers(found_sequences: List[str], primer_db: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Checks a list of found primer sequences against the loaded primer database.
    """
    validated = []
    if not primer_db:
        return validated
    
    for seq in found_sequences:
        match = primer_db.get(seq.upper())
        if match:
            validated.append({
                "sequence_found": seq,
                "probebase_match": match
            })
    return validated

# --- Secondary Citation Helpers ---

def find_citations_near_accession(full_text: str, accession: str, context_chars: int = 250) -> Tuple[List[Dict[str, Any]], int]:
    """
    Finds instances of the accession, extracts context for citation patterns,
    and returns the found citations along with the total count of accession mentions.
    """
    # First, normalize all whitespace in the full text to single spaces for easier regex.
    normalized_text = re.sub(r'\s+', ' ', full_text)
    
    # Remove spaces from the accession ID for a canonical search key.
    clean_accession = accession.replace(" ", "")

    # Normalize the text specifically for finding the accession number without spaces.
    search_text = re.sub(r'(\w)\s+(\d+)', r'\1\2', normalized_text, flags=re.IGNORECASE)

    citations_found = []
    
    # Find all matches first to get a total count.
    matches = list(re.finditer(re.escape(clean_accession), search_text, re.IGNORECASE))
    total_mentions = len(matches)
    logging.info(f"Total mentions: {total_mentions}")
    for match in matches:
        # Define a wide context for the snippet to be returned
        context_start = max(0, match.start())
        context_end = min(len(search_text), match.end() + context_chars)
        context_snippet = search_text[context_start:context_end]
        logging.info(context_snippet)
        
        # Define a NARROW search zone for finding the actual citation clues
        search_zone_start = max(0, match.start()) # Look 40 chars before
        search_zone_end = min(len(search_text), match.end() + 40) # Look 40 chars after
        search_zone = search_text[search_zone_start:search_zone_end]

        # --- Broader and more robust citation clue extraction ---

        # Pattern 1: Author-Year style, e.g., (Lloyd-Price et al., 2017) or Smith, 2020.
        author_year_pattern = re.compile(
            r'\(?((?:[\w-]+\s?){1,3} et al\.?,? \d{4}|(?:[\w-]+\s?){1,3}, \d{4})\)?', 
            re.IGNORECASE
        )
        # Search only within the narrow zone
        author_year_matches = author_year_pattern.findall(search_zone)
        if author_year_matches:
            logging.info(author_year_matches)

        # Pattern 2: Numbered style in brackets or parentheses, e.g., [1], (1), [5–8], (5, 10).
        numbered_pattern = re.compile(r'[\[\(]\s*(\d+)\s*(?:[–-]\s*\d+)?(?:\s*,\s*\d+)*\s*[\]\)]')
        # Search only within the narrow zone
        numbered_matches = numbered_pattern.findall(search_zone)
        if numbered_matches:
            logging.info(numbered_matches)

        clues = []
        if author_year_matches:
            clues.extend([f"Author-Year clue: {c.strip()}" for c in set(author_year_matches)])
        if numbered_matches:
            clues.extend([f"Numbered clue: {c}" for c in set(numbered_matches)])

        if clues:
            # We still return the wider context snippet for user readability,
            # but the clues are now accurately associated with the accession ID.
            citations_found.append({
                "context_snippet": context_snippet.strip(),
                "citation_clues": clues
            })
            
    return citations_found, total_mentions

def find_citation_entry_by_number(reference_section: str, number: str) -> Optional[str]:
    """Searches the ISOLATED reference section for a numbered citation entry."""
    try:
        next_number = str(int(number) + 1)
    except ValueError:
        return None

    current_ref_start_pattern = r'(\s*|^)(\[' + re.escape(number) + r'\]|\b' + re.escape(number) + r'\.)\s*'
    next_ref_start_pattern = r'(\s*|^)(\[' + re.escape(next_number) + r'\]|\b' + re.escape(next_number) + r'\.)\s*'
    full_entry_pattern = re.compile(
        current_ref_start_pattern + r'(.*?)' + r'(?=' + next_ref_start_pattern + r'|$)',
        re.IGNORECASE | re.DOTALL
    )
    match = full_entry_pattern.search(reference_section)
    if match:
        citation_number_part = match.group(2).strip()
        citation_content = match.group(3).strip()
        return f"{citation_number_part} {citation_content}"
    return None

def search_citation_details_via_crossref(title_or_author_year: str, accession: str) -> List[Dict[str, Any]]:
    """Uses Crossref to find a publication based on a title/author clue for secondary search."""
    publications = []
    params = {"query": title_or_author_year, "rows": 3, "mailto": UNPAYWALL_EMAIL, "sort": "relevance"}
    try:
        response = HTTP_SESSION.get("https://api.crossref.org/works", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('message', {}).get('items', [])
        for item in items:
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
                "pdf_url": None,
                "materials_and_methods": None,
                "extracted_info": {},
                "status": "Ready (Cited)"
            })
    except requests.exceptions.RequestException as e:
        logging.warning(f"Secondary Crossref lookup failed for '{title_or_author_year}': {e}")
    return publications


# --- Tiered API Search Functions ---

def get_publications_from_ncbi(accession: str) -> List[Dict[str, Any]]:
    """Tier 1: Uses NCBI ELink/ESummary to find linked publications."""
    publications = []
    try:
        elink_params = {"dbfrom": "bioproject", "db": "pubmed", "id": accession, "retmode": "json", "tool": "PublicationExtractor", "email": UNPAYWALL_EMAIL}
        elink_resp = HTTP_SESSION.get(f"{NCBI_EUTILS_BASE}/elink.fcgi", params=elink_params, timeout=10)
        elink_resp.raise_for_status()
        linksets = elink_resp.json().get('linksets', [])
        pmids = [link['Id'] for ls in linksets for lsd in ls.get('linksetdbs', []) for link in lsd.get('links', [])]
        if not pmids: return []

        esummary_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json", "tool": "PublicationExtractor", "email": UNPAYWALL_EMAIL}
        esummary_resp = HTTP_SESSION.get(f"{NCBI_EUTILS_BASE}/esummary.fcgi", params=esummary_params, timeout=15)
        esummary_resp.raise_for_status()
        result = esummary_resp.json()['result']
        for pmid, pub_data in result.items():
            if pmid == 'uids': continue
            doi = next((aid['value'] for aid in pub_data.get('articleids', []) if aid.get('idtype') == 'doi'), None)
            publications.append({"bioproject_accession": accession, "publication_title": pub_data.get('title'), "pub_year": pub_data.get('pubdate', '')[:4], "doi": doi, "status": "Ready (NCBI)"})
    except Exception as e:
        logging.error(f"Tier 1 (NCBI) failed: {e}")
    return publications

def get_publications_from_crossref(accession: str) -> List[Dict[str, Any]]:
    """Tier 2: Uses Crossref to find publications mentioning the accession."""
    publications = []
    params = {"query": accession, "rows": 10, "mailto": UNPAYWALL_EMAIL, "sort": "published", "order": "asc"}
    try:
        response = HTTP_SESSION.get("https://api.crossref.org/works", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('message', {}).get('items', [])
        for item in items:
            title = (item.get('title') or ["Unknown Title"])[0]
            year = item.get('issued', {}).get('date-parts', [[None]])[0][0]
            publications.append({"bioproject_accession": accession, "publication_title": title, "pub_year": str(year) if year else "N/A", "doi": item.get('DOI'), "status": "Ready (Crossref)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 2 (Crossref) failed: {e}")
    return publications

def get_publications_from_datacite(accession: str) -> List[Dict[str, Any]]:
    """Tier 3: Uses DataCite to find publications mentioning the accession."""
    publications = []
    params = {"query": accession, "page[size]": 10, "sort": "published"}
    try:
        response = HTTP_SESSION.get("https://api.datacite.org/works", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('data', [])
        for item in items:
            attrs = item.get('attributes', {})
            title = (attrs.get('titles', [{}])[0].get('title', "Unknown Title"))
            publications.append({"bioproject_accession": accession, "publication_title": title, "pub_year": str(attrs.get('publicationYear', 'N/A')), "doi": attrs.get('doi'), "status": "Ready (DataCite)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 3 (DataCite) failed: {e}")
    return publications

def get_publications_from_semantic_scholar(accession: str) -> List[Dict[str, Any]]:
    """Tier 4: Uses Semantic Scholar to find papers mentioning the accession."""
    publications = []
    params = {"query": accession, "fields": "title,year,externalIds", "limit": 10}
    try:
        response = HTTP_SESSION.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('data', [])
        for item in items:
            doi = item.get('externalIds', {}).get('DOI')
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('year', 'N/A')), "doi": doi, "status": "Ready (Semantic Scholar)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 4 (Semantic Scholar) failed: {e}")
    return publications

def get_publications_from_europe_pmc(accession: str) -> List[Dict[str, Any]]:
    """Tier 5: Uses Europe PMC to find papers mentioning the accession."""
    publications = []
    params = {"query": accession, "resultType": "lite", "format": "json", "pageSize": 10}
    try:
        response = HTTP_SESSION.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('resultList', {}).get('result', [])
        for item in items:
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('pubYear', 'N/A')), "doi": item.get('doi'), "status": "Ready (Europe PMC)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 5 (Europe PMC) failed: {e}")
    return publications

def get_publications_from_plos(accession: str) -> List[Dict[str, Any]]:
    """Tier 6: Uses PLOS API to find papers mentioning the accession."""
    publications = []
    params = {"q": f'"{accession}"', "fl": "id,publication_date,title", "wt": "json", "rows": 10, "sort": "publication_date asc"}
    try:
        response = HTTP_SESSION.get("http://api.plos.org/search", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('response', {}).get('docs', [])
        for item in items:
            year = item.get('publication_date', 'N/A')[:4]
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(year), "doi": item.get('id'), "status": "Ready (PLOS)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 6 (PLOS) failed: {e}")
    return publications

def get_publications_from_springer_nature(accession: str) -> List[Dict[str, Any]]:
    """Tier 7: Uses Springer Nature API to find publications."""
    if not SPRINGER_NATURE_API_KEY:
        logging.warning("Tier 7 (Springer Nature) skipped: API key not set.")
        return []
    publications = []
    params = {"q": f'fulltext:"{accession}"', "api_key": SPRINGER_NATURE_API_KEY, "p": 10}
    try:
        response = HTTP_SESSION.get("http://api.springernature.com/openaccess/json", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('records', [])
        for item in items:
            year = item.get('publicationDate', 'N/A')[:4]
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(year), "doi": item.get('doi'), "status": "Ready (Springer)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 7 (Springer Nature) failed: {e}")
    return publications

def get_publications_from_base_search(accession: str) -> List[Dict[str, Any]]:
    """Tier 8: Uses BASE Search API to find academic papers."""
    publications = []
    params = {"q": accession, "format": "json", "sort": "date:asc", "limit": 10}
    try:
        response = HTTP_SESSION.get("https://api.base-search.net/v2/search", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('response', {}).get('docs', [])
        for item in items:
            year = item.get('year') or (item.get('date', 'N/A')[:4])
            doi = item.get('doi')
            if isinstance(doi, list): doi = doi[0]
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(year), "doi": doi, "status": "Ready (BASE)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 8 (BASE Search) failed: {e}")
    return publications

def get_publications_from_doaj(accession: str) -> List[Dict[str, Any]]:
    """Tier 9: Uses DOAJ API to find open access papers."""
    publications = []
    params = {"q": f'bibjson.abstract:"{accession}"', "sort": "created_date:asc", "pageSize": 10}
    try:
        response = HTTP_SESSION.get("https://doaj.org/api/search/articles", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('results', [])
        for item in items:
            bibjson = item.get('bibjson', {})
            doi = next((i['id'] for i in bibjson.get('identifier', []) if i.get('type') == 'doi'), None)
            publications.append({"bioproject_accession": accession, "publication_title": bibjson.get('title', "Unknown Title"), "pub_year": str(bibjson.get('year', 'N/A')), "doi": doi, "status": "Ready (DOAJ)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 9 (DOAJ) failed: {e}")
    return publications

def get_publications_from_arxiv(accession: str) -> List[Dict[str, Any]]:
    """Tier 10: Uses ArXiv API to find pre-prints."""
    publications = []
    params = {"search_query": f'all:"{accession}"', "sortBy": "submittedDate", "sortOrder": "ascending", "max_results": 10}
    try:
        response = HTTP_SESSION.get("http://export.arxiv.org/api/query", params=params, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('a:entry', ns):
            title = entry.find('a:title', ns).text.strip()
            year = entry.find('a:published', ns).text[:4]
            doi_link = entry.find('a:link[@title="doi"]', ns)
            doi = doi_link.attrib.get('href', '').split('doi.org/')[-1] if doi_link is not None else None
            publications.append({"bioproject_accession": accession, "publication_title": title, "pub_year": str(year), "doi": doi, "status": "Ready (ArXiv)"})
    except Exception as e:
        logging.error(f"Tier 10 (ArXiv) failed: {e}")
    return publications

def get_publications_from_ieee_xplore(accession: str) -> List[Dict[str, Any]]:
    """Tier 11: Uses IEEE Xplore API to find publications."""
    if not IEEE_XPLORE_API_KEY:
        logging.warning("Tier 11 (IEEE Xplore) skipped: API key not set.")
        return []
    publications = []
    params = {"querytext": f'"{accession}"', "apikey": IEEE_XPLORE_API_KEY, "max_records": 10, "sortfield": "publication_year", "sortorder": "asc"}
    try:
        response = HTTP_SESSION.get("https://ieeexploreapi.ieee.org/api/v1/search/articles", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get('articles', [])
        for item in items:
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('publication_year', 'N/A')), "doi": item.get('doi'), "status": "Ready (IEEE)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 11 (IEEE Xplore) failed: {e}")
    return publications

def get_publications_from_mendeley(accession: str) -> List[Dict[str, Any]]:
    """Tier 12: Uses Mendeley Catalog API to find publications."""
    if not MENDELEY_API_KEY:
        logging.warning("Tier 12 (Mendeley) skipped: API key not set.")
        return []
    publications = []
    headers = {"Authorization": f"Bearer {MENDELEY_API_KEY}"}
    params = {"query": f'"{accession}"', "view": "all", "limit": 10, "sort": "year", "direction": "asc"}
    try:
        response = HTTP_SESSION.get("https://api.mendeley.com/catalog", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        items = response.json()
        for item in items:
            doi = item.get('identifiers', {}).get('doi')
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('year', 'N/A')), "doi": doi, "status": "Ready (Mendeley)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 12 (Mendeley) failed: {e}")
    return publications

def get_publications_from_core(accession: str) -> List[Dict[str, Any]]:
    """Tier 13: Uses CORE API to find papers from institutional repositories."""
    publications = []
    query_data = {"q": accession, "limit": 10, "sort": "yearPublished:asc"}
    try:
        response = HTTP_SESSION.post("https://api.core.ac.uk/v3/search/works", json=query_data, timeout=15)
        response.raise_for_status()
        items = response.json().get('results', [])
        for item in items:
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('yearPublished', 'N/A')), "doi": item.get('doi'), "status": "Ready (CORE)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 13 (CORE) failed: {e}")
    return publications

def get_publications_from_dimensions(accession: str) -> List[Dict[str, Any]]:
    """Tier 14: Uses Dimensions API to find publications."""
    if not DIMENSIONS_API_KEY:
        logging.warning("Tier 14 (Dimensions) skipped: API key not set.")
        return []
    publications = []
    query = f'search publications for "{accession}" return publications[title,year,doi] sort by year asc limit 10'
    try:
        response = HTTP_SESSION.post("https://app.dimensions.ai/api/dsl.json", data=query.encode('utf-8'), timeout=15)
        response.raise_for_status()
        items = response.json().get('publications', [])
        for item in items:
            publications.append({"bioproject_accession": accession, "publication_title": item.get('title', "Unknown Title"), "pub_year": str(item.get('year', 'N/A')), "doi": item.get('doi'), "status": "Ready (Dimensions)"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Tier 14 (Dimensions) failed: {e}")
    return publications

def get_publications_from_biorxiv(accession: str) -> List[Dict[str, Any]]:
    """Tier 15: Uses bioRxiv/medRxiv API to find pre-prints."""
    publications = []
    try:
        logging.warning("Tier 15 (bioRxiv) is a placeholder as the API does not support broad keyword searching.")
    except Exception as e:
        logging.error(f"Tier 15 (bioRxiv) failed: {e}")
    return publications

# --- Main Worker & Orchestrator ---

def analyze_single_publication(pub_data: Dict[str, Any], pub_index: int, total_pubs: int, primer_db: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Analyzes a single publication: finds text, extracts methods, and finds secondary citations."""
    title, doi, accession = pub_data.get("publication_title", "N/A"), pub_data.get("doi"), pub_data['bioproject_accession']
    log_prefix = f"Worker {pub_index}/{total_pubs} (DOI: {doi})"
    logging.info(f"{log_prefix}: Processing '{title}'")

    # Initialize the accession mentions count. It will be updated if text is found.
    pub_data["accession_mentions_in_text"] = 0

    if not doi:
        pub_data["status"] = "⚠️ No DOI available."
        return pub_data, []

    # 1. Find Full Text (PDF -> PMC -> DOI Page)
    full_text = None
    try:
        unpaywall_resp = HTTP_SESSION.get(f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}", timeout=10)
        if unpaywall_resp.status_code == 200 and (best_oa := unpaywall_resp.json().get("best_oa_location")) and (pdf_url := best_oa.get("url_for_pdf")):
            pub_data["pdf_url"] = pdf_url
            pdf_response = HTTP_SESSION.get(pdf_url, timeout=45)
            pdf_response.raise_for_status()
            with pdfplumber.open(io.BytesIO(pdf_response.content)) as pdf:
                full_text = "".join(p.extract_text() + "\n" for p in pdf.pages if p.extract_text())
    except Exception as e:
        logging.warning(f"{log_prefix}: PDF download/parse failed: {e}. Trying fallbacks.")

    if not full_text and (pmc_url := get_pmc_article_url(doi)):
        full_text = extract_text_from_webpage(pmc_url)
    if not full_text:
        full_text = extract_text_from_webpage(f"https://doi.org/{doi}")

    if not full_text:
        pub_data["status"] = "❌ Failed to retrieve text."
        return pub_data, []

    full_text_clean = fix_spacing_in_text(full_text)
    
    # 2. Find and process secondary citations
    newly_found_pubs = []
    secondary_citations, total_mentions = find_citations_near_accession(full_text_clean, accession)
    pub_data["accession_mentions_in_text"] = total_mentions
    pub_data["secondary_citations_found"] = secondary_citations

    if total_mentions > 0:
        logging.info(f"{log_prefix}: Found {total_mentions} total mention(s) of accession '{accession}' in the text.")
    else:
        logging.info(f"{log_prefix}: No mentions of accession '{accession}' found in the text.")

    if secondary_citations:
        logging.info(f"{log_prefix}: Found {len(secondary_citations)} of these mentions near citation patterns. Searching for cited works.")
        reference_section = isolate_reference_section(full_text_clean)
        for citation_info in secondary_citations:
            for clue in citation_info['citation_clues']:
                search_term = None
                if clue.startswith("Author-Year clue:"):
                    search_term = clue.replace("Author-Year clue:", "").strip()
                elif clue.startswith("Numbered clue:") and reference_section:
                    if number_match := re.search(r'\d+', clue):
                        number = number_match.group(0)
                        search_term = find_citation_entry_by_number(reference_section, number)
                
                if search_term:
                    found = search_citation_details_via_crossref(search_term, accession)
                    newly_found_pubs.extend(found)
    elif total_mentions > 0:
        logging.info(f"{log_prefix}: None of the accession mentions were found near citation patterns.")

    # 3. Extract Methods and Keywords
    methods_text = find_methods_section(full_text_clean)
    text_to_scan = methods_text
    
    if "not found" in methods_text:
        logging.warning(f"{log_prefix}: Methods section not found. Scanning full text for keywords.")
        text_to_scan = full_text_clean # FALLBACK: Scan the entire document
        pub_data["materials_and_methods"] = "Not found; keyword extraction performed on full text."
    else:
        pub_data["materials_and_methods"] = methods_text

    patterns = {
        "gene_mentions": r'16s\s*(?:rrna|rdna|gene)?',
        "variable_regions": r'\b(v[1-9](?:-v[1-9])?)\b',
        "primer_names": r'\b(\d+\s?[fr]|[fr]\s?\d+)\b',
        "primer_sequences": r'\b([acgtunryswmkbdhv]{15,})\b',
        "sequencing_platforms": r'\b(illumina|miseq|hiseq|novaseq|pacbio|sequel|oxford nanopore|minion|gridion|ion torrent|pyrosequencing)\b'
    }
    
    extracted_info = {}
    for key, p in patterns.items():
        matches = re.findall(p, text_to_scan, re.IGNORECASE)
        if not matches:
            continue

        if key == "primer_names":
            # Remove spaces from primer names like '515 F' that are artifacts of fix_spacing_in_text
            processed_matches = [re.sub(r'\s+', '', m) for m in matches]
        else:
            processed_matches = matches
        
        extracted_info[key] = sorted(list(set(processed_matches)))

    pub_data["extracted_info"] = extracted_info
    
    # NEW: Validate found primer sequences against the database
    if primer_db and "primer_sequences" in pub_data["extracted_info"]:
        found_sequences = pub_data["extracted_info"]["primer_sequences"]
        validated_primers = validate_primers(found_sequences, primer_db)
        if validated_primers:
            logging.info(f"{log_prefix}: Validated {len(validated_primers)} primer sequences against probebase.")
            pub_data["extracted_info"]["validated_primers"] = validated_primers

    pub_data["status"] = "✅ Extraction complete."
        
    return pub_data, newly_found_pubs

def extract_bioproject_sequencing_info(bioproject_accession: str, primer_db: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Orchestrates the tiered API search and parallel, recursive analysis of publications."""
    clean_accession = bioproject_accession.replace(" ", "").strip()
    
    logging.info(f"--- Starting Initial Search for '{clean_accession}' across 15 Tiers ---")
    tier_functions = [
        get_publications_from_ncbi, get_publications_from_crossref, get_publications_from_datacite,
        get_publications_from_semantic_scholar, get_publications_from_europe_pmc, get_publications_from_plos,
        get_publications_from_springer_nature, get_publications_from_base_search, get_publications_from_doaj,
        get_publications_from_arxiv, get_publications_from_ieee_xplore, get_publications_from_mendeley,
        get_publications_from_core, get_publications_from_dimensions, get_publications_from_biorxiv
    ]
    
    initial_publications = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tier_functions)) as executor:
        future_to_tier = {executor.submit(func, clean_accession): func.__name__ for func in tier_functions}
        for future in concurrent.futures.as_completed(future_to_tier):
            try:
                tier_pubs = future.result()
                if tier_pubs:
                    logging.info(f"Found {len(tier_pubs)} publications via {future_to_tier[future]}.")
                    initial_publications.extend(tier_pubs)
            except Exception as e:
                logging.error(f"Error in tier {future_to_tier[future]}: {e}")

    if not initial_publications:
        logging.warning(f"No publications found for '{bioproject_accession}'.")
        return [{"status": "No publications found."}]

    unique_initial_pubs = deduplicate_publications(initial_publications)
    unique_initial_pubs.sort(key=get_year)
    
    # --- Setup for Recursive Processing ---
    processed_dois = set()
    publications_queue = []
    for pub in unique_initial_pubs:
        if doi := pub.get('doi'):
            publications_queue.append(pub)
            processed_dois.add(doi)
            
    all_final_results = []
    round_count = 0
    MAX_ROUNDS = 3 # Safety limit for recursive depth
    
    logging.info(f"Found {len(publications_queue)} unique publications with DOIs to start analysis.")

    while publications_queue and round_count < MAX_ROUNDS:
        round_count += 1
        current_queue_size = len(publications_queue)
        logging.info(f"\n======== RECURSION ROUND {round_count} (Processing {current_queue_size} publications) ========")
        
        newly_discovered_pubs_this_round = []
        
        pubs_for_this_round = publications_queue[:]
        publications_queue.clear()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_result = {executor.submit(analyze_single_publication, pub, i + 1, current_queue_size, primer_db): pub for i, pub in enumerate(pubs_for_this_round)}
            
            for future in concurrent.futures.as_completed(future_to_result):
                try:
                    result, secondary_pubs = future.result()
                    all_final_results.append(result)
                    if secondary_pubs:
                        newly_discovered_pubs_this_round.extend(secondary_pubs)
                except Exception as e:
                    logging.error(f"Error in analysis worker (Round {round_count}): {e}", exc_info=True)

        if newly_discovered_pubs_this_round:
            unique_secondary = deduplicate_publications(newly_discovered_pubs_this_round)
            logging.info(f"✨ Round {round_count}: Found {len(unique_secondary)} potential secondary publications.")
            
            for pub in unique_secondary:
                if (doi := pub.get('doi')) and doi not in processed_dois:
                    publications_queue.append(pub)
                    processed_dois.add(doi)
        
        if publications_queue:
            logging.info(f"⏳ Queued {len(publications_queue)} new publications for Round {round_count + 1}.")
        else:
            logging.info(f"🏁 No new publications found in Round {round_count}.")

    logging.info("\n--- All Processing Rounds Complete ---")
    all_final_results.sort(key=get_year)
    return all_final_results

# --- Main Execution ---
if __name__ == '__main__':
    # Define paths for data files
    # Assumes 'primer_details_results.tsv' is in the same directory as the script.
    primer_db_path = "/usr2/people/macgregor/amplicon/workflow_16s/src/data/primer_details_results.tsv"
    
    # 1. Load the primer database into memory for quick searching
    # NOTE: Ensure the file 'primer_details_results.tsv' is present.
    primer_db = load_primer_database(primer_db_path)

    # 2. Proceed with the main extraction logic
    bioproject_id = 'PRJEB42019'
    logging.info(f"Starting extraction for BioProject ID: {bioproject_id}")
    
    # To run, set your email in the environment:
    # export UNPAYWALL_EMAIL="your.email@example.com"
    # And optionally, any API keys you have:
    # export SPRINGER_NATURE_API_KEY="your-key"
    
    extracted_data = extract_bioproject_sequencing_info(bioproject_id, primer_db)

    logging.info("--- Final Results ---")
    print(json.dumps(extracted_data, indent=2))
    
    # Find and log the oldest paper that mentions the accession ID in its text
    papers_with_mentions = [
        p for p in extracted_data if p.get("accession_mentions_in_text", 0) > 0
    ]

    if papers_with_mentions:
        # The main list is already sorted by year, so the first item is the oldest.
        oldest_paper = papers_with_mentions[0]
        logging.info("\n--- Oldest Paper with ID Mention ---")
        logging.info(f"The oldest publication found that explicitly mentions '{bioproject_id}' in its text is:")
        logging.info(f"  Title: {oldest_paper.get('publication_title', 'N/A')}")
        logging.info(f"  Year: {oldest_paper.get('pub_year', 'N/A')}")
        logging.info(f"  DOI: {oldest_paper.get('doi', 'N/A')}")
    else:
        logging.info(f"\n--- No publications found that explicitly mention '{bioproject_id}' in their text. ---")

