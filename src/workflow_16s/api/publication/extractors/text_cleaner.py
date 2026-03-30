# workflow_16s/api/publication/extractors/text_cleaner.py

import re
import requests
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, List, Tuple
from bs4 import BeautifulSoup
from workflow_16s.utils.logger import get_logger

def extract_text_from_webpage(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """Extracts the main article text from a given URL safely within a thread."""
    req_method = session.get if session else requests.get
        
    try:
        response = req_method(url, timeout=25)
        response.raise_for_status()
            
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('article') or soup.find('main') or soup.body
            
        if not main_content: 
            return None
                
        for tag in main_content.find_all('a'):
            link_text = tag.get_text().lower()
            if any(k in link_text for k in ['table', 'supp', 'si', 'file', 'data']):
                # Convert the link to a text marker the LLM can see
                tag.replace_with(f" [LINK: {tag.get_text()} - {tag.get('href')}] ")
            else:
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

def fix_spacing_in_text(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([.,;:])([a-zA-Z\d])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(-)([a-zA-Z])', r'\1 \2 \3', text)
    text = re.sub(r'([\]\)])([a-zA-Z\d\[])', r'\1 \2', text)
    return re.sub(r'\s+', ' ', text).strip()

def find_methods_section(text: str) -> str:
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
        pattern = r'(?m)^[ \t]*' + re.escape(header) + r'[ \t]*[:\.\n\r]'
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

def isolate_reference_section(full_text: str) -> str:
    ref_start_pattern = re.compile(r'\b(references|bibliography|works\s+cited|literature\s+cited)\b', re.IGNORECASE)
    search_start_index = len(full_text) * 2 // 3
    #ref_match = ref_start_pattern.search(full_text, pos=search_start_index)
    ref_match = ref_start_pattern.search(full_text, search_start_index)
    return full_text[ref_match.start():].strip() if ref_match else ""

def get_year(pub: Dict[str, Any]) -> int:
    try:
        return int(str(pub.get('pub_year'))[:4])
    except (TypeError, ValueError):
        return 9999
        
def find_citations_near_accession(
    full_text: str, accession: str, context_chars: int = 250
) -> Tuple[List[Dict[str, Any]], int]:
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

def find_citation_entry_by_number(reference_section: str, number: str) -> Optional[str]:
    try: next_number = str(int(number) + 1)
    except ValueError: return None
    current_ref_start_pattern = r'(\s*|^)(\[' + re.escape(number) + r'\]|\b' + re.escape(number) + r'\.)\s*'
    next_ref_start_pattern = r'(\s*|^)(\[' + re.escape(next_number) + r'\]|\b' + re.escape(next_number) + r'\.)\s*'
    full_entry_pattern = re.compile(current_ref_start_pattern + r'(.*?)' + r'(?=' + next_ref_start_pattern + r'|$)', re.IGNORECASE | re.DOTALL)
    match = full_entry_pattern.search(reference_section)
    if match: return f"{match.group(2).strip()} {match.group(3).strip()}"
    return None
    
# --- METHODOLOGY EXTRACTION ---
def extract_methodology_details(text_to_scan: str) -> Dict[str, Any]:
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

def extract_dna_sequences(text: str) -> List[str]:
    """
    Robustly mines DNA sequences, handling journal-specific formatting like 
    prime symbols, dashes, and internal spaces.
    """
    # 1. PRE-PROCESS: Clean common journal 'junk' from the text
    # Replace prime symbols (5', 3'), hyphens, and fancy quotes
    clean_text = text.replace("′", "'").replace("’", "'")
    
    # 2. TARGET REGEX:
    # This looks for DNA but allows internal dashes or spaces that we'll strip later
    # It catches: "AGAGTTTGATC MTGGCTCAG" or "AGAG-TTTG-ATC"
    dna_candidate_pattern = r'\b([ACGTUNRYSWKMBDHV\s-]{15,50})\b'
    
    candidates = re.findall(dna_candidate_pattern, clean_text.upper())
    
    final_sequences = []
    for cand in candidates:
        # Remove spaces and hyphens to get the raw sequence
        seq = cand.replace(" ", "").replace("-", "").replace("\n", "")
        
        # Validation: Must be 15-35bp and contain mostly DNA characters
        # This prevents catching long strings of "MATERIALS AND METHODS"
        if 15 <= len(seq) <= 40:
            # Check DNA density: ensure at least 80% of the string is ACTG + IUPAC
            dna_chars = sum(1 for char in seq if char in "ACGTUNRYSWKMBDHV")
            if (dna_chars / len(seq)) > 0.9:
                final_sequences.append(seq)
                
    return sorted(list(set(final_sequences)), key=len)

def fetch_si_text(id_1: str, id_2: str = "", session: Any = None, timeout: int = 60, *args, **kwargs) -> str:
    """
    Attempts to fetch Supplementary Information (SI) using the Europe PMC REST API.
    Accepts multiple identifiers (e.g., PMID and DOI) and automatically uses the best one.
    """
    logger = get_logger("workflow_16s")
    si_content = []
    # Grab the thread-safe session if provided, otherwise fallback to the default
    request_session = session
    # Intelligently grab the DOI if it exists in either argument, otherwise fallback to the first ID
    identifier = str(id_1)
    if id_2 and '/' in str(id_2):
        identifier = str(id_2)
    elif '/' in str(id_1):
        identifier = str(id_1)
            
    if not identifier or identifier.lower() == 'nan':
        return ""
        
    try:
        logger.debug(f"Querying Europe PMC for SI text: {identifier}")
            
        # STEP 1: Resolve the identifier (DOI or PMID) to a PMCID
        query_str = f'DOI:"{identifier}"' if '/' in identifier else f'EXT_ID:"{identifier}"'
           
        search_params = {
            "query": query_str,
            "format": "json",
            "resultType": "core"
        }
            
        search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        # Change self.http_session to request_session
        search_resp = request_session.get(search_url, params=search_params, timeout=timeout)
        #search_resp = self.http_session.get(search_url, params=search_params, timeout=self.timeout)
        search_resp.raise_for_status()
            
        results = search_resp.json().get("resultList", {}).get("result", [])
        if not results:
            logger.debug(f"No Europe PMC record found for {identifier}")
            return ""
                
        pmcid = results[0].get("pmcid")
        if not pmcid:
            logger.debug(f"No open-access PMCID available for {identifier}")
            return ""

        # STEP 2: Fetch the full-text XML using the PMCID
        xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        # Change self.http_session to request_session
        xml_resp = request_session.get(xml_url, timeout=timeout)
        #xml_resp = self.http_session.get(xml_url, timeout=self.timeout)
            
        if xml_resp.status_code != 200:
            logger.debug(f"Could not retrieve full text XML for {pmcid}")
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
        logger.warning(f"Network error fetching SI for {identifier}: {e}")
    except ET.ParseError as e:
        logger.warning(f"Failed to parse XML for {identifier}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error extracting SI for {identifier}: {e}")
            
    # Deduplicate (in case tags overlap) and join into a single text block
    unique_si = list(dict.fromkeys(si_content))
    final_text = "\n\n".join(unique_si)
        
    if final_text:
        logger.debug(f"Successfully extracted {len(final_text)} chars of SI for {identifier}")
    # 🟢 THE FIX: Detect "Link Ghosts"
    if len(final_text) < 100 and "click here" in final_text.lower():
        logger.warning(f"SI content for {identifier} is just a link placeholder.")
        return "" # Return empty so the fetcher knows to try the Zenodo/Lens fallback
           
    return final_text