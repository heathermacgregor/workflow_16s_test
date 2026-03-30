# workflow_16s/upstream/sequences/primers/probebase.py

import csv
import re
import sqlite3
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar


PROBEBASE_LIST_URL = "https://probebase.net/lists/probes/"
PROBEBASE_SEARCH_URL = "https://probebase.net/search/results/reference/"
PROBEBASE_DETAIL_URL = "https://probebase.net/results/{primer_id}/"

DATA_DIR = Path("data")
INPUT_CSV_PATH = DATA_DIR / "probe_data.csv"
OUTPUT_TSV_PATH = DATA_DIR / "primer_details_results.tsv"
DB_PATH = DATA_DIR / "primer_data.db"


def build_primer_database_direct(session: requests.Session, csv_path: Path, db_path: Path):
    """
    Reads primer names from CSV, scrapes details, and streams DIRECTLY to SQLite,
    eliminating the need for an intermediate TSV file.
    """
    logger = get_logger("workflow_16s")
    
    # 1. Read the initial CSV
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            primer_names = sorted(list(set([row[0].strip() for row in reader if row and row[0].strip()])))
    except (IOError, csv.Error) as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    # 2. Setup SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS primers')
    
    # Pre-define schema based on expected ProbeBase fields
    schema = (
        "Primer_Name TEXT, ProbeBase_ID TEXT, Sequence TEXT, Length TEXT, "
        "Position TEXT, Direction TEXT, Accession_no TEXT, References TEXT, "
        "Position_Start INTEGER, Position_End INTEGER"
    )
    cursor.execute(f'CREATE TABLE primers ({schema})')
    
    insert_sql = '''INSERT INTO primers 
                    (Primer_Name, ProbeBase_ID, Sequence, Length, Position, 
                     Direction, Accession_no, References, Position_Start, Position_End) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

    # 3. Scrape and Stream to DB
    total_primers = len(primer_names)
    logger.info(f"Found {total_primers} unique primers. Scraping directly to SQLite...")
    
    successful_inserts = 0
    with get_progress_bar() as progress:
        task = progress.add_task("[cyan]Scraping & Building DB...", total=total_primers)
        
        for name in primer_names:
            primer_id = get_primer_id_from_search(session, name)
            if primer_id:
                details = get_primer_details(session, primer_id, name)
                if details:
                    # Parse positions
                    start, end = None, None
                    pos_val = details.get("Position", "")
                    match = re.match(r'(\d+)\s*-\s*(\d+)', pos_val)
                    if match:
                        start, end = int(match.group(1)), int(match.group(2))
                    
                    # Insert directly into DB
                    cursor.execute(insert_sql, (
                        details.get("Primer Name"), details.get("ProbeBase ID"), 
                        details.get("Sequence"), details.get("Length"), pos_val,
                        details.get("Direction"), details.get("Accession no."), 
                        details.get("References"), start, end
                    ))
                    successful_inserts += 1
                    
            sleep(0.1) # Be respectful to the server
            progress.update(task, advance=1)

    # 4. Create Indexes and Commit
    logger.info("Creating database indexes...")
    cursor.execute('CREATE INDEX idx_pos_start ON primers (Position_Start)')
    cursor.execute('CREATE INDEX idx_pos_end ON primers (Position_End)')
    cursor.execute('CREATE INDEX idx_direction ON primers (Direction)')
    
    conn.commit()
    conn.close()
    logger.info(f"✅ Successfully built database with {successful_inserts} primers at {db_path}")
    
def download_probebase_csv(session: requests.Session, save_path: Path) -> bool:
    """
    Downloads a CSV file from probebase.net using a persistent session.

    Args:
        session: The requests.Session object for making HTTP requests.
        save_path: The pathlib.Path object where the CSV should be saved.

    Returns:
        True if download was successful, False otherwise.
    """
    logger = get_logger("workflow_16s")
    params = {
        'category': '',
        'target_rna': '16',
        'insitu': '',
        'is_primer': 'True',
        '_export': 'csv'
    }
    try:
        logger.info(f"Attempting to download data from {PROBEBASE_LIST_URL}...")
        response = session.get(PROBEBASE_LIST_URL, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Ensure the parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"✅ Successfully downloaded and saved to {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ An error occurred during download: {e}")
        return False


def get_primer_id_from_search(session: requests.Session, primer_name: str) -> Optional[str]:
    """
    Searches ProbeBase for a primer ID using a two-step process.

    1.  First, it tries to find an <a> tag whose link text is an exact,
        case-insensitive match for the primer name. This is the most reliable method.
    2.  If that fails, it searches for list items with the pattern:
        Primer Name (<a href="/results/...">...</a>) and extracts the ID from the link.

    Args:
        session: The requests.Session object.
        primer_name: The name of the primer to search for.

    Returns:
        The primer ID as a string, or None if not found or an error occurs.
    """
    logger = get_logger("workflow_16s")
    if not primer_name: return None

    target_name = primer_name.strip()
    params = {'probename': target_name, 'link': 'or', 'filter': 'pcr'}
    try:
        response = session.get(PROBEBASE_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml') # type: ignore

        # --- STRATEGY 1: Find an exact link text match (most reliable) ---
        detail_links = soup.find_all('a', href=re.compile(r'/results/\d+/')) # type: ignore
        for link in detail_links:
            if link.get_text(strip=True).lower() == target_name.lower():
                match = re.search(r'/results/(\d+)/', link['href']) # type: ignore
                if match:
                    primer_id = match.group(1)
                    logger.info(f" ✅ Strategy 1: Found exact link for '{target_name}' with ID {primer_id}.")
                    return primer_id

        # --- STRATEGY 2: Fallback to find "Primer Name (<a href...>" pattern ---
        # This regex looks for the exact primer name as a whole word.
        search_pattern = re.compile(r'\b' + re.escape(target_name) + r'\b', re.IGNORECASE)
        
        # Find the text node that contains the primer name.
        primer_text_node = soup.find(string=search_pattern) # type: ignore
        
        if primer_text_node:
            # Find the parent element (likely <li> or <p>) that contains both the text and the link.
            parent_element = primer_text_node.find_parent()
            if parent_element:
                link = parent_element.find('a', href=re.compile(r'/results/\d+/')) # type: ignore
                if link and 'href' in link.attrs: # type: ignore
                    match = re.search(r'/results/(\d+)/', link['href']) # type: ignore
                    if match:
                        primer_id = match.group(1)
                        logger.info(f" ✅ Strategy 2: Found linked ID {primer_id} for '{target_name}' in list format.")
                        return primer_id

        # If both strategies fail
        logger.warning(f" ❌ Could not find a link for '{target_name}' using either strategy.")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f" ❌ Error searching for '{target_name}': {e}")
        return None


def get_primer_details(session: requests.Session, primer_id: str, primer_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetches and parses the detail page for a specific primer ID using BeautifulSoup.
    Also logs the extracted data at the DEBUG level.

    Args:
        session: The requests.Session object.
        primer_id: The ID of the primer.
        primer_name: The name of the primer.

    Returns:
        A dictionary containing the extracted primer details, or None on error.
    """
    logger = get_logger("workflow_16s")
    detail_url = PROBEBASE_DETAIL_URL.format(primer_id=primer_id)
    logger.debug(f"Fetching details from: {detail_url}")
    details = {"Primer Name": primer_name, "ProbeBase ID": f"pB-{primer_id}"}
    
    try:
        response = session.get(detail_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml') # type: ignore

        header = soup.find('strong', string='Accession no.') # type: ignore
        if not header:
            logger.warning(f"Could not find details table for primer ID {primer_id}.")
            return None
        
        table = header.find_parent('table')
        if not table:
            logger.warning(f"Could not find parent table for primer ID {primer_id}.")
            return None
        
        for row in table.find_all('tr'): # type: ignore
            cells = row.find_all('td') # type: ignore
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(separator=' ', strip=True).replace(';', ',')
                details[key] = value

        return details
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching details for ID {primer_id}: {e}")
        return None


# --- Data Processing and Storage Functions ---

def process_and_save_primer_data(session: requests.Session, csv_path: Path, tsv_path: Path):
    """Reads primer names, scrapes their details, and saves them to a TSV file."""
    logger = get_logger("workflow_16s")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            primer_names = sorted(list(set([row[0].strip() for row in reader if row and row[0].strip()])))
    except (IOError, csv.Error) as e:
        logger.error(f"Failed to read or parse CSV file {csv_path}: {e}")
        return

    all_primer_data = []
    total_primers = len(primer_names)
    logger.info(f"Found {total_primers} unique primer names to process.")
    
    with get_progress_bar() as progress:
        task = progress.add_task("[cyan]Scraping primer data...", total=total_primers)
        for name in primer_names:
            primer_id = get_primer_id_from_search(session, name)
            if primer_id:
                details = get_primer_details(session, primer_id, name)
                if details:
                    all_primer_data.append(details)
            sleep(0.1) # Be respectful to the server
            progress.update(task, advance=1)
    
    if not all_primer_data:
        logger.warning("No primer data was successfully scraped. TSV file will not be created.")
        return

    # Build a comprehensive set of all unique fieldnames from all scraped items
    all_fieldnames = set()
    for item in all_primer_data:
        all_fieldnames.update(item.keys())
    
    # Define a preferred order for common columns for readability
    preferred_order = [
        "Primer Name", "ProbeBase ID", "Sequence", "Length", "Position", 
        "Direction", "Accession no.", "References"
    ]
    # Create the final ordered list of fieldnames for the TSV header
    fieldnames = sorted(
        list(all_fieldnames), 
        key=lambda x: (preferred_order.index(x) if x in preferred_order else len(preferred_order), x)
    )

    try:
        with open(tsv_path, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(all_primer_data)
        logger.info(f"\n✅ Successfully saved details for {len(all_primer_data)} primers to {tsv_path}")
    except IOError as e:
        logger.error(f"❌ Error writing results to TSV file {tsv_path}: {e}")


def create_and_populate_db(tsv_path: Path, db_path: Path):
    """
    Reads data from a TSV file and populates an optimized SQLite database.
    """
    logger = get_logger("workflow_16s")
    if not tsv_path.exists():
        logger.error(f"TSV file not found at {tsv_path}. Cannot create database.")
        return

    conn = None  # Define conn here to ensure it's available in the finally block
    try:
        with open(tsv_path, 'r', newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            headers = next(reader)
            
            sanitized_headers = [re.sub(r'[^a-zA-Z0-9_]', '_', h) for h in headers]
            final_headers = sanitized_headers + ['Position_Start', 'Position_End']
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('DROP TABLE IF EXISTS primers')
            columns_schema_parts = [f'"{h}" TEXT' for h in sanitized_headers]
            columns_schema_parts.extend(['"Position_Start" INTEGER', '"Position_End" INTEGER'])
            create_table_sql = f'CREATE TABLE primers ({", ".join(columns_schema_parts)})'
            cursor.execute(create_table_sql)
            
            logger.info("Created SQLite table 'primers'.")

            quoted_headers = ', '.join([f'"{h}"' for h in final_headers])
            placeholders = ', '.join(['?'] * len(final_headers))
            insert_sql = f'INSERT INTO primers ({quoted_headers}) VALUES ({placeholders})'

            data_to_insert = []
            position_idx = headers.index('Position') if 'Position' in headers else -1
            if position_idx == -1:
                logger.error("'Position' column not found in TSV. Cannot parse start/end positions.")
                return

            for row in reader:
                pos_val = row[position_idx]
                start, end = None, None
                match = re.match(r'(\d+)\s*-\s*(\d+)', pos_val)
                if match:
                    start, end = int(match.group(1)), int(match.group(2))
                
                data_to_insert.append(tuple(row) + (start, end))

            cursor.executemany(insert_sql, data_to_insert)
            
            logger.info("Creating database indexes for faster queries...")
            cursor.execute('CREATE INDEX idx_pos_start ON primers (Position_Start)')
            cursor.execute('CREATE INDEX idx_pos_end ON primers (Position_End)')
            cursor.execute('CREATE INDEX idx_direction ON primers (Direction)')
            
            conn.commit()
            logger.info(f" ✅ Successfully populated database with {len(data_to_insert)} records.")

    except (sqlite3.Error, IOError, csv.Error, ValueError) as e:
        logger.error(f" ❌ An error occurred during database creation: {e}")
    finally:
        if conn:
            conn.close()


def query_primers(db_path: Path, target_position: int, leeway: int, direction: str) -> List[Dict[str, Any]]:
    """
    Queries the SQLite database for primers within a specified position range.
    """
    logger = get_logger("workflow_16s")
    if not db_path.exists():
        logger.error(f"Database file not found at {db_path}.")
        return []
        
    query_range_start = max(0, target_position - leeway)
    query_range_end = target_position + leeway

    # This query efficiently finds any primer whose position range (Start to End)
    # overlaps with the query range (query_range_start to query_range_end).
    query_sql = """
        SELECT * FROM primers
        WHERE
            Position_Start <= ? AND
            Position_End >= ? AND
            TRIM(Direction) = ?
        ORDER BY Position_Start;
    """
    params = (query_range_end, query_range_start, direction.strip())
    
    logger.debug(
        f"Searching for primers that overlap with range "
        f"{query_range_start}-{query_range_end} and have Direction '{direction.strip()}'..."
    )

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query_sql, params)
            results = [dict(row) for row in cursor.fetchall()]
            return results
            
    except sqlite3.Error as e:
        logger.error(f" ❌ SQLite query error: {e}")
        return []
    
# --- Main Workflow & Demonstration ---

def import_and_save_database(
    csv_path: Path = INPUT_CSV_PATH,
    db_path: Path = DB_PATH
):
    logger = get_logger("workflow_16s")
    if db_path.exists():
        logger.info(f" ✅ Database already exists at '{db_path}'. Skipping import.")
        return

    logger.info("Database not found. Starting scraping and import process...")
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
        })

        if not download_probebase_csv(session, csv_path):
            logger.critical("Failed to download initial data. Aborting.")
            return

        # 🚀 Use the new direct-to-db function
        build_primer_database_direct(session, csv_path, db_path)
        
        # Cleanup the downloaded CSV to save space
        if csv_path.exists():
            csv_path.unlink()


def query_primer_pairs(
    db_path: Path,
    forward_position: int,
    reverse_position: int,
    leeway: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Finds all possible combinations of forward and reverse primers based on
    target positions and a leeway.

    Args:
        db_path (Path): The path to the SQLite database.
        forward_position (int): The target center position for forward primers.
        reverse_position (int): The target center position for reverse primers.
        leeway (int): The search range (+/-) around the target positions.

    Returns:
        A list of tuples, where each tuple contains a pair of matching
        (forward_primer_dict, reverse_primer_dict). Returns an empty list
        if no pairs are found.
    """
    forward_primers = query_primers(
        db_path=db_path, target_position=forward_position,
        leeway=leeway, direction='Forward primer'
    )

    reverse_primers = query_primers(
        db_path=db_path, target_position=reverse_position,
        leeway=leeway, direction='Reverse primer'
    )

    if not forward_primers or not reverse_primers:
        return []

    # Create every possible combination of the found primers
    primer_pairs = []
    for f_primer in forward_primers:
        for r_primer in reverse_primers:
            # Sanity check to ensure the forward primer starts before the reverse primer
            if f_primer.get('Position_Start') and r_primer.get('Position_Start') and \
                f_primer['Position_Start'] < r_primer['Position_Start']:
                primer_pairs.append((f_primer, r_primer))

    return primer_pairs

def main():
    """
    Main function to build the database and then query for primers matching
    several common 16S rRNA gene subfragments.
    """
    logger = get_logger("workflow_16s")
    # Step 1: Ensure the database is created and populated.
    import_and_save_database()

    if not DB_PATH.exists():
        logger.critical("Database file not found or created. Cannot run queries.")
        return

    # Step 2: Define common 16S variable regions to query.
    V_REGIONS = {
        "V4": {"fwd_pos": 515, "rev_pos": 806, "leeway": 50},
        "V3-V4": {"fwd_pos": 341, "rev_pos": 805, "leeway": 50},
        "V1-V2": {"fwd_pos": 27, "rev_pos": 338, "leeway": 40},
        "V1-V3": {"fwd_pos": 27, "rev_pos": 534, "leeway": 50},
        "V6-V8": {"fwd_pos": 926, "rev_pos": 1392, "leeway": 75}
    }

    # Step 3: Iterate through each region, query for pairs, and print results.
    for region_name, params in V_REGIONS.items():
        header = f" 🔍 Querying Primer Pairs for Region: {region_name} "
        print("\n\n" + f"{header:=^80}")

        primer_pairs = query_primer_pairs(
            db_path=DB_PATH,
            forward_position=params["fwd_pos"],
            reverse_position=params["rev_pos"],
            leeway=params["leeway"]
        )

        if primer_pairs:
            logger.info(f"✅ Found {len(primer_pairs)} possible primer pair combinations for {region_name}.")
            print(f"\n--- Top 5 Matching Primer Pairs for {region_name} ---")
            for fwd, rev in primer_pairs[:5]:
                fwd_name = fwd.get('Primer_Name', 'N/A')
                fwd_pos = fwd.get('Position', 'N/A')
                rev_name = rev.get('Primer_Name', 'N/A')
                rev_pos = rev.get('Position', 'N/A')
                print(
                    f"  FWD: {fwd_name:<15} (Pos: {fwd_pos:<12})  |  "
                    f"REV: {rev_name:<15} (Pos: {rev_pos:<12})"
                )
            if len(primer_pairs) > 5:
                print(f"  ... and {len(primer_pairs) - 5} more pairs.")
        else:
            logger.warning(f" ❌ No matching primer pairs found for region {region_name}.")

    print("\n" + "="*80)
    logger.info(" ✅ All region queries complete.")


if __name__ == "__main__":
    main()