import time
import tempfile
from pathlib import Path
import pandas as pd
from io import StringIO
import re
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import subprocess
from rich.progress import track
import shutil
import warnings
import csv
import sys # Added for dynamic import check

# New imports for OCR-based table extraction (kept, but the logic is now superseded
# by the unstructured-based approach for the target tables)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    # Dynamic import for the preferred library for table extraction
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    # Set placeholders if not available, the code handles this by printing an error.
    pytesseract, convert_from_path, Image = None, None, None
    partition_pdf = None


# pypdf (a camelot dependency) may use a deprecated feature in the cryptography library.
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass

# --- CORE PARSING DATA & UTILS (From Block 1) ---

# Known Total Volumes from the document image context. Used as a fallback anchor.
KNOWN_VOLUMES = {
    "Banner Spring Down": "431862320", # 431,862,320
    "Sewer": "22409038",            # 22,409,038
    "West Ditch": "102302320",      # 102,302,320
    "WWTF": "2995114"               # 2,995,114
}

# The combined list of known location titles for splitting the raw text
KNOWN_LOCATIONS = r'(Banner\s*Spring\s*Down|Total:\s*Sewer|Total:\s*West\s*Ditch|Total:\s*WWTF)'

def clean_ocr_value(val: str) -> str:
    """Cleans common OCR errors in scientific notation for reliable conversion to float/int."""
    if not isinstance(val, str):
        return str(val)
    
    cleaned = (val.strip()
               .replace('l', '1') # Common OCR mistake: 'l' (lowercase L) -> '1'
               .replace('I', '1') # Common OCR mistake: 'I' (uppercase I) -> '1'
               .replace('O', '0') # Common OCR mistake: 'O' (uppercase O) -> '0'
               .replace('S', '5') # Occasional OCR mistake: 'S' -> '5'
               .replace('£', 'E') # Currency symbol mistaken for 'E' in E-notation
               .replace(' ', '')  # Remove spaces within numbers (e.g., '431 ,862,320')
               .replace(',', '')  # Remove thousands separator commas
               )
    
    # Heuristically remove a second decimal point if found
    if cleaned.count('.') > 1:
        parts = cleaned.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])

    return cleaned

# Refined Number Pattern: Matches scientific notation or large integers, handling OCR errors (l, I, O)
NUMBER_PATTERN_REFINED = re.compile(
    r'([lI0\d][\s\.,]*[lI0\d]*[Ee][\+\-][lI0\d]+|\d{1,3}(?:,\d{3})*|\d[\.\,]\d+)', 
    re.IGNORECASE
)

def parse_raw_text_table(raw_text: str) -> pd.DataFrame:
    """
    Custom parser to convert the raw, messy OCR text into a clean DataFrame.
    This is necessary when HTML extraction fails. Uses KNOWN_VOLUMES.
    """
    print("--- Starting Custom Raw Text Parser (Fallback) ---")
    
    COLUMNS = [
        "Location", "Activity", "Total Volume (l)", "Activity Concentration (µCi/ml)", 
        "Error Estimate (µCi/ml)", "LLD (µCi/ml)", "Quantity Released (Ci)", 
        "Quantity Released (g)", "Fraction of ECV"
    ]
    all_rows = []
    
    current_location = None
    expected_volume = None
    
    # Refined Isotope Pattern
    ISOTOPE_NAMES = r'(Pu\s*-\s*\d{2,3}(?:\s*/\s*\d{2,3})?|Tc\s*-\s*99|Th\s*-\s*\d{2,3}|U\s*-\s*\d{2,3}(?:\s*/\s*\d{2,3})?|Am\s*-\s*241|Cs\s*-\s*137|Na\s*-\s*22|Np\s*-\s*237|Pb\s*-\s*212)'

    # NEW ROW PATTERN: Find the isotope name, followed by all data, using a lookahead for the next isotope or end marker.
    ROW_PATTERN_SIMPLIFIED = re.compile(
        ISOTOPE_NAMES + 
        r'(.*?)' + # Group 2: Capture all data values (non-greedy)
        r'(?=' + ISOTOPE_NAMES + r'|Total:|$)', # Lookahead for next isotope, 'Total:', or end of block
        re.IGNORECASE | re.DOTALL
    )
    
    raw_text = raw_text.replace('\n', ' ').strip()
    
    # Process the text by splitting on location markers 
    parts = []
    last_end = 0
    
    location_matches = list(re.finditer(KNOWN_LOCATIONS, raw_text, re.IGNORECASE))
    
    if not location_matches and len(raw_text) > 100:
        # Subsequent table without header
        current_location = "Table Data (No Header)"
        expected_volume = None 
        parts = ["Unknown Location Header", raw_text]
    else:
        for m in location_matches:
            parts.append(raw_text[last_end:m.start()].strip()) 
            parts.append(m.group(0).strip())
            last_end = m.end()
        parts.append(raw_text[last_end:].strip())
    
    
    for part in parts:
        if not part.strip():
            continue
        
        location_match = re.match(KNOWN_LOCATIONS, part, re.IGNORECASE)
        
        if location_match or part == "Unknown Location Header":
            if part == "Unknown Location Header":
                current_location = "Table Data (No Header)"
                expected_volume = None
            else:
                current_location = location_match.group(0).replace("Total:", "").strip()
                
                clean_loc = " ".join(current_location.split())
                expected_volume = KNOWN_VOLUMES.get(clean_loc, None)
            continue
        
        data_block = part

        for match in ROW_PATTERN_SIMPLIFIED.finditer(data_block):
            activity = match.group(1).strip().replace(" ", "")
            raw_values = match.group(2).strip()
            
            number_tokens = NUMBER_PATTERN_REFINED.findall(raw_values)
            cleaned_values = [clean_ocr_value(t) for t in number_tokens if clean_ocr_value(t)]

            volume_str = None
            max_vol = 0
            idx_to_remove = -1
            
            # 3a. Try to find the large volume integer in the tokens
            for i, val in enumerate(cleaned_values):
                try:
                    int_val = int(val.split('E')[0].split('.')[0])
                    if int_val > max_vol and len(str(int_val)) > 5:
                         max_vol = int_val
                         volume_str = val
                         idx_to_remove = i
                except ValueError:
                    continue

            remaining_metrics = []

            if volume_str:
                remaining_metrics = cleaned_values[:idx_to_remove] + cleaned_values[idx_to_remove+1:]
                
            elif expected_volume:
                volume_str = expected_volume
                remaining_metrics = cleaned_values
                
            if volume_str is None or len(remaining_metrics) < 6:
                if not activity.lower().startswith("total"):
                    # print(f"Skipping row for {activity} in {current_location or 'None'}: Could not anchor volume and/or find enough metrics.")
                    pass
                continue
            
            ordered_data = []
            ordered_data.append(volume_str)
            ordered_data.extend(remaining_metrics[:6])
            
            if len(ordered_data) == 7:
                row_data = [current_location, activity]
                row_data.extend(ordered_data)
                
                if len(row_data) < len(COLUMNS):
                    row_data.extend([None] * (len(COLUMNS) - len(row_data)))
                    
                all_rows.append(row_data)

    if not all_rows:
        return pd.DataFrame(columns=COLUMNS)
    
    # Create DataFrame and apply type conversion
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    
    for col in COLUMNS[2:]:
        df[col] = df[col].astype(str).str.replace(r'[^0-9eE\.\+\-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("--- Custom Raw Text Parser Complete ---")
    return df

# --- END CORE PARSING DATA & UTILS ---


class PDFProcessor:
    """
    Handles downloading a PDF, finding tables, and saving them as TSVs.
    The primary extraction method uses unstructured.partition_pdf with a 
    fallback to the custom raw text parser.
    """
    def __init__(self, title, url, date):
        """
        Initializes the PDFProcessor.
        Args:
            title (str): The title of the document.
            url (str): The URL to the PDF file.
            date (datetime): The document date.
        """
        self.title = title
        self.url = url
        self.date = date
        self.sanitized_title = self._sanitize_filename(self.title)
        self.output_dir = Path.cwd() / str(self.date.year) / self.sanitized_title
        self.pdf_filename_base = os.path.splitext(self.sanitized_title)[0]
        self._create_output_dir()

    def _sanitize_filename(self, name):
        """Removes illegal characters from a string to make it a valid folder name."""
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        return name[:150]

    def _create_output_dir(self):
        """Creates the necessary output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_dataframe(self, df, table_index, suffix):
        """Saves the DataFrame to a TSV file in the output directory."""
        output_filename = f"{self.pdf_filename_base}_table_{table_index}{suffix}.tsv"
        output_path = self.output_dir / output_filename
        
        # Save to TSV
        df.to_csv(output_path, sep='\t', index=False)
        return output_path

    def process_document(self):
        """Main processing method for a single PDF document."""
        temp_pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name
            
            print(f"  Downloading from {self.url} using wget...")
            # Use 'time.sleep' before subprocess call to avoid race conditions on some filesystems
            time.sleep(1) 
            wget_command = ['wget', self.url, '-O', temp_pdf_path]
            
            # Subprocess runs wget to download the file
            subprocess.run(
                wget_command, check=True, timeout=300, capture_output=True
            )
            print(f"  Downloaded to temporary file: {temp_pdf_path}")
            
            self._extract_tables_unstructured(temp_pdf_path)

        except subprocess.CalledProcessError as e:
            print(f"  ERROR: wget failed to download PDF. Return code: {e.returncode}")
            # print(f"  wget stderr: {e.stderr.decode().strip()}")
        except subprocess.TimeoutExpired:
            print("  ERROR: wget download timed out after 300 seconds.")
        except FileNotFoundError:
            print("  ERROR: 'wget' command not found. Please ensure it is installed and in your PATH.")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred during processing. {type(e).__name__}: {e}")
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
                
    def _extract_tables_unstructured(self, pdf_path):
        """
        Extracts tables from the PDF using unstructured, with a fallback
        to the custom raw text parser when HTML tables are not found.
        """
        if partition_pdf is None:
            print("  CRITICAL ERROR: 'unstructured.partition_pdf' or its dependencies (like 'pytesseract', 'pdf2image') are not installed or configured. Skipping table extraction.")
            print("  Required packages: unstructured[all-docs], pandas, rich, pytesseract, pdf2image, Pillow.")
            return
            
        print("  Attempting to partition PDF using unstructured...")
        
        try:
            elements = partition_pdf(filename=pdf_path,
                                     skip_infer_table_types=False,
                                     strategy='hi_res',
                                     )
            tables = [el for el in elements if el.category == "Table"]

            if not tables:
                print("  No tables found in the PDF using unstructured.partition_pdf.")
                return

            for i, table_element in enumerate(tables):
                print(f"  Found {len(tables)} table(s). Parsing table {i+1}...")
                
                # Attempt 1: Get HTML content (preferred for structured data)
                html_content = table_element.metadata.text_as_html
                
                df = pd.DataFrame()
                
                if html_content:
                    print(f"    Table {i+1}: HTML content detected. Attempting pandas.read_html...")
                    try:
                        # Use StringIO to read HTML content directly
                        df_list = pd.read_html(StringIO(html_content))
                        df = df_list[0] if df_list else pd.DataFrame()
                        
                        if not df.empty:
                            output_filename = self._save_dataframe(df, i + 1, "_html")
                            print(f"    ✅ Successfully saved DataFrame from HTML to {output_filename}")
                            continue # Move to the next table element
                    except Exception as e:
                        print(f"    ⚠️ WARNING: HTML parsing failed for Table {i+1}. Error: {e}")
                
                # Attempt 2: FALLBACK PATH: Use Raw Text Parser (from Block 1)
                raw_text = table_element.text
                if raw_text:
                    print(f"    Table {i+1}: HTML extraction failed. Falling back to Custom Raw Text Parser.")
                    df = parse_raw_text_table(raw_text)

                    if not df.empty:
                        output_filename = self._save_dataframe(df, i + 1, "_fallback")
                        print(f"    ✅ Successfully saved DataFrame from RAW TEXT (FALLBACK) to {output_filename}")
                        print("    NOTE: The custom parser is tailored to a specific document format and may produce NaNs.")
                    else:
                        print(f"    ❌ FAILED: The custom raw text parser could not extract any structured rows for table {i+1}.")
                else:
                    print(f"    WARNING: Raw text output for table {i+1} was also empty.")

        except Exception as e:
            print(f"  A general error occurred during table processing: {type(e).__name__}: {e}")
            
# --- AdamsSearchNrc Class (No Functional Changes, only documentation updates) ---
class AdamsSearchNrc:
    """
    A class to automate searching and downloading reports from adams-search.nrc.gov,
    and then processing the downloaded PDFs for tables.
    """
    def __init__(self, geckodriver_path="/usr2/people/macgregor/drivers/geckodriver"):
        self.geckodriver_path = geckodriver_path
        self.temp_dir = tempfile.mkdtemp(prefix="nrc_downloads_")
        self.driver = None
        self.downloaded_csv_path = None
        self.dataframe = None
        print(f"Temporary download directory created at: {self.temp_dir}")

    def _setup_driver(self):
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.set_preference("browser.download.folderList", 2)
        firefox_options.set_preference("browser.download.dir", str(self.temp_dir) if self.temp_dir is not None else "")
        firefox_options.set_preference("browser.download.useDownloadDir", True)
        firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
        firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        service = FirefoxService(executable_path=self.geckodriver_path)
        self.driver = webdriver.Firefox(service=service, options=firefox_options)

    def download_environmental_reports_csv(self):
        self._setup_driver()
        search_url = "https://adams-search.nrc.gov/results/%257B%2522keywords%2522%253A%2522%2522%252C%2522legacyLibFilter%2522%253Atrue%252C%2522mainLibFilter%2522%253Atrue%252C%2522any%2522%253A%255B%257B%2522propertyItem%2522%253A%2522%2522%252C%2522keywords%2522%253A%2522%2522%252C%2522startDate%2522%253A%2522%2522%252C%2522endDate%2522%253A%2522%2522%252C%2522dateOperator%2522%253A%2522between%2522%252C%2522textOperator%2522%253A%2522contains%2522%252C%2522isDate%2522%253Afalse%257D%255D%252C%2522all%2522%253A%255B%257B%2522propertyItem%2522%253A%2522d31d1c41-8cad-4e89-8ead-3a5977c97126%2522%252C%2522keywords%2522%253A%2522TEXT-ENVIRONMENTAL%2520REPORTS%2522%252C%2522startDate%2522%253A%2522%2522%252C%2522endDate%2522%253A%2S22%252C%2522dateOperator%2522%253A%2522between%2522%252C%2522textOperator%2522%253A%2522contains%2522%252C%2522isDate%2522%253Afalse%257D%255D%257D"
        try:
            if self.driver is None:
                raise RuntimeError("WebDriver is not initialized. Call _setup_driver() before using it.")
            wait = WebDriverWait(self.driver, 60)
            print("Navigating to the search results page...")
            self.driver.get(search_url)

            print("Waiting for search results to load and clicking 'Select All'...")
            
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'tbody')))
            
            wait.until(EC.element_to_be_clickable((By.XPATH, '//input[@aria-label="Select All Checkbox"]'))).click()
            print("'Select All' checkbox clicked.")

            print("Clicking the main 'Report' button...")
            wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Report")]'))).click()
            print("Main 'Report' button clicked.")

            print("Waiting for download modal to appear...")
            wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@role='dialog']")))
            print("Download modal is visible.")
            time.sleep(1)

            print("Selecting 'CSV'...")
            wait.until(EC.element_to_be_clickable((By.XPATH, '//p-radiobutton[@inputid="csv"]'))).click()
            print("'CSV' option selected.")

            print("Clicking 'Create Report' to start the download...")
            wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Create Report")]'))).click()
            print("Download initiated.")

            print("Waiting for download to complete (up to 120 seconds)...")
            download_wait_timeout = 120
            start_time = time.time()
            file_found = False
            while time.time() - start_time < download_wait_timeout:
                if self.temp_dir is not None:
                    downloaded_files = list(Path(self.temp_dir).glob('*.csv'))
                else:
                    downloaded_files = []
                if downloaded_files and not str(downloaded_files[0]).endswith('.part'):
                    self.downloaded_csv_path = downloaded_files[0]
                    print(f"SUCCESS: File downloaded to {self.downloaded_csv_path}")
                    file_found = True
                    break
                time.sleep(1)

            if not file_found:
                print("WARNING: Download timed out or did not complete. No CSV file found.")
        except Exception as e:
            print("An error occurred. Saving debug information...")
            debug_path = Path.cwd()
            if self.driver is not None:
                self.driver.save_screenshot(str(debug_path / "debug_screenshot.png"))
                with open(debug_path / "debug_page_source.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"Debug files saved to: {debug_path}")
            else:
                print("WebDriver is not initialized; cannot save screenshot or page source.")
            print(f"\nError details: {e}")
        finally:
            self._close_driver()

    def load_csv_to_dataframe(self):
        if not self.downloaded_csv_path or not self.downloaded_csv_path.exists():
            print("Error: CSV file not found.")
            return False
        try:
            print(f"Loading CSV data from {self.downloaded_csv_path} into a DataFrame...")
            self.dataframe = pd.read_csv(self.downloaded_csv_path, on_bad_lines='skip')
            print("DataFrame loaded successfully.")
            return True
        except Exception as e:
            print(f"An error occurred while loading the CSV into a DataFrame: {e}")
            return False

    def process_and_extract_tables(self):
        """
        Processes the dataframe to select, sort, and then extract tables from each PDF.
        """
        if self.dataframe is None:
            print("DataFrame not loaded. Please run load_csv_to_dataframe() first.")
            return
            
        required_columns = ['DocumentTitle', 'Url', 'DocumentDate']
        if not all(col in self.dataframe.columns for col in required_columns):
            print(f"Error: The CSV is missing one or more required columns: {required_columns}")
            return

        df = self.dataframe.copy()
        df['DocumentDate'] = pd.to_datetime(df['DocumentDate'], errors='coerce')
        df = df.dropna(subset=['DocumentTitle', 'DocumentDate'])

        print("Filtering documents to include only those from 1995 onwards...")
        original_count = len(df)
        df = df[df['DocumentDate'].dt.year >= 1995]
        print(f"Removed {original_count - len(df)} documents from before 1995.")

        df = df.sort_values(by='DocumentDate').reset_index(drop=True)
        
        print(f"\nFound {len(df)} documents to process from the report.")

        for idx, (index, row) in track(enumerate(df.iterrows()), description="Processing PDF documents..."):
            print(f"\n[Row {idx + 1}/{len(df)}] Starting processing for: {row['DocumentTitle']}")
            
            url_pattern = "https://www.nrc.gov/docs"
            url = row.get('Url')
            found_url = False

            if isinstance(url, str) and url.startswith(url_pattern):
                found_url = True
            
            if not found_url:
                print(f"  URL in 'Url' column is invalid or missing: '{url}'. Searching the entire row.")
                for value in row:
                    if isinstance(value, str) and value.startswith(url_pattern):
                        url = value
                        found_url = True
                        print(f"  Found a valid URL in another column: {url}")
                        break
            
            if not found_url:
                print(f"  Skipping row: No URL starting with '{url_pattern}' could be found.")
                continue
            
            processor = PDFProcessor(
                title=row['DocumentTitle'],
                url=url,
                date=row['DocumentDate']
            )
            processor.process_document()

    def _close_driver(self):
        if self.driver:
            print("Closing the browser.")
            self.driver.quit()

    def cleanup(self):
        """Removes the temporary download directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


if __name__ == "__main__":
    searcher = AdamsSearchNrc()
    try:
        searcher.download_environmental_reports_csv()
        
        if not searcher.downloaded_csv_path:
            print("--- Web download failed. Checking for local fallback file. ---")
            fallback_path = Path("/usr2/people/macgregor/amplicon/workflow_16s/resources/adams_search_202509.csv")
            if fallback_path.exists():
                print(f"--- Found local fallback file: {fallback_path} ---")
                searcher.downloaded_csv_path = fallback_path
            else:
                print(f"--- Local fallback file not found at {fallback_path}. ---")

        if searcher.downloaded_csv_path:
            print("\n--- Starting Data Processing ---")
            if searcher.load_csv_to_dataframe():
                print("--- DataFrame loaded. Starting PDF extraction. This may take a long time. ---")
                searcher.process_and_extract_tables()
                print("\n--- All documents processed. ---")
            else:
                print("--- Halting due to failure in loading DataFrame. ---")
        else:
            print("--- Halting because CSV download failed and no fallback was available. ---")
    finally:
        searcher.cleanup()
