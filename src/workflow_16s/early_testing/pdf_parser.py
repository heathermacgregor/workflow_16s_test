# NRC ADAMS Search and CSV Downloader
# This script automates searching for specific documents on adams-search.nrc.gov,
# selecting all results, downloading them as a CSV file, and then processing
# each document to extract tables from PDFs.

# Required packages:
# You must install selenium, pandas, rich, pdfplumber, Pillow, and requests before running this script.
# You can install them using pip:
# pip install selenium pandas rich pdfplumber Pillow requests

import time
import tempfile
from pathlib import Path
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import re
import requests
import pdfplumber
from PIL import Image
from rich.progress import track
import shutil
import csv
import tempfile

def load_fixed_csv(path, expected_cols=None):
    # Step 1: Peek header to determine expected column count
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if expected_cols is None:
            expected_cols = len(header)

    # Step 2: Create a cleaned temp file
    tmpfile = tempfile.NamedTemporaryFile(mode="w+", newline="", delete=False, encoding="utf-8")
    writer = csv.writer(tmpfile)
    writer.writerow(header)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < expected_cols:
                # pad missing fields with empty strings
                row.extend([""] * (expected_cols - len(row)))
            elif len(row) > expected_cols:
                # merge overflow into last column
                row = row[:expected_cols-1] + [",".join(row[expected_cols-1:])]
            writer.writerow(row)

    tmpfile.flush()
    tmpfile.close()

    # Step 3: Load into pandas safely
    return pd.read_csv(tmpfile.name)


class PDFProcessor:
    """
    Handles downloading a PDF, finding tables, and saving them as images and CSVs.
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
        self._create_output_dir()

    def _sanitize_filename(self, name):
        """Removes illegal characters from a string to make it a valid folder name."""
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        return name[:150]  # Limit length to avoid issues with long paths

    def _create_output_dir(self):
        """Creates the necessary output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_document(self):
        """Main processing method for a single PDF document."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name
            try:
                # 1. Download the PDF
                print(f"  Downloading from {self.url}...")
                response = requests.get(self.url, stream=True, timeout=60)
                response.raise_for_status()  # Raise an exception for bad status codes
                for chunk in response.iter_content(chunk_size=8192):
                    temp_pdf.write(chunk)
                print(f"  Downloaded to temporary file: {temp_pdf_path}")

                # 2. Find and extract tables
                self._find_and_extract_tables(temp_pdf_path)

            except requests.exceptions.RequestException as e:
                print(f"  ERROR: Failed to download PDF. {e}")
            except Exception as e:
                print(f"  ERROR: An unexpected error occurred during processing. {e}")
            finally:
                # Ensure the temporary file is removed
                os.unlink(temp_pdf_path)

    def _find_and_extract_tables(self, pdf_path):
        """Opens a PDF and extracts all found tables."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"  PDF has {len(pdf.pages)} pages. Searching for tables...")
                found_tables = False
                for i, page in enumerate(pdf.pages):
                    # Extract tables from the page using pdfplumber's capabilities
                    tables = page.find_tables()
                    if tables:
                        found_tables = True
                        print(f"    Found {len(tables)} table(s) on page {i + 1}.")
                        for j, table in enumerate(tables):
                            # a. Save table data as a CSV file
                            table_data = table.extract()
                            if table_data and len(table_data) > 1:
                                csv_path = self.output_dir / f"page_{i + 1}_table_{j + 1}.csv"
                                pd.DataFrame(table_data[1:], columns=table_data[0]).to_csv(csv_path, index=False)
                                print(f"      - Saved table data to {csv_path}")

                            # b. Crop the table from a page image and save as PNG
                            img = page.to_image(resolution=150)
                            bbox = (table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3])
                            cropped_img = img.original.crop(bbox)
                            png_path = self.output_dir / f"page_{i + 1}_table_{j + 1}.png"
                            cropped_img.save(png_path)
                            print(f"      - Saved table image to {png_path}")
                if not found_tables:
                    print("  No tables were automatically detected in this document.")
        except Exception as e:
            print(f"  ERROR: Could not process PDF file '{pdf_path}'. It might be corrupted or unreadable. Details: {e}")

class AdamsSearchNrc:
    """
    A class to automate searching and downloading reports from adams-search.nrc.gov.
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
        firefox_options.set_preference("browser.download.dir", str(self.temp_dir))
        firefox_options.set_preference("browser.download.useDownloadDir", True)
        firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
        firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        service = FirefoxService(executable_path=self.geckodriver_path)
        self.driver = webdriver.Firefox(service=service, options=firefox_options)

    def download_environmental_reports_csv(self):
        self._setup_driver()
        if self.driver is None:
            raise RuntimeError("WebDriver was not initialized properly.")
        search_url = "https://adams-search.nrc.gov/results/%257B%2522keywords%2522%253A%2522%2522%252C%2522legacyLibFilter%2522%253Atrue%252C%2522mainLibFilter%2522%253Atrue%252C%2522any%2522%253A%255B%257B%2522propertyItem%2522%253A%2522%2522%252C%2522keywords%2522%253A%2522%2522%252C%2522startDate%2522%253A%2522%2522%252C%2522endDate%2522%253A%2522%2522%252C%2522dateOperator%2522%253A%2522between%2522%252C%2522textOperator%2522%253A%2522contains%2522%252C%2522isDate%2522%253Afalse%257D%255D%252C%2522all%2522%253A%255B%257B%2522propertyItem%2522%253A%2522d31d1c41-8cad-4e89-8ead-3a5977c97126%2522%252C%2522keywords%2522%253A%2522TEXT-ENVIRONMENTAL%2520REPORTS%2522%252C%2522startDate%2522%253A%2522%2522%252C%2522endDate%2522%253A%2522%2522%252C%2522dateOperator%2522%253A%2522between%2522%252C%2522textOperator%2522%253A%2522contains%2522%252C%2522isDate%2522%253Afalse%257D%255D%257D"
        try:
            wait = WebDriverWait(self.driver, 30)
            print("Navigating to the search results page...")
            self.driver.get(search_url)

            print("Waiting for search results to load and clicking 'Select All'...")
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
                if self.temp_dir is None:
                    break
                downloaded_files = list(Path(self.temp_dir).glob('*.csv'))
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
            self.driver.save_screenshot(str(debug_path / "debug_screenshot.png"))
            with open(debug_path / "debug_page_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print(f"Debug files saved to: {debug_path}")
            print(f"\nError details: {e}")
        finally:
            self._close_driver()

    def load_csv_to_dataframe(self):
        if not self.downloaded_csv_path or not self.downloaded_csv_path.exists():
            print("Error: CSV file not found. Please run download_environmental_reports_csv() first.")
            return False
        try:
            print(f"Loading CSV data from {self.downloaded_csv_path} into a DataFrame...")
            
            # Use pandas to read the CSV. The 'on_bad_lines' parameter is set to 'skip'
            # to automatically ignore rows that have an incorrect number of columns.
            # This is handled by the default 'c' parsing engine and is the most
            # efficient way to handle these errors.
            self.dataframe = load_fixed_csv(self.downloaded_csv_path)
            
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

        df = self.dataframe[['DocumentTitle', 'Url', 'DocumentDate']].dropna().copy()
        df['DocumentDate'] = pd.to_datetime(df['DocumentDate'], errors='coerce')
        df = df.dropna(subset=['DocumentDate'])
        df = df.sort_values(by='DocumentDate').reset_index(drop=True)
        
        print(f"\nFound {len(df)} documents to process from the report.")

        for idx, (index, row) in enumerate(track(df.iterrows(), description="Processing PDF documents...")):
            print(f"\n[Row {idx + 1}/{len(df)}] Starting processing for: {row['DocumentTitle']}")
            # Add a check to ensure the URL is a valid-looking string before processing
            if not isinstance(row['Url'], str) or not row['Url'].lower().startswith('http'):
                print(f"  Skipping row due to invalid URL: {row['Url']}")
                continue
            
            processor = PDFProcessor(
                title=row['DocumentTitle'],
                url=row['Url'],
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
        
        if searcher.downloaded_csv_path:
            print("\n--- Starting Data Processing ---")
            if searcher.load_csv_to_dataframe():
                print("--- DataFrame loaded. Starting PDF extraction. This may take a long time. ---")
                searcher.process_and_extract_tables()
                print("\n--- All documents processed. ---")
            else:
                print("--- Halting due to failure in loading DataFrame. ---")
        else:
            print("--- Halting because CSV download failed or was skipped. ---")
    finally:
        searcher.cleanup()

