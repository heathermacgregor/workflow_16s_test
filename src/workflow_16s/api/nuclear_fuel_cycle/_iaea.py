# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_nfcis.py
# ==================================================================================== #

# Standard Imports
import glob
import logging
import os
import time
from pathlib import Path
from typing import Optional

# Third Party Imports
import pandas as pd

# Selenium imports wrapped to prevent import errors if not installed
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

class NFCFDB:
    """A class that handles the IAEA NFCIS NFCFDB."""
    
    DEFAULT_NFCIS_COLUMNS = {
        'facility_country': "Country",
        'facility': "Facility Name",
        'facility_type': "Facility Type",
        'facility_capacity': "Design Capacity",
        'facility_status': "Facility Status",
        'facility_start_year': "Start of Operation",
        'facility_end_year': "End of Operation"
    }
    
    URL = "https://infcis.iaea.org/NFCFDB/facilities"

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: The directory where the NFCIS file will be stored.
        """
        if output_dir: self.output_dir = Path(output_dir)
        else: self.output_dir = Path.cwd() / "nfc_facilities"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_dir / "iaea.xlsx"
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """Loads the NFCIS Excel file into a DataFrame."""
        # 1. Check if file exists
        if not self.file_path.exists():
            logger.info("NFCIS file not found. Attempting download...")
            try:
                self._download()
            except Exception as e:
                logger.error(f"Failed to download NFCIS data: {e}")
                return pd.DataFrame()

        # 2. Check again after download attempt
        if not self.file_path.exists():
            logger.warning(f"NFCIS file missing at {self.file_path}. Skipping NFCIS data.")
            return pd.DataFrame()
            
        # 3. Load and Parse
        try:
            # Load raw without header first to find the correct row
            df_raw = pd.read_excel(self.file_path, header=None)
            
            # Dynamic Header Detection: Find the row that contains "Facility Name"
            header_row_idx = None
            for idx, row in df_raw.iterrows():
                # Check if "Facility Name" and "Country" are in this row's values
                row_vals = [str(x).strip() for x in row.values]
                if "Facility Name" in row_vals and "Country" in row_vals:
                    header_row_idx = idx
                    break
            
            if header_row_idx is not None:
                # Reload with correct header
                df = pd.read_excel(self.file_path, header=header_row_idx)
            else:
                # Fallback to old hardcoded skip if detection fails
                df = pd.read_excel(self.file_path, header=0, skiprows=8)

            # Standardize columns
            # Keep only columns we map, rename them to internal names
            rename_map = {v: k for k, v in self.DEFAULT_NFCIS_COLUMNS.items()}
            
            # Filter for columns that actually exist in the file
            valid_cols = [c for c in self.DEFAULT_NFCIS_COLUMNS.values() if c in df.columns]
            df = df[valid_cols]
            df = df.rename(columns=rename_map)
            
            df['data_source'] = "NFCIS"
            
            # Normalize coordinates if they exist (rare in this sheet, but good practice)
            if 'latitude' not in df.columns: df['latitude'] = None
            if 'longitude' not in df.columns: df['longitude'] = None

            self.df = df
            logger.info(f"Successfully loaded NFCIS data with {self.df.shape[0]} facilities.")
            return self.df

        except Exception as e:
            logger.error(f"Error parsing NFCIS Excel data: {e}")
            return pd.DataFrame()

    def _download(self, headless: bool = True, timeout: int = 90):
        """Downloads the NFCIS facilities spreadsheet using Selenium."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not installed. Cannot scrape NFCIS.")

        chrome_opts = Options()
        prefs = {"download.default_directory": os.path.abspath(self.output_dir)}
        chrome_opts.add_experimental_option("prefs", prefs)
        
        # --- CRITICAL SERVER SETTINGS ---
        if headless: 
            chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--window-size=1920,1080")
        chrome_opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        # --------------------------------

        driver = None
        try:
            logger.info("Initializing Headless Chrome for NFCIS download...")
            driver = webdriver.Chrome(options=chrome_opts)
            driver.get(self.URL)

            # Wait for the download button (Increased robustness)
            wait = WebDriverWait(driver, 30)
            
            # Try multiple selector strategies for the button
            button_selectors = [
                "//button[contains(., 'Download Spreadsheet')]",
                "//button[contains(@class, 'export')]", 
                "//span[contains(text(), 'Download')]/.."
            ]
            
            download_btn = None
            for selector in button_selectors:
                try:
                    download_btn = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                    break
                except: continue
            
            if not download_btn:
                raise Exception("Could not find Download button on NFCIS page.")

            download_btn.click()
            logger.info("Clicked download button. Waiting for file...")

            # Wait for the download to complete
            end_time = time.time() + timeout
            while time.time() < end_time:
                # Find most recent xlsx file
                files = glob.glob(str(self.output_dir / "*.xlsx"))
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    # Ensure it's not a partial download (.crdownload)
                    if not latest_file.endswith(".crdownload"):
                        # Check if it's the file we just downloaded (created recently)
                        if os.path.getctime(latest_file) > (time.time() - timeout):
                            Path(latest_file).rename(self.file_path)
                            logger.info(f"Download complete: {self.file_path}")
                            return
                time.sleep(2)

            raise TimeoutError("NFCIS download timed out.")

        except Exception as e:
            # Clean logging of the error without crashing the script
            logger.warning(f"NFCIS Scraper skipped: {e}")
            raise e
            
        finally:
            if driver: 
                try: driver.quit()
                except: pass