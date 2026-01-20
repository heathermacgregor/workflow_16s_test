# ==================================================================================== #

# Standard Imports
import glob
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third Party Imports
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Local Imports
from workflow_16s.constants import (
    DEFAULT_GEM_PATH, DEFAULT_NFCIS_PATH, REFERENCES_DIR
)

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

DEFAULT_GEM_COLUMNS = {
    'country': "Country/Area",
    'facility': "Project Name",
    'facility_type': "Reactor Type",
    'facility_capacity': " Capacity (MW) ",
    'facility_status': "Status",
    'facility_start_year': "Start Year",
    'facility_end_year': "Retirement Year"
}

DEFAULT_NFCIS_COLUMNS = {
    'country': "Country",
    'facility': "Facility Name",
    'facility_type': "Facility Type",
    'facility_capacity': "Design Capacity",
    'facility_status': "Facility Status",
    'facility_start_year': "Start of Operation",
    'facility_end_year': "End of Operation"
}

# ==================================================================================== #

def download_nfcis(
    download_dir: Union[str, Path] = REFERENCES_DIR / 'nfc_facilities', 
    headless: bool = True, 
    timeout: int = 60
):
    """Downloads the NFCFDB facilities spreadsheet from IAEA and loads it into a 
    Pandas DataFrame.

    Args:
        download_dir: Directory where the file will be downloaded (default: 
                      REFERENCES_DIR).
        headless:     If True, runs Chrome in headless mode (no GUI).
        timeout:      Maximum wait time (in seconds) for the download to finish.

    Returns:
        df: The downloaded spreadsheet loaded into a Pandas DataFrame.

    Raises:
        TimeoutError: If the file is not downloaded within the given timeout.
    """
    # Make sure the download directory exists
    download_dir.mkdir(parents=True, exist_ok=True)

    # Configure Chrome options
    chrome_opts = Options()
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_settings.popups": 0,
        "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,
    }
    chrome_opts.add_experimental_option("prefs", prefs)
    if headless:
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--window-size=1920x1080")

    # Launch Chrome WebDriver
    driver = webdriver.Chrome(options=chrome_opts)
    driver.get("https://infcis.iaea.org/NFCFDB/facilities")

    file_path = None
    try:
        # Wait for the download button to appear
        download_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, 
                                        "//button[contains(., 'Download Spreadsheet')]"))
        )
        download_btn.click()
        logger.debug("Clicked 'Download Spreadsheet' button...")

        # Wait until the file finishes downloading
        for _ in range(timeout):
            files = glob.glob(os.path.join(download_dir, "*"))
            if files:
                latest_file = max(files, key=os.path.getctime)
                if not latest_file.endswith(".crdownload"):  # Ensure fully downloaded
                    file_path = latest_file
                    break
            time.sleep(1)

        if not file_path:
            raise TimeoutError("Download failed or timed out.")

        logger.debug(f"Downloaded file: {file_path}")

    finally:
        driver.quit()

    return file_path

# ==================================================================================== #

class NFCFacilityDB:
    """A class that handles loading and processing of NFC facility databases.
    
    Attributes:
        databases:  List of database names to process.
        output_dir: Directory where processed results will be saved.
        result:     Combined DataFrame containing processed database information.
    """
    DBConfig = {
        "GEM": (0, False, DEFAULT_GEM_COLUMNS, DEFAULT_GEM_PATH),
        "NFCIS": (8, True, DEFAULT_NFCIS_COLUMNS, DEFAULT_NFCIS_PATH)
    }
    def __init__(
        self, 
        databases: List[str] = ["GEM", "NFCIS"], 
        output_dir: Optional[Union[str, Path]] = REFERENCES_DIR
    ):
        """Initialize NFC facility database processor.
        
        Args:
            databases:  List of database names to process. Defaults to ["GEM", "NFCIS"].
            output_dir: Output directory for processed results. Defaults to REFERENCES_DIR.
        """
        self.databases = databases
        self.database_names = [db['name'] for db in self.databases 
                               if db['name'] in list(self.DBConfig.keys())]
        self.output_dir = output_dir
        self.result = None

    def run(self):
        dfs = self._process_dbs()
        self.result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if self.output_dir:
            tsv_path = Path(self.output_dir) / f"{'_'.join(valid_dbs)}.tsv"
            self.result.to_csv(tsv_path, sep='\t', index=False)
        return self.result
        
    def _process_dbs(self):
        dfs = []
        for name in self.database_names:
            file_path = self.DBConfig[name][3]
            if Path(file_path).exists():
                df = self._load_df_from_file(name)
            else:
                if name == "NFCIS":
                    file_path = self._download_nfcis()
                    df = self._load_df_from_file(name)
                    
            # Drop first column if needed
            df = df.iloc[:, 1:] if self.DBConfig[name][1] else df.copy()
            # Set header and reset
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            logger.info(f"Loaded '{name}' data with {df.shape[0]} NFC facilities")
            # Filter and rename
            df = df[list(self.DBConfig[name][2].values())]
            df = df.rename(columns={v: k for k, v in self.DBConfig[name][2].items()})
            df = df[list(self.DBConfig[name][2].keys())]
            df['data_source'] = name
            dfs.append(df)
        return dfs

    def _load_df_from_file(self, name):
        skip_rows, skip_first_col, use_cols, file_path = self.DBConfig[name]
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            try:
                return pd.read_excel(file_path, header=None, skiprows=skip_rows, usecols=use_cols)
            except Exception as e:
                return pd.read_csv(file_path, sep='\t', header=None, skiprows=skip_rows, 
                                   encoding_errors='replace')
        else:
            try:
                return pd.read_csv(file_path, sep='\t', header=None, skiprows=skip_rows, 
                                   usecols=use_cols, encoding_errors='replace')
            except Exception as e:
                return pd.read_csv(file_path, sep='\t', header=None, skiprows=skip_rows, 
                                   encoding_errors='replace')
        
    def _download_nfcis(self, download_dir: Union[str, Path] = REFERENCES_DIR / 'nfc_facilities'):
        df_raw, _ = download_nfcis(download_dir=download_dir)
        return df_raw        

# ==================================================================================== #

def load_nfc_facilities(
    config: Dict, 
    output_dir: Optional[Union[str, Path]] = REFERENCES_DIR
) -> pd.DataFrame:
    """Load NFC facilities from configured databases.
    
    Args:
        config:     Configuration dictionary containing database settings.
        output_dir: Directory where processed results will be saved. Defaults to REFERENCES_DIR.
        
    Returns:
        DataFrame containing combined facilities from all configured databases.
        
    Note:
        Can use locally cached version if configured and available.
    """
    DBConfig = {
        "GEM": (0, False, DEFAULT_GEM_COLUMNS, DEFAULT_GEM_PATH),
        "NFCIS": (8, True, DEFAULT_NFCIS_COLUMNS, DEFAULT_NFCIS_PATH)
    }
    databases = config.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
    db_names = [db['name'] for db in databases]
    valid_dbs = [database for database in db_names if database in list(DBConfig.keys())]
    use_local = config.get("nfc_facilities", {}).get('use_local', False)
    if output_dir:
        tsv_path = Path(output_dir) / f"{'_'.join(valid_dbs)}.tsv"
    if use_local and tsv_path.exists():
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        db_loader = NFCFacilityDB(databases=databases, output_dir=output_dir)
        df = db_loader.run()
    
    logger.info(f"NFC facilities from databases ({', '.join(valid_dbs)}): {df.shape}")
    return df
  
