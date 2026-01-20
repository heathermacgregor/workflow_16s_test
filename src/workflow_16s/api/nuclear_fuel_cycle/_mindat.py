# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_mindat.py
# ==================================================================================== #

# Standard Imports
import contextlib
import logging
import os
import requests
import sys
from pathlib import Path
from typing import Optional, Union

# Third Party Imports
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pycountry
from bs4 import BeautifulSoup
from openmindat import LocalitiesRetriever

# Local Imports
from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.progress import get_progress_bar

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old___stdout__ = sys.__stdout__
        old___stderr__ = sys.__stderr__
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            sys.__stdout__ = devnull
            sys.__stderr__ = devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.__stdout__ = old___stdout__
            sys.__stderr__ = old___stderr__
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

# ==================================================================================== #

def gpd_from_df(df):
    """Create a GeoDataFrame from a DataFrame with latitude/longitude coordinates.
    
    Robustly handles both 'latitude'/'longitude' and 'lat'/'lon' column names.

    Args:
        df: Input DataFrame containing coordinate columns.

    Returns:
        GeoDataFrame with geometry points created from coordinate columns.
    """
    # Detect coordinate columns
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    
    # Check if columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        # If coordinates are missing, return empty GDF with correct structure
        return gpd.GeoDataFrame(
            df, 
            geometry=[],
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

# ==================================================================================== #

class MinDatScraper:
    """Scraper for retrieving locality data from Mindat website.
    
    Attributes:
        MinDatLocalitiesURL: URL for Mindat's country list page.
        session:            Persistent HTTP session.
        localities:         List of locality names extracted from Mindat.
    """
    
    MinDatLocalitiesURL = 'https://www.mindat.org/countrylist.php'
    
    def __init__(self, user_agent):
        """Initialize scraper and fetch localities."""
        self.user_agent = user_agent
        self._get_session()
        self._get_localities()
      
    def _get_session(self) -> None:
        """Configure HTTP session with custom User-Agent."""
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
      
    def _get_soup(self, url):
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')

    def _get_mindattable(self, soup):
        """Retrieve and parse HTML content from URL."""
        return soup.find('table', class_='mindattable')
      
    def _get_localities(self):
        """Extract mindattable element from parsed HTML."""
        soup = self._get_soup(self.MinDatLocalitiesURL)
        table = self._get_mindattable(soup)
        localities = []
        if table:
            for row in table.find_all('tr')[1:]:  # type: ignore # Skip the header
                cols = row.find_all('td') # type: ignore
                if cols:
                    a_tag = cols[0].find('a') # type: ignore
                    if a_tag:
                        locality_name = a_tag.get_text(strip=True) # type: ignore
                        localities.append(locality_name)
        self.localities = localities


class MinDatAPI:
    """API client for retrieving and processing uranium mine data from MinDat.
    
    Attributes:
        MPLWorldURL:  URL for Natural Earth countries dataset.
        output_path:  Output file path.
        plot_package: Visualization package to use ('mpl' for matplotlib).
        localities:   List of available localities.
        verbose:      Verbosity flag.
    """
    MPLWorldURL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    
    def __init__(self, api_key: str, user_agent: str, output_dir: Optional[Union[str, Path]] = None,
                 plot_package: str = 'mpl', verbose: bool = False):
        """Initialize Mindat API client.
        
        Args:
            api_key:      Mindat API key for authentication.
            user_agent:   Custom User-Agent string for HTTP requests.
            verbose:      Enable verbose logging.
            output_dir:   The directory where the NFCIS file will be stored.
                          A default directory './nfc_facilities/' will be used if None.
            plot_package: Visualization package to use. Currently supports 'mpl'.
        """
        if not api_key: raise ValueError("API key must be provided")
        os.environ["MINDAT_API_KEY"] = api_key
        
        self.user_agent = user_agent
        self.verbose = verbose
        
        if output_dir: self.output_dir = Path(output_dir)
        else: self.output_dir = Path.cwd() / "nfc_facilities"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "mindat.tsv"
        
        self.plot_package = plot_package
      
        try: self.localities = self._get_mindat_localities()
        except Exception as e: self.log(f"Error getting Mindat localities: {e}")
        
        # Ensure we have localities
        if not self.localities:
            self.log("No localities found. Using pycountry fallback.")
            self.localities = self._get_pycountry_countries()
            
    def log(self, msg):
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)
        
    def _get_mindat_localities(self):
        """Retrieve localities from Mindat website."""
        scraper = MinDatScraper(self.user_agent)
        return scraper.localities

    def _get_pycountry_countries(self):
        """Fallback method to get country names from pycountry."""
        return [country.name for country in pycountry.countries] # type: ignore
    
    def _get_uranium_mines_locality(self, locality: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Retrieve uranium mines data for a specific locality."""
        lr = LocalitiesRetriever()
        lr.country(locality).description("mine").elements_inc("U")
        # Suppress CLI output from openmindat
        with suppress_output(): # type: ignore
            logging.disable(logging.CRITICAL)
            results = lr.get_dict()
            logging.disable(logging.NOTSET)

        if 'results' in results and results['results']:
            df = pd.DataFrame(results['results'])
            df['facility'] = df['txt'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
            df['data_source'] = "MinDat"
            return df, gpd_from_df(df)
        else:
            return pd.DataFrame(), gpd.GeoDataFrame()

    def _mpl_plot_uranium_mines_locality(self, locality: str, gdf: gpd.GeoDataFrame) -> None:
        """Generate matplotlib visualization of uranium mines.
        
        Args:
            locality: Name of locality being plotted.
            gdf:      GeoDataFrame containing mine coordinates.
        """
        if gdf.empty:
            self.log(f"No data to plot for {locality}")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        world_url = self.MPLWorldURL
        world = gpd.read_file(world_url)
        world.plot(ax=ax, color='lightgrey', edgecolor='white')
        gdf.plot(ax=ax, color='red', markersize=5)
        plt.title(f"{locality.capitalize()} Uranium Mines on World Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(alpha=0.3)
        plt.savefig(Path(self.output_path).parent / f'{locality}_mines_map.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _filter_data(self, df):
        # Split revtxtd into loc columns, handling NaN values
        locs_lod = [
            {f"loc_{i}": loc for i, loc in enumerate(str(row).split(','))} if pd.notna(row) else {}
            for row in df['revtxtd']
        ]
        df['country'] = df['country'].str.replace('USA', 'United States of America')
        df['country'] = df['country'].str.replace('UK', 'United Kingdom')
        # Count number of locations (handle NaN and ensure integer type)
        df['n_locs'] = df['revtxtd'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        # Check if number of locations > 3 (convert to integer for comparison)
        df['keep'] = df['n_locs'].apply(lambda x: int(x) > 3 if pd.notna(x) else False)
        # Check if 'mine' is in the text (convert to string and use lower())
        df['is_mine'] = df['txt'].apply(lambda x: 'mine' in str(x).lower() if pd.notna(x) else False)
        df['is_uranium_txt'] = df['txt'].apply(lambda x: 'uranium' in str(x).lower() if pd.notna(x) else False)
        df['is_uranium_desc'] = df['description_short'].apply(lambda x: 'uranium' in str(x).lower() if pd.notna(x) else False)
        df['is_u_txt'] = df['txt'].apply(lambda x: ' u ' in str(x).lower() if pd.notna(x) else False)
        df['is_u_desc'] = df['description_short'].apply(lambda x: ' u ' in str(x).lower() if pd.notna(x) else False)
        
        # Create DataFrame and align indices
        locs_df = pd.DataFrame(locs_lod, index=df.index)
        
        # Merge loc columns into df
        df = pd.concat([df, locs_df], axis=1)
        
        # Filter rows based on conditions
        # Keep rows where 'keep' and 'is_mine' are True AND either 'is_uranium_txt' or 'is_uranium_desc' is True
        filter_condition = (df['keep'] & df['is_mine'] & 
                            (df['is_uranium_txt'] | df['is_uranium_desc']) | df['is_u_txt'] | df['is_u_desc'])
        df = df[filter_condition]

        # Drop rows where BOTH latitude AND longitude are 0
        df = df[~((df['latitude'] == 0) & (df['longitude'] == 0))].reset_index(drop=True)
        
        # Rename to lat/lon for consistency with rest of workflow
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        
        return df
        
    def _get_uranium_mines_world(self) -> pd.DataFrame:
        """Retrieve and process uranium mines data for all available localities.
        
        Returns:
            pd.DataFrame: Combined DataFrame of all mine data
        """
        dfs = []
        with get_progress_bar() as progress:
            task = progress.add_task("Processing localities", total=len(self.localities))
            for locality in self.localities: 
                self.log(f"Processing {locality}")
                try:
                    df, gdf = self._get_uranium_mines_locality(locality)
                    if not df.empty: dfs.append(df)
                    else: self.log(f"No uranium mines found in {locality}")
                except Exception as e: self.log(f"Error with {locality}: {e}")
                finally: progress.update(task, advance=1)
        
        if dfs:
            df = pd.concat(dfs, axis=0)
            df['facility'] = [i.split(',')[0] for i in df['txt']]
            df = self._filter_data(df)
            df.to_csv(self.output_path, sep="\t", index=False)
            
            # Re-create GDF with renamed columns (lat/lon)
            gdf = gpd_from_df(df)
            
            if not gdf.empty and self.plot_package == 'mpl':
                self._mpl_plot_uranium_mines_locality('world', gdf)
            return df
        else:
            self.log("No data found for any locality")
            return pd.DataFrame()

# ==================================================================================== #

def world_uranium_mines(config: AppConfig):
    """Main function to retrieve worldwide uranium mines data.
    
    Args:
        config: Application configuration containing credentials and settings.
        
    Returns:
        Tuple containing combined DataFrame and GeoDataFrame of uranium mines.
    """
    api_key = config.credentials.mindat_api_key
    if not api_key:
        raise ValueError("Mindat API key must be provided in configuration")
    use_local = config.nfc_facilities.use_cache
    verbose = config.verbose
    user_agent = config.web.user_agent
    # Setup project directory
    project_dir = Project(config)
    if project_dir:
        output_dir = project_dir.raw_data / "_nfc_facilities" 
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "mindat.tsv"
        if use_local and output_path.exists():
            logger.info(f"Using local MinDat data from {output_path}")
            df = pd.read_csv(output_path, sep='\t')
            # Ensure required columns exist
            for col in ["longitude", "latitude", "country"]:
                if col not in df.columns:
                    # Check if lat/lon exist instead
                    if col == 'latitude' and 'lat' in df.columns:
                        df['latitude'] = df['lat']
                    elif col == 'longitude' and 'lon' in df.columns:
                        df['longitude'] = df['lon']
                    else:
                        df[col] = None
            return df
    else:
        output_path = None
        
    mindat_api = MinDatAPI(api_key, user_agent, output_path, verbose=verbose)
    df = mindat_api._get_uranium_mines_world()
    
    # Ensure required columns exist for downstream consumption
    for col in ["longitude", "latitude", "country"]:
        if col not in df.columns:
            # If we have lat/lon from _filter_data, alias them to latitude/longitude
            if col == 'latitude' and 'lat' in df.columns:
                df['latitude'] = df['lat']
            elif col == 'longitude' and 'lon' in df.columns:
                df['longitude'] = df['lon']
            else:
                df[col] = None
    return df