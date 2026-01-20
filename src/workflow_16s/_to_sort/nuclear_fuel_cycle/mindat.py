# ==================================================================================== #

# Standard Imports
import contextlib
import io
import logging
import os
import requests
from pathlib import Path
from typing import Dict, Optional, Union

# Third Party Imports
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pycountry
from bs4 import BeautifulSoup
from openmindat import LocalitiesRetriever

# Local Imports
from workflow_16s.logger import get_logger

# ==================================================================================== #

logger = get_logger() #logging.getLogger("workflow_16s")

# ==================================================================================== #

def gpd_from_df(df):
    """Create a GeoDataFrame from a DataFrame with latitude/longitude coordinates.

    Args:
        df: Input DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
        GeoDataFrame with geometry points created from coordinate columns.
    """
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

# ==================================================================================== #

class MinDatScraper:
    """Scraper for retrieving locality data from Mindat website.
    
    Attributes:
        MinDatLocalitiesURL: URL for Mindat's country list page.
        session:             Persistent HTTP session.
        localities:          List of locality names extracted from Mindat.
    """
    
    MinDatLocalitiesURL = 'https://www.mindat.org/countrylist.php'
    
    def __init__(self):
        """Initialize scraper and fetch localities."""
        self._get_session()
        self._get_localities()
      
    def _get_session(self) -> None:
        """Configure HTTP session with custom User-Agent."""
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
      
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
            for row in table.find_all('tr')[1:]:  # Skip the header
                cols = row.find_all('td')
                if cols:
                    a_tag = cols[0].find('a')
                    if a_tag:
                        locality_name = a_tag.get_text(strip=True)
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
    
    def __init__(
        self, 
        api_key: str, 
        output_path: Optional[Union[str, Path]] = Path(REFERENCES_DIR) / 'mindat.tsv',
        plot_package: str = 'mpl',
        verbose: bool = False
    ):
        """Initialize Mindat API client.
        
        Args:
            api_key:      Mindat API key for authentication.
            output_path:  Directory for storing output files. Defaults to REFERENCES_DIR.
            plot_package: Visualization package to use. Currently supports 'mpl'.
        """
        os.environ["MINDAT_API_KEY"] = api_key
        self.output_path = output_path
        self.plot_package = plot_package
        self.verbose = verbose
      
        try:
            self.localities = self._get_mindat_localities()
        except Exception as e:
            self.log(f"Error getting Mindat localities: {e}")
        
        # Ensure we have localities
        if not self.localities:
            self.log("No localities found. Using pycountry fallback.")
            self.localities = self._get_pycountry_countries()
            
    def log(self, msg):
        """Log message if verbose mode is enabled."""
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)
        
    def _get_mindat_localities(self):
        """Retrieve localities from Mindat website."""
        scraper = MinDatScraper()
        return scraper.localities

    def _get_pycountry_countries(self):
        """Fallback method to get country names from pycountry."""
        return [country.name for country in pycountry.countries]
    
    def _get_uranium_mines_locality(self, locality: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Retrieve uranium mines data for a specific locality."""
        lr = LocalitiesRetriever()
        lr.country(locality).description("mine").elements_inc("U")
        
        # Suppress stdout/stderr from openmindat
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results = lr.get_dict()
        
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
        # Keep rows where 'keep' and 'is_mine' are True AND 
        # either 'is_uranium_txt' or 'is_uranium_desc' is True
        filter_condition = (
            df['keep'] & 
            df['is_mine'] & 
            (df['is_uranium_txt'] | df['is_uranium_desc']) | df['is_u_txt'] | df['is_u_desc']
        )
        df = df[filter_condition]

        # Drop rows where BOTH latitude AND longitude are 0
        df = df[~((df['latitude'] == 0) & (df['longitude'] == 0))].reset_index(drop=True)
        
        return df

        
    def _get_uranium_mines_world(self):
        """Retrieve and process uranium mines data for all available localities.
        
        Returns:
            Tuple containing:
            - Combined DataFrame of all mine data
            - Combined GeoDataFrame of all geometric data
        """
        dfs = []
        for locality in self.localities: 
            self.log(f"Processing {locality}")
            try:
                df, gdf = self._get_uranium_mines_locality(locality)
                if not df.empty:
                    dfs.append(df)
                    #self._mpl_plot_uranium_mines_locality(locality, gdf)
                else:
                    self.log(f"No uranium mines found in {locality}")
            except Exception as e:
                self.log(f"Error with {locality}: {e}")
        
        if dfs:
            df = pd.concat(dfs, axis=0)
            df['facility'] = [i.split(',')[0] for i in df['txt']]
            df = self._filter_data(df)
            df.to_csv(self.output_path, sep="\t", index=False)
            gdf = gpd_from_df(df)
            if not gdf.empty:
                if self.plot_package == 'mpl':
                    self._mpl_plot_uranium_mines_locality('world', gdf)
            return df, gdf
        else:
            self.log("No data found for any locality")
            return pd.DataFrame(), gpd.GeoDataFrame()

# ==================================================================================== #

def world_uranium_mines(
    config: Dict,
    api_key: str, 
    output_dir: Union[str, Path] = REFERENCES_DIR
):
    """Main function to retrieve worldwide uranium mines data.
    
    Args:
        api_key:    Mindat API key.
        output_dir: Output directory for results. Defaults to REFERENCES_DIR.
        
    Returns:
        Tuple containing combined DataFrame and GeoDataFrame of uranium mines.
    """
    output_path = Path(output_dir) / "mindat.tsv"
    if config.get("nfc_facilities", {}).get("use_local", False):
        if output_path.exists():
            return pd.read_csv(output_path, sep='\t'), gpd.GeoDataFrame()
    mindat_api = MinDatAPI(api_key, output_path)
    return mindat_api._get_uranium_mines_world()
