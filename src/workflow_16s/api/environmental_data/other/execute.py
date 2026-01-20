# workflow_16s/api/environmental_data/other/execute.py

# ==================================================================================== #

# Standard Imports
import json
import logging
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.table import Table

# Local Imports
from workflow_16s.api.environmental_data.other import (
    BaseEnvironmentalAPI, _inaturalist, _meteostat, _noaa, _nrel, _nws, 
    _openmeteo, _soilgrids, _radnet, _usgs
)
from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.dir_utils import Project
from workflow_16s.utils.logger import setup_logging
from workflow_16s.utils.progress import get_progress_bar

# ==================================================================================== #

load_dotenv()
logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

MAX_WORKERS = 10

# A list of API client names that support the 'fetch_date' parameter
APIS_WITH_DATE_SUPPORT = [
    "Meteostat", "NOAA_Tides", "USGS_Earthquake", "NREL_Solar",
    "EnvironmentalHealth", "SoilState", "OpenMeteo"
]

# ==================================================================================== #

class EnvironmentalDataCollector:
    """
    Collects environmental data from multiple APIs for given coordinates and dates,
    then returns the aggregated data as a pandas DataFrame.
    """
    def __init__(
        self, data: pd.DataFrame, config: AppConfig,
        output_file: Optional[Path] = Path("./environmental_data.json"),
        max_workers: int = MAX_WORKERS, verbose: bool = False
    ):

        if not all(col in data.columns for col in ['lat', 'lon', 'collection_date']):
            raise ValueError("Input DataFrame must contain 'lat', 'lon', and 'collection_date' columns.")

        self.data = data.dropna(subset=['lat', 'lon', 'collection_date']).copy()
        self.data['collection_date'] = pd.to_datetime(
            self.data['collection_date'], format='mixed', errors='coerce'
        ).dt.strftime('%Y-%m-%d')
        self.data = self.data.drop_duplicates(subset=['lat', 'lon', 'collection_date'])
        self.coordinates = list(self.data[['lat', 'lon', 'collection_date']].itertuples(index=False, name=None))
        self.config = config
        self.output_file = output_file
        self.max_workers = max_workers
        self.verbose = verbose
        self.results: List[Dict] = []
        self.api_statuses: List[Dict] = []
        self.skipped_handlers: List[Dict] = []
        
        # Statistics tracking
        self.stats = {
            'total_api_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'cached_calls': 0,
            'total_locations': len(self.coordinates),
            'api_performance': {}  # Track performance per API
        }
        
        # Initialize and validate APIs
        self.active_handlers: Dict[str, BaseEnvironmentalAPI] = self._initialize_apis()


    def _initialize_apis(self) -> Dict[str, BaseEnvironmentalAPI]:
        """Initializes all API handlers and validates their requirements."""
        if not self.config:
            logger.error("Configuration object is missing. Cannot initialize APIs.")
            return {}
            
        email = str(self.config.credentials.email or "contact@example.com")
        
        # Define all potential API handlers
        all_apis = {
            "EnvironmentalHealth": _openmeteo.EnvironmentalHealthAPI(verbose=self.verbose),
            "iNaturalist": _inaturalist.iNaturalistAPI(verbose=self.verbose),
            "Meteostat": _meteostat.MeteostatAPI(verbose=self.verbose),
            "NOAA_Tides": _noaa.NOAA_Tides_API(verbose=self.verbose),
            "NREL_Solar": _nrel.NREL_Solar_API(verbose=self.verbose),
            "NWS": _nws.NWS_API(email=email, verbose=self.verbose),
            "OpenMeteo": _openmeteo.OpenMeteoAPI(verbose=self.verbose),
            "RadNet": _radnet.RadNetAPI(verbose=self.verbose),
            "SevereWeather": _nws.SevereWeatherAPI(email=email, verbose=self.verbose),
            "SoilGrids": _soilgrids.SoilGridsAPI(verbose=self.verbose),
            "SoilState": _openmeteo.SoilStateAPI(verbose=self.verbose),
            "USGS_Earthquake": _usgs.USGS_Earthquake_API(verbose=self.verbose)
        }

        active_handlers = {}
        # Validate each handler and add it to the active list or skipped list
        for name, handler in all_apis.items():
            is_ok, reason = handler.check_requirements()
            if is_ok:
                active_handlers[name] = handler
            else:
                self.skipped_handlers.append({"api": name, "reason": reason})

        if self.skipped_handlers:
            reasons_str = "\n".join([f"  - {s['api']}: {s['reason']}" for s in self.skipped_handlers])
            logger.warning(f"Disabling {len(self.skipped_handlers)} API(s) due to missing requirements:\n{reasons_str}")
        
        return active_handlers

    def fetch_api_data(
        self, api_name: str, api_instance: BaseEnvironmentalAPI, 
        lat: float, lon: float, date: str
    ) -> Dict:
        """Fetches data from a single API, handling errors. Caching is now handled by a decorator."""
        result = {"api": api_name, "lat": lat, "lon": lon, "date": date}
        
        try:
            kwargs = {'lat': lat, 'lon': lon}
            if api_name in APIS_WITH_DATE_SUPPORT:
                kwargs['fetch_date'] = str(date) # type: ignore
            
            data = api_instance.get_data(**kwargs)
            
            status = "SUCCESS" if data is not None else "FAILED"
            details = None if data is not None else "API returned no data."

            result.update({"status": status, "data": data, "details": details})

        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.debug(f"API Error in {api_name} for ({lat}, {lon}): {error_message}", exc_info=self.verbose)
            result.update({"status": "FAILED", "details": error_message, "data": None})
        
        return result
    
    def _flatten_api_data(self, data: Dict, prefix: str = '', sep: str = '_') -> Dict:
        """
        Recursively flattens a nested dictionary. 
        e.g. {'a': {'b': 1}} -> {'a_b': 1}
        """
        items = {}
        for k, v in data.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            
            if isinstance(v, dict):
                # Recurse into nested dictionaries
                items.update(self._flatten_api_data(v, new_key, sep=sep))
            elif isinstance(v, list):
                # Handle lists: if singular, unpack it. If empty, None. 
                # If multiple items, keep as list (or json stringify if strictness needed)
                if not v:
                    items[new_key] = None
                elif len(v) == 1:
                    # Recursively check if the single item is a dict
                    if isinstance(v[0], dict):
                        items.update(self._flatten_api_data(v[0], new_key, sep=sep))
                    else:
                        items[new_key] = v[0]
                else:
                    # If it's a list of strings/numbers, join them for cleaner CSVs
                    if all(isinstance(x, (str, int, float)) for x in v):
                        items[new_key] = ";".join(map(str, v))
                    else:
                        items[new_key] = v 
            else:
                items[new_key] = v
        return items

    def run_apis(self) -> pd.DataFrame:
        """
        Runs API calls concurrently, generates summaries, and returns the collected data as a DataFrame.
        """
        if not self.active_handlers and not self.skipped_handlers:
            logger.warning("No environmental APIs are defined. Skipping.")
            return pd.DataFrame()
        if not self.active_handlers:
            logger.warning("All environmental APIs are disabled due to missing requirements.")
            self._summarize_api_calls()
            return pd.DataFrame()

        all_results_map = {
            loc: {"location": {"lat": loc[0], "lon": loc[1], "collection_date": loc[2]}}
            for loc in self.coordinates
        }
        
        total_tasks = len(self.coordinates) * len(self.active_handlers)
        
        workflow_logger = logging.getLogger("workflow_16s")
        original_level = workflow_logger.level
        workflow_logger.setLevel(logging.CRITICAL)  # Suppress logs during fetch

        try:
            with get_progress_bar() as progress:
                task = progress.add_task("[cyan]Fetching environmental data", total=total_tasks)
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self.fetch_api_data, name, api, lat, lon, date)
                        for name, api in self.active_handlers.items()
                        for lat, lon, date in self.coordinates
                    ]
    
                    for future in as_completed(futures):
                        res = future.result()
                        self.api_statuses.append(res)
                        loc_tuple = (res['lat'], res['lon'], res['date'])
                        
                        if res.get('status') == "SUCCESS":
                            all_results_map[loc_tuple][res['api']] = res.get('data') or {}
                        else:
                            all_results_map[loc_tuple][res['api']] = {"error": res.get('details')}
                        
                        progress.update(task, advance=1)
        finally:
            workflow_logger.setLevel(original_level) # Always restore the original level

        self.results = list(all_results_map.values())
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Environmental data saved to {self.output_file}")
            
        self._summarize_api_calls()
        self._summarize_location_data()
        
        # Log final statistics
        logger.info(f"Environmental data collection complete:")
        logger.info(f"  - Total API calls: {self.stats['total_api_calls']}")
        logger.info(f"  - Success rate: {self.stats['successful_calls']/max(1,self.stats['total_api_calls'])*100:.1f}%")
        logger.info(f"  - Active APIs: {len(self.active_handlers)}")
        logger.info(f"  - Locations processed: {self.stats['total_locations']}")
        
        # Create, log, and return the final DataFrame
        results_df = self._create_results_dataframe()
        if not results_df.empty:
            self._log_dataframe_as_table(results_df, "Retrieved Environmental Data Columns")
        
        return results_df

    def _create_results_dataframe(self) -> pd.DataFrame:
        """
        Transforms the nested results list into a flat pandas DataFrame.
        """
        if not self.results:
            return pd.DataFrame()

        records = []
        for result_group in self.results:
            # Extract location info and initialize the record
            location_info = result_group.get('location', {})
            record = {
                'lat': location_info.get('lat'),
                'lon': location_info.get('lon'),
                'collection_date': location_info.get('collection_date')
            }
            if all(v is None for v in record.values()):
                continue

            # Unnest data from each successful API call
            for api, data in result_group.items():
                if api == 'location' or not isinstance(data, dict) or 'error' in data:
                    continue
                
                # --- CHANGE: Use recursive flattening instead of shallow prefixing ---
                # This ensures nested keys like 'processing_metadata' -> 'soilgrids' -> 'bdod' 
                # become 'SoilGrids_processing_metadata_soilgrids_bdod' columns
                flattened_data = self._flatten_api_data(data, prefix=api)
                record.update(flattened_data)
                # -------------------------------------------------------------------

            records.append(record)

        if not records: return pd.DataFrame()

        df = pd.DataFrame(records)
        
        # Ensure location columns are first
        id_cols = ['lat', 'lon', 'collection_date']
        # Filter out cols that might not exist in this batch of records
        existing_id_cols = [col for col in id_cols if col in df.columns]
        data_cols = [col for col in df.columns if col not in existing_id_cols]
        
        return df[existing_id_cols + sorted(data_cols)]

    def _log_dataframe_as_table(self, df: pd.DataFrame, title: str):
        """Logs a pandas DataFrame as a plain-text table for log file readability."""
        header = f"--- {title} ---"
        logger.info(f"\n{header}\n{df.to_string()}\n" + "-" * len(header))

    def _summarize_api_calls(self):
        """Creates and prints a summary table of all API call statuses."""
        fetch_summary = {}
        for status in self.api_statuses:
            api = status['api']
            if api not in fetch_summary:
                fetch_summary[api] = {"SUCCESS": 0, "FAILED": 0, "errors": set()}
            s = status['status']
            fetch_summary[api][s] += 1
            if s == "FAILED":
                fetch_summary[api]['errors'].add(status.get('details', 'Unknown error'))
        
        table_data = []
        all_handler_names = list(self.active_handlers.keys()) + [s['api'] for s in self.skipped_handlers]
        for api_name in sorted(all_handler_names):
            if api_name in self.active_handlers:
                handler = self.active_handlers[api_name]
                counts = fetch_summary.get(api_name, {"SUCCESS": 0, "FAILED": 0, "errors": set()})
                table_data.append([
                    api_name, "OPERATIONAL", handler.cache_hits, handler.cache_misses,
                    counts["SUCCESS"], counts["FAILED"], "; ".join(counts['errors'])
                ])
            else: # Skipped handlers
                reason = next((s['reason'] for s in self.skipped_handlers if s['api'] == api_name), "N/A")
                table_data.append([api_name, "SKIPPED", 0, 0, 0, 0, reason])

        df = pd.DataFrame(table_data, columns=["API", "Status", "Cache Hits", "Fetches", "Successful", "Failed", "Details"])
        self._log_dataframe_as_table(df, "Environmental Data API Status Summary")

        console = Console()
        table = Table(title="Environmental Data API Status Summary", box=box.ROUNDED, show_header=True)
        for col in ["API", "Status", "Cache Hits", "Fetches\n(Cache Misses)", "Successful\nFetches", "Failed\nFetches", "Details"]:
            table.add_column(col)
        
        for row in table_data:
            status, success, failed = row[1], row[4], row[5]
            if status == "OPERATIONAL":
                status_style = "green"
                if failed > 0: status_style = "yellow" if success > 0 else "red"
                status_text = f"[{status_style}]OPERATIONAL[/]"
            else:
                status_text = "[yellow]SKIPPED[/]"
            table.add_row(row[0], status_text, str(row[2]), str(row[3]), str(row[4]), str(row[5]), row[6])
        console.print(table)


    def _summarize_location_data(self):
        """Creates and prints a summary table of data fetched for each location."""
        if not self.results: return

        KEY_DATA_POINTS = {
            'tavg': 'Avg Temp', 
            'temperature_2m_max': 'Max Temp', 
            'precipitation_sum': 'Precip', 
            'mag': 'EQ Mag'
        }
        
        table_data = []
        for result in self.results:
            location = result.get('location', {})
            lat, lon, date = location.get('lat'), location.get('lon'), location.get('collection_date', 'N/A')
            successful_apis, summary_points = [], []
            for api, data in result.items():
                if api == 'location' or not isinstance(data, dict) or 'error' in data: continue
                successful_apis.append(api)
                for key, name in KEY_DATA_POINTS.items():
                    if key in data and isinstance(data[key], (int, float)):
                        val = data[key]
                        summary_points.append(f"{name}: {val:.1f}" if isinstance(val, float) else f"{name}: {val}")
            
            table_data.append([
                f"{lat:.2f}, {lon:.2f}" if isinstance(lat, float) and isinstance(lon, float) else "N/A",
                str(date), ", ".join(sorted(successful_apis)), len(successful_apis),
                "; ".join(summary_points) if summary_points else "N/A"
            ])
        
        df = pd.DataFrame(table_data, columns=["Location", "Date", "Successful APIs", "# APIs", "Example Data"])
        self._log_dataframe_as_table(df, "Fetched Data Summary by Location")

        console = Console()
        table = Table(title="Fetched Data Summary by Location", box=box.ROUNDED, show_header=True)
        table.add_column("Location (Lat, Lon)", style="cyan")
        table.add_column("Date", style="magenta")
        table.add_column("Successful APIs (#)", style="green", overflow="fold")
        table.add_column("Example Data Points", overflow="fold")
        for row in table_data:
            table.add_row(row[0], row[1], f"{row[2]} ({row[3]})", row[4])
        console.print(table)


# ==================================================================================== #

if __name__ == "__main__":
    from types import SimpleNamespace

    # --- Basic Setup for Standalone Execution ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n--- Running EnvironmentalDataCollector in Test Mode ---")
    sample_df = pd.DataFrame({
        'lat': [37.7749], 'lon': [-122.4194], 'collection_date': ['2025-10-14'] 
    })
    print(f"Fetching data for {len(sample_df)} location(s)...")

    mock_config = SimpleNamespace(credentials=SimpleNamespace(email="test.user@example.com"))
    
    collector = EnvironmentalDataCollector(
        data=sample_df, config=mock_config, # type: ignore
        output_file=Path("./test_environmental_data.json"), verbose=True
    )
    # The run_apis method now returns a DataFrame
    final_data_df = collector.run_apis()
    
    print("\n--- Test Run Complete ---")
    if not final_data_df.empty:
        print("--- Returned DataFrame ---")
        print(final_data_df)