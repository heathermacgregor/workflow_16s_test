# workflow_16s/utils/taxonomy.py

from __future__ import annotations
import io
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd # Added for type hint fix

from workflow_16s.utils.logger import get_logger


class FaprotaxError(Exception):
    """Base exception for FaprotaxDB errors."""

class DownloaderError(FaprotaxError):
    """Raised when the database download fails."""

class ParserError(FaprotaxError):
    """Raised during database file parsing."""

class FaprotaxDB:
    """
    A class to manage and query the FAPROTAX functional annotation database.

    Handles automatic download, caching, and parsing of the latest FAPROTAX database.
    Provides a method to predict functions based on taxonomic strings.
    """
    BASE_URL = "https://pages.uoregon.edu/slouca/LoucaLab/archive/FAPROTAX/lib/php/index.php?section=Download"
    def __init__(
        self, 
        db_path: Union[str, Path, None] = None, 
        project_name: str = "workflow_16s"
    ):
        """Initializes the FaprotaxDB instance.

        Args:
            db_path: Optional path to directory for database cache. If None, searches '<project_name>/references/faprotax' or uses cache.
            project_name: Project directory name used if `db_path` is None.
        """
        if db_path: self.db_dir = Path(db_path)
        else: self.db_dir = self._find_references_dir(project_name) / "faprotax"
        self.db_file = self.db_dir / "FAPROTAX.txt"
        self._db_data = None # Lazy-loaded database content
        self.logger = get_logger("workflow_16s")
        self.logger.info(f"FAPROTAX DB directory set to: {self.db_dir}")

    @property
    def data(self) -> Dict:
        """Lazily loads the FAPROTAX database into memory on first access."""
        if self._db_data is None: self._db_data = self._load_database()
        return self._db_data

    @staticmethod
    def _find_references_dir(project_name: str) -> Path:
        """Locates '<project_name>/references' directory by walking up from CWD."""
        start = Path.cwd().resolve()
        for parent in [start, *start.parents]:
            # Check if 'src/<project_name>' exists first for common project layouts
            src_project_dir = parent / "src" / project_name
            if src_project_dir.is_dir():
                ref_dir = src_project_dir.parent.parent / "references" # Go up two levels from src
                if ref_dir.is_dir():
                    get_logger("workflow_16s").debug(f"Found references dir via src structure: {ref_dir}")
                    return ref_dir

            # Check if '<project_name>' directory exists directly
            project_dir = parent / project_name
            if project_dir.is_dir():
                ref_dir = project_dir.parent / "references" # Go up one level
                if ref_dir.is_dir():
                    get_logger("workflow_16s").debug(f"Found references dir via direct project structure: {ref_dir}")
                    return ref_dir

        # Fallback if specific project structure not found
        fallback_dir = Path.home() / ".cache" / project_name / "references"
        get_logger("workflow_16s").warning(f"Could not find '{project_name}/references' structure. Using fallback cache: {fallback_dir}")
        return fallback_dir

    def _download_latest(self) -> Path:
        """Scrapes, downloads, and extracts the latest FAPROTAX.txt file."""
        self.logger.info("Searching for the latest FAPROTAX database download link...")
        try:
            response = requests.get(self.BASE_URL, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Regex to find links like FAPROTAX_1.2.1.zip
            rgx = re.compile(r"FAPROTAX_(\d+(?:\.\d+)*)\.zip$", re.IGNORECASE)
            links: List[Tuple[str, str]] = []
            for a in soup.find_all("a", href=True):
                href = a.get('href')
                if isinstance(href, str):
                    match = rgx.search(href)
                    if match:
                        full_url = urljoin(self.BASE_URL, href)
                        version = match.group(1)
                        links.append((full_url, version))

            if not links: raise DownloaderError("No FAPROTAX *.zip links found on download page.")

            # Sort by version number (handle versions like 1.2.1 vs 1.2)
            links.sort(key=lambda x: tuple(map(int, x[1].split('.'))), reverse=True)
            zip_url, version = links[0]

            self.logger.info(f"Downloading FAPROTAX v{version} from {zip_url}")
            # Use streaming for potentially large files
            with requests.get(zip_url, timeout=120, stream=True) as r:
                r.raise_for_status()
                zip_content = r.content # Read content into memory (adjust if files become huge)

            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # Find the correct file within the zip (case-insensitive)
                txt_members = [m for m in zf.namelist() if m.lower().endswith("faprotax.txt")]
                if not txt_members: raise DownloaderError("FAPROTAX.txt not found in the downloaded ZIP archive.")

                # Prioritize exact match if multiple found (unlikely)
                target_member = "FAPROTAX.txt" if "FAPROTAX.txt" in txt_members else txt_members[0]

                self.db_dir.mkdir(parents=True, exist_ok=True)
                # Extract directly to the final filename to avoid race conditions/renaming issues
                with zf.open(target_member) as source, open(self.db_file, "wb") as target:
                    target.write(source.read())

                self.logger.info(f"Successfully saved FAPROTAX.txt to {self.db_file}")

        except requests.RequestException as e:
            raise DownloaderError(f"Network error while downloading: {e}") from e
        except Exception as e:
            # Catch other potential errors (zipfile, parsing, etc.)
            raise DownloaderError(f"Error during FAPROTAX download/extraction: {e}") from e

        return self.db_file

    @lru_cache(maxsize=1) # Cache the parsed result
    def _load_database(self) -> Dict:
        """Loads and parses FAPROTAX.txt. Downloads if needed. Caches result."""
        if not self.db_file.exists():
            try:
                self._download_latest()
            except DownloaderError as e:
                self.logger.error(f"Failed to download FAPROTAX DB: {e}")
                return {} # Return empty dict if download fails

        self.logger.info(f"Parsing FAPROTAX database from {self.db_file}...")
        trait_dict = {}
        current_trait = None

        try:
            with self.db_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue

                    if not line.startswith(" ") and ":" in line:
                        parts = line.split(maxsplit=1)
                        trait = parts[0]
                        if re.search(r"^\*.*\*$", trait) or ";" in trait:
                            current_trait = None; continue # Skip malformed group names
                        current_trait = trait
                        trait_dict[current_trait] = {"taxa": []}
                    elif current_trait:
                        fields = line.split("//", 1)
                        pattern_raw = fields[0].strip()
                        if not pattern_raw or pattern_raw == "*": continue

                        # Convert FAPROTAX glob-style pattern (* means any sequence) to regex
                        regex_str = re.escape(pattern_raw).replace(r"\*", ".*")
                        try:
                            trait_dict[current_trait]["taxa"].append({
                                "pattern": re.compile(f"^{regex_str}$", re.IGNORECASE), # Anchor and case-insensitive
                                "reference": fields[1].strip() if len(fields) > 1 else ""
                            })
                        except re.error as e:
                            self.logger.warning(f"Skipping invalid regex pattern for trait '{current_trait}': '{regex_str}' ({e})")
        except FileNotFoundError:
            self.logger.error(f"FAPROTAX file not found at {self.db_file} even after download attempt.")
            return {}
        except Exception as e:
            raise ParserError(f"Error parsing {self.db_file}: {e}") from e

        # Filter out traits that ended up with no valid patterns
        parsed_data = {t: d for t, d in trait_dict.items() if d.get("taxa")}
        self.logger.info(f"Parsed {len(parsed_data)} functional groups from FAPROTAX.")
        return parsed_data

    def predict_functions(
        self, 
        taxonomy: Union[str, float], 
        include_references: bool = False
    ) -> Union[List[str], Dict[str, List[str]]]: # Allow float for potential NaN
        """Finds all FAPROTAX functions for a given taxonomy string."""
        if not taxonomy or pd.isna(taxonomy): # Handle empty/NaN input
            return {} if include_references else []

        # Normalize the taxonomy string: remove prefixes, collapse separators, lowercase
        norm_taxon = re.sub(r"\s*[a-z]__\s*", ";", str(taxonomy).strip().lower())
        norm_taxon = re.sub(r";+", ";", norm_taxon).strip(";")

        if include_references:
            results: Dict[str, List[str]] = {}
            for trait, entry in self.data.items():
                for record in entry["taxa"]:
                    # Match against any part of the semicolon-separated lineage
                    if any(record["pattern"].match(part) for part in norm_taxon.split(';')):
                        results.setdefault(trait, []).append(record["reference"])
            return results
        else:
            results_set = set() # Use a set to avoid duplicates initially
            for trait, entry in self.data.items():
                for record in entry["taxa"]:
                    if any(record["pattern"].match(part) for part in norm_taxon.split(';')):
                        results_set.add(trait)
                        break # Found a match for this trait, move to the next trait
            return sorted(list(results_set))