#!/usr/bin/env python3

"""
FAPROTAX Database Manager (faprotax.py)

This script provides a Python class to download, parse, and query the
FAPROTAX (Functional Annotation of Prokaryotic Taxa) database.

It is optimized for performance by:
1. Lazily downloading and parsing the database file on first access.
2. Compiling all 90+ functional groups into a single "master regex".
3. Using this master regex to perform all function predictions with a single
   regex search per taxon, which is orders of magnitude faster than the naive approach.
4. Providing a parallelized, 'batch' prediction method for processing
   large lists of taxa (e.g., from an AnnData object) using all CPU cores.
"""

from __future__ import annotations
import io
import logging
import re
import zipfile
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Union, Tuple, Set, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar


# Custom Exceptions
class FaprotaxError(Exception):
    """Base exception for FaprotaxDB errors."""

class DownloaderError(FaprotaxError):
    """Raised when the database download fails."""

class ParserError(FaprotaxError):
    """Raised during database file parsing."""

# Main FaprotaxDB Class
class FaprotaxDB:
    """
    A class to manage and query the FAPROTAX functional annotation database.

    Handles automatic download, caching, and optimized parsing of the FAPROTAX database.
    Resolves the internal group hierarchy and compiles a single "master regex"
    for extremely fast functional prediction on many taxa.
    """
    BASE_URL = "https://pages.uoregon.edu/slouca/LoucaLab/archive/FAPROTAX/lib/php/index.php?section=Download"

    def __init__(self, db_path: Union[str, Path, None] = None):
        """
        Initializes the FaprotaxDB instance.

        Args:
            db_path: 
                Optional path to directory for database cache. 
                If None, defaults to '~/.cache/faprotax_db'.
        """
        if db_path: self.db_dir = Path(db_path)
        else: self.db_dir = Path.home() / ".cache" / "faprotax_db"

        self.db_file = self.db_dir / "FAPROTAX.txt"

        # Attributes for optimized prediction, loaded lazily
        self._db_taxa_patterns: Union[Dict[str, List[str]], None] = None
        self._group_metadata: Union[Dict, None] = None
        self._master_regex_compiled: Union[re.Pattern, None] = None
        self._clean_name_map: Union[Dict[str, str], None] = None # {clean_name: original_name}
        self.logger = get_logger("workflow_16s")
        self.logger.info(f"FAPROTAX DB directory set to: {self.db_dir}")

    @property
    def data(self) -> Dict[str, List[str]]:
        """
        Lazily loads the FAPROTAX database on first access.
        Returns a dictionary of {group_name: [list_of_regex_pattern_strings]}.
        """
        if self._db_taxa_patterns is None: self._db_taxa_patterns, self._group_metadata = self._load_database()
        return self._db_taxa_patterns

    @property
    def metadata(self) -> Dict:
        """Lazily loads and returns the group metadata on first access."""
        if self._group_metadata is None: self._db_taxa_patterns, self._group_metadata = self._load_database()
        return self._group_metadata

    @property
    def _clean_map(self) -> Dict[str, str]:
        """
        Builds and caches a {clean_name: original_name} map.
        Regex-named groups must be valid Python identifiers (no hyphens, etc.).
        """
        if self._clean_name_map is None:
            self._clean_name_map = {}
            for group_name in self.data.keys():
                # Clean name: replace non-word chars with _, ensure no leading digit
                clean_name = re.sub(r'\W|^(?=\d)', '_', group_name)

                # Handle potential name collisions
                if clean_name in self._clean_name_map:
                    i = 0
                    while f"{clean_name}_{i}" in self._clean_name_map: i += 1
                    clean_name = f"{clean_name}_{i}"

                self._clean_name_map[clean_name] = group_name
        return self._clean_name_map

    @property
    def _master_regex(self) -> re.Pattern:
        """
        Builds, caches, and returns the single compiled "master regex".
        This regex uses named groups to find all functions in one pass.
        Format: (?P<group1>...)|(?P<group2>...)|...
        """
        if self._master_regex_compiled is None:
            master_pattern_list = []
            # Need the reverse map: {original_name: clean_name}
            reverse_map = {v: k for k, v in self._clean_map.items()}

            for group_name, patterns in self.data.items():
                # Skip groups that resolved to no taxa
                if not patterns: continue
                clean_name = reverse_map[group_name]
                # Combine all patterns for this group: (pattern1|pattern2|...)
                group_sub_pattern = f"(?:{'|'.join(patterns)})"
                # Create a named group: (?P<clean_group_name>(...))
                master_pattern_list.append(f"(?P<{clean_name}>{group_sub_pattern})")

            # Combine all named groups
            master_regex_str = "|".join(master_pattern_list)
            # Compile without ^ or $ anchors to allow matching anywhere
            self._master_regex_compiled = re.compile(master_regex_str, re.IGNORECASE)
            self.logger.info("Master regex compiled.")
        return self._master_regex_compiled

    def _download_latest(self) -> Path:
        """Scrapes, downloads, and extracts the latest FAPROTAX.txt file."""
        self.logger.info("Searching for the latest FAPROTAX database download link...")
        try:
            response = requests.get(self.BASE_URL, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Regex to find links like FAPROTAX_1.0.0.zip
            rgx = re.compile(r"FAPROTAX_(\d+(?:\.\d+)*)\.zip$", re.IGNORECASE)
            links: List[Tuple[str, str]] = []
            for a in soup.find_all("a", href=True):
                if not hasattr(a, 'get'): continue
                href = a.get('href') # type: ignore
                if isinstance(href, str):
                    match = rgx.search(href)
                    if match:
                        full_url = urljoin(self.BASE_URL, href)
                        version = match.group(1)
                        links.append((full_url, version))

            if not links: raise DownloaderError("No FAPROTAX *.zip links found on download page.")

            # Sort by version number to get the latest
            links.sort(key=lambda x: tuple(map(int, x[1].split('.'))), reverse=True)
            zip_url, version = links[0]

            self.logger.info(f"Downloading FAPROTAX v{version} from {zip_url}")
            with requests.get(zip_url, timeout=120, stream=False) as r:
                r.raise_for_status()
                zip_content = r.content

            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # Find the correct file within the zip
                txt_members = [m for m in zf.namelist() if m.lower().endswith("faprotax.txt")]
                if not txt_members: raise DownloaderError("FAPROTAX.txt not found in the downloaded ZIP archive.")

                target_member = "FAPROTAX.txt" if "FAPROTAX.txt" in txt_members else txt_members[0]

                self.db_dir.mkdir(parents=True, exist_ok=True)
                with zf.open(target_member) as source, open(self.db_file, "wb") as target: target.write(source.read())

                self.logger.info(f"Successfully saved FAPROTAX.txt to {self.db_file}")

        except requests.RequestException as e: raise DownloaderError(f"Network error while downloading: {e}") from e
        except Exception as e: raise DownloaderError(f"Error during FAPROTAX download/extraction: {e}") from e

        return self.db_file

    def _parse_metadata(self, metadata_str: str) -> Dict[str, List[str]]:
        """Parse metadata string in format 'key1:value1,value2; key2:value3'"""
        metadata = {}
        if not metadata_str.strip(): return metadata

        for item in metadata_str.split(';'):
            item = item.strip()
            if not item or ':' not in item: continue
            key, values = item.split(':', 1)
            key = key.strip()
            # Split values by comma and strip whitespace
            value_list = [v.strip() for v in values.split(',')]
            metadata[key] = value_list
        return metadata

    @lru_cache(maxsize=1)
    def _load_database(self) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Loads and parses FAPROTAX.txt using blank lines as delimiters.
        
        Returns:
            Tuple:
            1. A dict of {group_name: [list_of_uncompiled_regex_patterns]}
            2. A dict of {group_name: {metadata_key: [metadata_values]}}
        """
        if not self.db_file.exists():
            try: self._download_latest()
            except DownloaderError as e: self.logger.error(f"Failed to download FAPROTAX DB: {e}"); return {}, {}

        self.logger.info(f"Parsing FAPROTAX database from {self.db_file}...")

        try:
            # === Pass 1: Parse raw groups and operations (using blank lines) ===
            raw_groups: Dict[str, Dict[str, Any]] = {}
            group_metadata: Dict[str, Dict[str, List[str]]] = {}
            current_group = None
            current_operations = []

            with self.db_file.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line_stripped = line.strip()

                    # A blank line signifies the end of a group block
                    if not line_stripped:
                        if current_group:
                            # Save the completed group
                            raw_groups[current_group] = {"operations": current_operations.copy()}
                            self.logger.debug(f"Finished group: {current_group} with {len(current_operations)} ops.")
                        # Reset for the next block
                        current_group = None
                        current_operations = []
                        continue

                    # Skip comment lines
                    if line_stripped.startswith("#"): continue

                    # If we are not in a group, this line must be a new group header
                    if current_group is None:
                        parts = line_stripped.split(None, 1)
                        group_name = parts[0]
                        metadata = {}
                        if len(parts) > 1: metadata = self._parse_metadata(parts[1])
                        group_metadata[group_name] = metadata
                        current_group = group_name
                        self.logger.debug(f"Found group: {group_name}")

                    # Otherwise, this line is an operation for the current group
                    else:
                        if line_stripped.startswith("add_group:"):
                            op_part = line_stripped[len("add_group:"):].split("#", 1)[0].split("//", 1)[0].strip()
                            if op_part: current_operations.append(("add", op_part))
                        elif line_stripped.startswith("subtract_group:"):
                            op_part = line_stripped[len("subtract_group:"):].split("#", 1)[0].split("//", 1)[0].strip()
                            if op_part: current_operations.append(("subtract", op_part))
                        elif line_stripped.startswith("intersect_group:"):
                            op_part = line_stripped[len("intersect_group:"):].split("#", 1)[0].split("//", 1)[0].strip()
                            if op_part: current_operations.append(("intersect", op_part))
                        else:
                            # It must be a taxon pattern
                            taxon_parts = line_stripped.split("//", 1)
                            taxon_pattern = taxon_parts[0].split("#", 1)[0].strip()
                            reference = taxon_parts[1].strip() if len(taxon_parts) > 1 else ""
                            if taxon_pattern.startswith('"') and taxon_pattern.endswith('"'): taxon_pattern = taxon_pattern[1:-1]
                            if taxon_pattern: current_operations.append(("taxon", (taxon_pattern, reference)))

            # Save the very last group in the file
            if current_group:
                raw_groups[current_group] = {"operations": current_operations.copy()}
                self.logger.debug(f"Finished last group: {current_group} with {len(current_operations)} ops.")

            self.logger.info(f"Parsed {len(raw_groups)} group definitions with metadata.")

            # === Pass 2: Resolve group hierarchies ===
            direct_taxa: Dict[str, Set[Tuple[str, str]]] = {}
            for group_name, group_data in raw_groups.items():
                direct_taxa[group_name] = set()
                for op_type, operand in group_data["operations"]:
                    if op_type == "taxon": direct_taxa[group_name].add(operand)

            @lru_cache(maxsize=None)
            def resolve_group(group_name: str, visited: frozenset) -> Set[Tuple[str, str]]:
                """Recursively resolve a group to its constituent taxa."""
                if group_name not in raw_groups: self.logger.warning(f"Group '{group_name}' not found in raw groups"); return set()
                if group_name in visited: self.logger.warning(f"Circular dependency detected in group: {group_name}"); return set()

                new_visited = visited | {group_name}
                result = set(direct_taxa.get(group_name, set()))

                for op_type, operand in raw_groups.get(group_name, {}).get("operations", []):
                    if op_type == "add": result |= resolve_group(operand, new_visited)
                    elif op_type == "subtract": result -= resolve_group(operand, new_visited)
                    elif op_type == "intersect": result &= resolve_group(operand, new_visited)
                return result

            # === Pass 3: Build final database with resolved UNCOMPILED pattern strings ===
            resolved_db_patterns = {}
            all_groups_resolved = 0
            
            # Use progress bar for this loop
            with get_progress_bar() as progress:
                task = progress.add_task("Resolving FAPROTAX groups...", total=len(raw_groups))
                
                for group_name in raw_groups.keys():
                    progress.update(task, description=f"Resolving {group_name}...")
                    taxa = resolve_group(group_name, frozenset())

                    patterns = []
                    if taxa:
                        all_groups_resolved += 1
                        for taxon_pattern, reference in taxa:
                            # Convert FAPROTAX's wildcard '*' to regex '.*'
                            # Use re.escape to safely handle special chars
                            regex_pattern_string = ".*".join(re.escape(part) for part in taxon_pattern.split("*"))
                            patterns.append(regex_pattern_string)
                    
                    # Store the list of uncompiled pattern strings
                    resolved_db_patterns[group_name] = patterns
                    progress.update(task, advance=1)

            self.logger.info(f"Successfully resolved {all_groups_resolved} functional groups.")
            return resolved_db_patterns, group_metadata

        except FileNotFoundError: self.logger.error(f"Database file not found at {self.db_file}."); return {}, {}
        except Exception as e: raise ParserError(f"Failed to parse FAPROTAX.txt: {e}") from e

    @lru_cache(maxsize=4096)
    def predict_functions(self, taxonomy: str) -> List[str]:
        """
        Predicts functional groups for a single taxonomy string
        using the pre-compiled master regex.

        Args:
            taxonomy: A taxonomy string (semicolon-delimited or not).

        Returns:
            A sorted list of matching functional group names.
        """
        found_functions = set()
        
        # .finditer() finds *all* non-overlapping matches from the master regex
        # This is fast: one search, many results.
        for match in self._master_regex.finditer(taxonomy):
            # groupdict() returns {clean_name: match_text, ...}
            for clean_name, match_text in match.groupdict().items():
                if match_text: # If this named group participated in the match
                    # Map the clean_name back to the original FAPROTAX name
                    found_functions.add(self._clean_map[clean_name])

        return sorted(list(found_functions))

    def predict_functions_batch(self, taxa_list: List[str], processes: int | None = None) -> List[List[str]]:
        """
        Predicts functions for a list of taxa in parallel.

        Args:
            taxa_list: 
                A list of taxonomy strings. Duplicates are fine.
            processes: 
                Number of processes to use. Defaults to os.cpu_count().

        Returns:
            A list of function lists, in the same order as the input taxa_list.
        """
        # Ensure the master regex is compiled *before* forking processes
        # This triggers all the lazy-loading properties on the main thread.
        if self._master_regex_compiled is None:  _ = self._master_regex

        # We process the unique set of taxa to leverage the lru_cache and minimize work for the subprocesses.
        unique_taxa = sorted(list(set(taxa_list)))
        self.logger.info(f"Predicting functions for {len(unique_taxa)} unique taxa...")

        with Pool(processes=processes) as pool:
            # pool.map runs self.predict_functions (which is cached) on every item in unique_taxa
            results = pool.map(self.predict_functions, unique_taxa)

        self.logger.info("Batch prediction complete. Mapping results...")

        # Create a fast lookup map: {taxon_string: [function_list]}
        result_map = dict(zip(unique_taxa, results))

        # Map the results back to the *original* list order, correctly handling duplicates.
        final_results_list = [result_map[taxon] for taxon in taxa_list]

        return final_results_list

# ==================================================================================== #
# Main execution block for testing
# ==================================================================================== #
if __name__ == "__main__":
    # --- Configuration ---
    logging.basicConfig(
        level=logging.INFO, # Use INFO for a cleaner test run
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Set this script's logger to DEBUG to see resolving steps
    logging.getLogger('workflow_16s.faprotax').setLevel(logging.INFO)

    # --- Database Loading ---
    print("--- Initializing FaprotaxDB ---")
    db = FaprotaxDB()

    try:
        # Trigger the lazy-loading
        db_data = db.data
        metadata = db.metadata
        print(f"\n--- Database Loaded Successfully ---")
        print(f"Total functional groups: {len(db_data)}")
        print(f"Groups with metadata: {len(metadata)}")

        # Show some examples
        print("\n--- Example Functional Groups ---")
        if db_data:
            example_groups = [g for g, p in db_data.items() if p][:5] # Get first 5 non-empty
            print(f"Showing first 5 of {len([p for p in db_data.values() if p])} resolved groups:")
            for group in example_groups:
                print(f"- {group}")
        else:
            print("No functional groups were loaded.")

        # --- Example Predictions (Single) ---
        print("\n--- Example Predictions (Single-threaded) ---")

        test_taxonomies = [
            "k__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Pseudomonales;f__Pseudomonadaceae;g__Pseudomonas",
            "Bacteria;Proteobacteria;Gammaproteobacteria;Methylococcales;Methylococcaceae;Methylococcus",
            "Archaea;Euryarchaeota;Methanobacteria;Methanobacteriales;Methanobacteriaceae;Methanobacterium",
            "This;is;a;fake;taxonomy;string",
        ]

        for taxonomy in test_taxonomies:
            print(f"\nTaxonomy: {taxonomy}")
            # Use the single, cached prediction function
            functions = db.predict_functions(taxonomy)
            print(f"Functions found: {len(functions)}")
            for func in functions:
                print(f"  - {func}")
        
        # --- Example Predictions (Batch) ---
        print("\n--- Example Predictions (Batch-parallel) ---")
        
        # Create a large list with many duplicates
        large_taxa_list = (test_taxonomies * 1000) + \
                          (["d__Bacteria;p__Firmicutes;..."] * 500)
        
        print(f"Predicting functions for a large list of {len(large_taxa_list)} taxa...")
        
        # Use the fast, parallel batch method
        batch_results = db.predict_functions_batch(large_taxa_list)
        
        print(f"Batch prediction finished.")
        print(f"Total results in list: {len(batch_results)}")
        print(f"Functions for '...;g__Pseudomonas': {batch_results[0]}")
        print(f"Functions for '...;g__Methylococcus': {batch_results[1]}")
        print(f"Functions for '...;g__Methanobacterium': {batch_results[2]}")
        print(f"Functions for '...;fake;taxonomy': {batch_results[3]}")
        print(f"Functions for '...;p__Firmicutes': {batch_results[4000]}")


    except FaprotaxError as e:
        logger.error(f"A FAPROTAX error occurred: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)