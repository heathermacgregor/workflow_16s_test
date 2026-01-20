# ===================================== IMPORTS ====================================== #

# Standard library imports
from __future__ import annotations
import logging
import os
import re
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Pattern, TextIO, Tuple, Union
from urllib.parse import urljoin

# Third-party imports
import requests
from bs4 import BeautifulSoup
from rich.progress import Progress, TaskID

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")  # Suppress warnings

FaprotaxDB = Dict[str, Dict[str, Union[Dict[str, str], List[Dict[str, Union[str, Pattern[str]]]]]]]

# ==================================== FUNCTIONS ===================================== #

def find_references_dir(project_name: str = "workflow_16s") -> Path:
    """
    Locate '<project_name>/references' directory by walking upward from 
    CWD/script dir.
    
    Returns:
        Path: Absolute path to references directory
        
    Raises:
        FileNotFoundError: If directory cannot be found
    """
    start = (
        Path(__file__).resolve().parent
        if "__file__" in globals()
        else Path.cwd().resolve()
    )

    for parent in [start] + list(start.parents):
        if parent.name == project_name:
            ref_dir = parent / "references"
            if ref_dir.is_dir():
                return ref_dir

    raise FileNotFoundError(
        f"Could not locate '{project_name}/references' directory "
        f"starting from {start}"
    )


def _scrape_latest_zip_url(base_url: str) -> Tuple[str, str]:
    """Scrape download page for newest FAPROTAX zip URL and version string."""
    soup = BeautifulSoup(
        requests.get(base_url, timeout=30).text, 
        "html.parser"
    )
    zip_links: List[Tuple[str, Tuple[int, ...]]] = []
    rgx = re.compile(r"FAPROTAX_(\d+(?:\.\d+)*)\.zip$", re.I)

    for a in soup.find_all("a", href=True):
        if (m := rgx.search(a["href"])) is not None:
            version_tuple = tuple(int(p) for p in m.group(1).split("."))
            zip_links.append((urljoin(base_url, a["href"]), version_tuple))

    if not zip_links:
        raise RuntimeError("No FAPROTAX *.zip links found on download page")

    zip_links.sort(key=lambda x: x[1], reverse=True)
    url, ver = zip_links[0]
    return url, ".".join(map(str, ver))


def _extract_faprotax_txt(zip_path: Path, out_folder: Path) -> Path:
    """
    Extract FAPROTAX.txt from zip archive to output folder.
    
    Returns:
        Path: Path to extracted FAPROTAX.txt file
    """
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [m for m in zf.namelist() if m.endswith("FAPROTAX.txt")]
        if not txt_members:
            raise RuntimeError("FAPROTAX.txt not found in ZIP archive")
        
        extracted = zf.extract(txt_members[0], path=out_folder)
        dst = out_folder / "FAPROTAX.txt"
        Path(extracted).replace(dst)  # Overwrite/standardize name
        return dst


def download_latest_faprotax(destination: Path) -> Path:
    """
    Ensure FAPROTAX.txt exists in destination directory (downloads if needed).
    
    Args:
        destination: Directory where FAPROTAX files should be stored
        
    Returns:
        Path: Path to up-to-date FAPROTAX.txt file
    """
    base_url = (
        "https://pages.uoregon.edu/slouca/LoucaLab/archive/"
        "FAPROTAX/lib/php/index.php?section=Download"
    )
    destination.mkdir(parents=True, exist_ok=True)

    txt_path = destination / "FAPROTAX.txt"
    if txt_path.exists():
        return txt_path  # Already cached

    zip_url, version = _scrape_latest_zip_url(base_url)
    zip_path = destination / Path(zip_url.split("?")[0]).name

    logger.info("Downloading FAPROTAX v%s from %s", version, zip_url)
    zip_path.write_bytes(requests.get(zip_url, timeout=120).content)

    logger.info("Extracting FAPROTAX.txt to %s", destination)
    return _extract_faprotax_txt(zip_path, destination)


def _yield_faprotax_records(fh: TextIO) -> Iterator[Tuple[str, str]]:
    """
    Yield pairs of (header, line) from FAPROTAX file handle.
    
    Header lines contain metadata (; or key:value pairs) without leading spaces
    """
    header: str | None = None
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Heuristic for header line
        if (
            (";" in line or "elements:" in line or "exclusively_prokaryotic" in line)
            and not raw.startswith(" ")
        ):
            header = line
            yield header, "__HEADER__"
        else:
            if header is None:
                continue  # Skip lines before header
            yield header, line


def _metadata_kv_iter(blob: str) -> Iterator[Tuple[str, str]]:
    """
    Parse metadata blob into key-value pairs.
    
    Format: "key1:value1; key2:value2; ..."
    """
    for part in re.split(r"[;\s]+", blob):
        if not part.strip():
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            yield k.strip(), v.strip()
        elif "=" in part:
            k, v = part.split("=", 1)
            yield k.strip(), v.strip()
        else:
            yield part.strip(), ""  # Key with empty value


def parse_faprotax_db(path: str | Path, *, compile_regex: bool = True) -> FaprotaxDB:
    """
    Parse FAPROTAX.txt file into structured dictionary.
    
    Args:
        path:          Path to FAPROTAX.txt file
        compile_regex: Compile patterns with re.compile() for faster matching
        
    Returns:
        Dictionary mapping valid trait names to metadata and taxa patterns
    """
    trait_dict: FaprotaxDB = {}
    current_trait: str | None = None
    path = Path(path)

    with path.open(encoding="utf-8") as fh:
        for header, line in _yield_faprotax_records(fh):
            if line == "__HEADER__":
                parts = header.split(maxsplit=1)
                trait = parts[0]
                meta_blob = parts[1] if len(parts) > 1 else ""
                
                # Skip traits with suspicious names (e.g., *Chondromyces*crocatus*)
                if re.search(r"^\*.*\*$", trait) or ";" in trait:
                    logger.debug(f"Skipping invalid trait: {trait}")
                    current_trait = None
                    continue
                    
                trait_dict[trait] = {
                    "metadata": dict(_metadata_kv_iter(meta_blob)),
                    "taxa": [],
                }
                current_trait = trait
                continue

            # Only process pattern lines for valid traits
            if current_trait is None:
                continue

            # Handle pattern lines
            fields = line.split(None, 1)
            pattern_raw = fields[0].strip()
            ref = fields[1][2:].strip() if len(fields) > 1 and fields[1].startswith("//") else ""

            # Skip empty/invalid patterns
            if not pattern_raw or pattern_raw == "*":
                continue

            # Convert FAPROTAX pattern to regex
            regex_str = re.sub(r"\*+", ".*", pattern_raw)  # Handle wildcards
            regex_str = f"^{regex_str}$"  # Match full taxonomic strings
            
            if compile_regex:
                try:
                    regex_pat = re.compile(regex_str, re.IGNORECASE)
                except re.error:
                    logger.warning(f"Invalid regex pattern for {current_trait}: {regex_str}")
                    continue
            else:
                regex_pat = regex_str

            trait_dict[current_trait]["taxa"].append({
                "pat": regex_pat,
                "ref": ref
            })

    # Remove traits with no valid patterns
    return {t: d for t, d in trait_dict.items() if d["taxa"]}

def get_faprotax_parsed(*, compile_regex: bool = True) -> FaprotaxDB | None:
    """
    Locate, download, and parse latest FAPROTAX database.
    
    Returns:
        Parsed FAPROTAX dictionary or None on failure
    """
    try:
        references_dir = find_references_dir()
        faprotax_dir = references_dir / "faprotax"
        faprotax_txt = download_latest_faprotax(faprotax_dir)
        return parse_faprotax_db(faprotax_txt, compile_regex=compile_regex)
    except Exception as exc:
        import traceback
        logger.exception(
            "Failed to prepare FAPROTAX database: %s\n%s", exc, traceback.format_exc()
        )
        return None


def faprotax_functions_for_taxon(
    taxon: str,
    faprotax_db: FaprotaxDB,
    *,
    include_references: bool = False,
) -> Union[List[str], Dict[str, List[str]]]:
    """Find all FAPROTAX functions matching a taxonomy string."""
    # Normalize taxon string by removing level markers and standardizing
    taxon_norm = re.sub(r'\s*[a-z]__\s*', ';', taxon.strip().lower())
    taxon_norm = re.sub(r'[;]+', ';', taxon_norm).strip(';')
    
    if include_references:
        trait_to_refs: Dict[str, List[str]] = {}
    else:
        traits: List[str] = []

    for trait, entry in faprotax_db.items():
        for rec in entry["taxa"]:
            pat = rec["pat"]
            ref = rec["ref"]

            # FIXED: Handle both compiled patterns and strings consistently
            if isinstance(pat, str):
                pat = re.compile(pat, re.IGNORECASE)

            # FIXED: Use search() with normalized taxon string
            if pat.search(taxon_norm):
                if include_references:
                    trait_to_refs.setdefault(trait, []).append(ref)
                else:
                    traits.append(trait)
                break  # One match per trait is sufficient

    return trait_to_refs if include_references else list(dict.fromkeys(traits))
