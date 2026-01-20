# workflow_16s/api/ena/metadata/cache.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import aiofiles
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

# Local Imports
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class CacheManager:
    """Handles file-based caching (using JSON) for asynchronous network requests."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Metadata cache enabled at: {self.cache_dir.resolve()}")

    def get_cache_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """Generates a stable SHA256 hash for a given set of arguments."""
        sorted_kwargs = sorted(kwargs.items())
        representation = (prefix, args, sorted_kwargs)
        # Using JSON serialization for human-readable keys if needed, ensure sort_keys
        serialized = json.dumps(representation, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Reads and deserializes data from a cache file if it exists."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                    return json.loads(await f.read())
            except (json.JSONDecodeError, IOError, asyncio.TimeoutError) as e: # Added TimeoutError
                logger.warning(f"Could not read cache file {cache_file}: {e}. Treating as cache miss.")
                # Optional: Attempt to delete corrupt file
                try:
                    cache_file.unlink()
                    logger.debug(f"Deleted potentially corrupt cache file: {cache_file}")
                except OSError as unlink_err:
                    logger.error(f"Failed to delete corrupt cache file {cache_file}: {unlink_err}")
                return None
        return None

    async def set(self, key: str, data: Any) -> None:
        """Serializes and writes data to a cache file."""
        if not data: # Don't cache empty lists/dicts if they represent no results
            logger.debug(f"Skipping cache set for key {key} due to empty data.")
            return

        cache_file = self.cache_dir / f"{key}.json"
        try:
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2)) # Use indent for readability
        except (IOError, asyncio.TimeoutError) as e: # Added TimeoutError
            logger.error(f"Could not write to cache file {cache_file}: {e}")