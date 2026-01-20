# workflow_16s/api/ena/cache.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import aiofiles
import asyncio
import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

# Local Imports
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class CacheManager:
    """
    Handles file-based caching using pickle with a Time-To-Live (TTL).
    
    Attributes:
        cache_dir (Path): Directory where cache files are stored.
        ttl (int):        Time-to-live for cache entries in seconds.
    """
    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400): # Default TTL of 1 day
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache enabled at directory: {self.cache_dir.resolve()} with TTL: {self.ttl}s")

    def get_cache_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """Generates a stable SHA256 hash for a given set of arguments."""
        sorted_kwargs = sorted(kwargs.items())
        representation = (prefix, args, sorted_kwargs)
        serialized = pickle.dumps(representation)
        return hashlib.sha256(serialized).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'rb') as f:
                    payload = pickle.loads(await f.read())
                
                if time.time() - payload.get('timestamp', 0) > self.ttl:
                    logger.debug(f"Cache STALE for key {key}. Deleting.")
                    cache_file.unlink() # Delete stale file
                    return None

                return payload.get('data')

            except (EOFError, pickle.UnpicklingError) as e:
                # Self-heal by deleting the corrupt file
                logger.warning(f"Corrupt cache file {cache_file} ({e}). Deleting it.")
                try: cache_file.unlink()
                except OSError as unlink_e:
                    logger.error(f"Failed to delete corrupt cache file: {unlink_e}")
                return None # Treat as a cache MISS
                
            except (IOError, KeyError) as e:
                logger.warning(f"Could not read cache file {cache_file}: {e}")
                return None
        return None

    async def set(self, key: str, data: Any) -> None:
        """Serializes and writes data to a cache file with a timestamp."""
        if not data:
            logger.debug(f"Skipping cache write for key {key} because data is empty.")
            return  # Don't cache empty results from failed requests

        cache_file = self.cache_dir / f"{key}.pkl"
        
        # Wrap the data in a payload with a timestamp for TTL checks
        payload = {'timestamp': time.time(), 'data': data}
        
        try:
            serialized_payload = pickle.dumps(payload)
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(serialized_payload)
        except IOError as e:
            logger.error(f"Could not write to cache file {cache_file}: {e}")

    def clear_expired(self):
        """Synchronously iterates through the cache and removes expired files."""
        logger.debug("Starting cache cleanup of expired files...")
        now = time.time()
        expired_count = 0
        total_files = 0
        for cache_file in self.cache_dir.glob('*.pkl'):
            total_files += 1
            try:
                with open(cache_file, 'rb') as f: payload = pickle.load(f)
                
                if now - payload.get('timestamp', 0) > self.ttl:
                    cache_file.unlink()  # Delete the expired file
                    expired_count += 1
            except (pickle.UnpicklingError, IOError, KeyError, EOFError):
                # File might be corrupt or malformed, delete it
                logger.warning(f"Removing corrupt cache file: {cache_file}")
                cache_file.unlink()
                expired_count += 1
                continue
        logger.debug(f"Cache cleanup complete. Removed {expired_count} of {total_files} files.")