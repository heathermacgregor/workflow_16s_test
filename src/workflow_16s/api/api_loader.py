"""
API Loader - Config-driven API Registry & Factory

Provides runtime control over which APIs are available based on user configuration.
Ensures efficient initialization and caching of API clients.

Location: workflow_16s/src/workflow_16s/api/api_loader.py

Copy this file to the location above when ready to implement.
"""

from typing import Dict, Any, Optional, Type, List
from pathlib import Path
import logging

from workflow_16s.utils.logger import get_logger

logger = get_logger(__name__)


class APIRegistry:
    """Central registry for all available APIs, respects user configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize API Registry.
        
        Args:
            config: Configuration dict (from config.yaml). 
                   If None, loads from default location.
        """
        if config is None:
            from workflow_16s.config import load_config
            config = load_config()
        
        self.config = config
        self.apis_config = config.get('apis', {})
        self._cache: Dict[str, Any] = {}
        self._initialized = set()
    
    def is_enabled(self, category: str, api_name: Optional[str] = None) -> bool:
        """
        Check if an API or category is enabled.
        
        Args:
            category: API category name (e.g., 'gee', 'sequence')
            api_name: Specific API name (e.g., 'era5'). 
                     If None, checks category-level enabled flag.
        
        Returns:
            True if API/category is enabled; False otherwise.
        
        Examples:
            >>> registry = APIRegistry()
            >>> registry.is_enabled('gee')  # Is GEE category enabled?
            True
            >>> registry.is_enabled('gee', 'era5')  # Is ERA5 dataset enabled?
            True
            >>> registry.is_enabled('publication', 'crossref')  # Is Crossref enabled?
            False
        """
        # Master switch
        if not self.apis_config.get('enabled', True):
            return False
        
        # Category-level check
        category_config = self.apis_config.get(category, {})
        if not category_config.get('enabled', False):
            return False
        
        # API-level check
        if api_name:
            datasets = category_config.get('datasets', {})
            api_config = category_config.get(api_name, {})
            
            # Check in datasets dict (for GEE, USGS, etc.)
            if api_name in datasets:
                return datasets[api_name]
            
            # Check as nested config (for ENA, QIIME, etc.)
            if isinstance(api_config, dict):
                return api_config.get('enabled', True)
            
            return False
        
        return True
    
    def get_config(self, category: str, api_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for an API.
        
        Args:
            category: API category name
            api_name: Specific API name (optional)
        
        Returns:
            Configuration dictionary for the API.
        
        Examples:
            >>> config = registry.get_config('gee', 'era5')
            >>> print(config.get('batch_size'))  # GEE-level setting
            100
        """
        category_config = self.apis_config.get(category, {})
        
        if api_name:
            # Try nested config first (ENA, QIIME)
            if api_name in category_config and isinstance(category_config[api_name], dict):
                return category_config[api_name]
            
            # Try datasets dict (GEE, USGS)
            datasets = category_config.get('datasets', {})
            if api_name in datasets:
                return {
                    'enabled': datasets[api_name],
                    **{k: v for k, v in category_config.items() 
                       if k not in ['enabled', 'datasets', api_name]}
                }
        
        return category_config or {}
    
    def load_api(self, category: str, api_name: str, 
                 api_class: Type) -> Optional[Any]:
        """
        Load an API client if enabled, cache result.
        
        Args:
            category: API category name
            api_name: API name within category
            api_class: Python class to instantiate
        
        Returns:
            Initialized API client, or None if disabled.
        
        Raises:
            ImportError: If api_class cannot be imported
            Exception: If API initialization fails (logs error)
        
        Examples:
            >>> from workflow_16s.api.gee.era5 import ERA5API
            >>> registry = APIRegistry()
            >>> era5 = registry.load_api('gee', 'era5', ERA5API)
            >>> if era5:
            ...     data = era5.fetch(coords)
            ... else:
            ...     logger.info("ERA5 disabled, skipping")
        """
        cache_key = f"{category}.{api_name}"
        
        # Return cached instance
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check if enabled
        if not self.is_enabled(category, api_name):
            logger.debug(f"API {cache_key} disabled in config, skipping load")
            return None
        
        # Load API
        try:
            config = self.get_config(category, api_name)
            instance = api_class(config=config)
            self._cache[cache_key] = instance
            logger.info(f"Loaded API: {cache_key}")
            return instance
        
        except Exception as e:
            logger.error(f"Failed to load API {cache_key}: {e}", exc_info=True)
            return None
    
    def get_enabled_apis(self, category: str) -> Dict[str, bool]:
        """
        Get all enabled APIs in a category.
        
        Args:
            category: API category name (e.g., 'gee', 'sequence')
        
        Returns:
            Dict of {api_name: enabled_bool} for all ENABLED APIs in category.
        
        Examples:
            >>> enabled = registry.get_enabled_apis('gee')
            >>> print(list(enabled.keys()))
            ['era5', 'copernicus_dem', 'world_cover', ...]
        """
        category_config = self.apis_config.get(category, {})
        datasets = category_config.get('datasets', {})
        
        enabled = {}
        
        # Add datasets-based APIs (GEE, USGS)
        for api_name, is_enabled in datasets.items():
            if is_enabled:
                enabled[api_name] = True
        
        # Add config-based APIs (ENA, QIIME)
        for key, value in category_config.items():
            if key not in ['enabled', 'datasets', 'cache_enabled', 'auth_required', 
                          'batch_size', 'max_workers', 'cache_ttl_days']:
                if isinstance(value, dict) and value.get('enabled', True):
                    enabled[key] = True
        
        return enabled
    
    def validate_requirements(self) -> Dict[str, List[str]]:
        """
        Validate that required APIs/auth are available.
        
        Returns:
            Dict of {category: [warning_messages]} for validation issues.
            Empty dict if all OK.
        
        Examples:
            >>> warnings = registry.validate_requirements()
            >>> if warnings:
            ...     for category, msgs in warnings.items():
            ...         logger.warning(f"{category}: {', '.join(msgs)}")
        """
        warnings = {}
        
        # Check GEE auth if enabled
        if self.is_enabled('gee'):
            gee_config = self.get_config('gee')
            if gee_config.get('auth_required'):
                try:
                    from workflow_16s.config import load_config
                    config = load_config()
                    creds = config.get('credentials', {})
                    if not creds.get('google_earth_engine_project'):
                        warnings['gee'] = ['google_earth_engine_project not configured in credentials']
                except Exception as e:
                    warnings['gee'] = [f'credentials error: {e}']
        
        # Check other APIs as needed
        # ... (expand as more APIs require validation)
        
        return warnings
    
    def report(self) -> str:
        """
        Generate a human-readable report of API status.
        
        Returns:
            Formatted string report of enabled/disabled APIs by category
        
        Examples:
            >>> print(registry.report())
            ====== API Status Report ======
            GEE:
              ✓ era5
              ✓ world_cover
              ✗ dynamic_world (disabled)
            ...
        """
        report_lines = ["=" * 50, "API Status Report", "=" * 50, ""]
        
        for category in ['gee', 'usgs', 'environmental', 'sequence', 'publication', 
                         'facility', 'llm', 'geospatial', 'metadata']:
            category_config = self.apis_config.get(category, {})
            if not category_config:
                continue
            
            is_enabled = category_config.get('enabled', False)
            status = "✓ ENABLED" if is_enabled else "✗ DISABLED"
            report_lines.append(f"{category.upper():15} {status}")
            
            if is_enabled:
                # List enabled APIs
                enabled_apis = self.get_enabled_apis(category)
                for api_name in sorted(enabled_apis.keys()):
                    report_lines.append(f"  • {api_name}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


# ===== Singleton Pattern =====

_registry_instance: Optional[APIRegistry] = None

def get_api_registry(config: Optional[Dict[str, Any]] = None) -> APIRegistry:
    """
    Get global API registry instance (singleton pattern).
    
    Ensures only one registry instance per runtime.
    
    Args:
        config: Configuration dict (optional, loads default if not provided)
    
    Returns:
        APIRegistry instance (same instance on subsequent calls)
    
    Examples:
        >>> registry = get_api_registry()
        >>> registry.is_enabled('gee', 'era5')
        True
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = APIRegistry(config)
    return _registry_instance


def reset_registry(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Reset API registry (useful for testing).
    
    Args:
        config: New configuration dict (optional)
    """
    global _registry_instance
    _registry_instance = APIRegistry(config)


# ===== Convenience Functions =====

def is_api_enabled(category: str, api_name: Optional[str] = None) -> bool:
    """
    Quick check if API is enabled (uses singleton registry).
    
    Args:
        category: API category name
        api_name: Specific API name (optional)
    
    Returns:
        True if enabled; False otherwise.
    
    Examples:
        >>> if is_api_enabled('gee', 'era5'):
        ...     from workflow_16s.api.gee import ERA5API
        ...     api = ERA5API()
    """
    return get_api_registry().is_enabled(category, api_name)


def get_enabled_apis(category: str) -> Dict[str, bool]:
    """
    Get all enabled APIs in a category (uses singleton registry).
    
    Args:
        category: API category name
    
    Returns:
        Dict of {api_name: True} for all enabled APIs in category.
    
    Examples:
        >>> gee_apis = get_enabled_apis('gee')
        >>> print(f"Enabled GEE datasets: {list(gee_apis.keys())}")
    """
    return get_api_registry().get_enabled_apis(category)


def get_api_config(category: str, api_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for an API (uses singleton registry).
    
    Args:
        category: API category name
        api_name: Specific API name (optional)
    
    Returns:
        Configuration dictionary for the API.
    """
    return get_api_registry().get_config(category, api_name)


def log_enabled_apis() -> None:
    """
    Log all enabled APIs for debugging/monitoring.
    
    Logs at INFO level with nice formatting.
    
    Examples:
        >>> log_enabled_apis()
        # Logs:
        # ====================================================
        # ENABLED APIs:
        # ====================================================
        #   ✓ gee              - era5
        #   ✓ gee              - world_cover
        #   ✓ sequence         - ena
        # ...
    """
    registry = get_api_registry()
    logger.info("=" * 60)
    logger.info("ENABLED APIS:")
    logger.info("=" * 60)
    
    for category in ['gee', 'usgs', 'environmental', 'sequence', 'publication', 
                     'facility', 'llm', 'geospatial', 'metadata']:
        enabled = registry.get_enabled_apis(category)
        if enabled:
            for api_name in enabled:
                logger.info(f"  ✓ {category:15} - {api_name}")
    
    logger.info("=" * 60)


def validate_api_config() -> Dict[str, List[str]]:
    """
    Validate API configuration and return any warnings.
    
    Returns:
        Dict of {category: [warning_messages]}.
        Empty dict if all OK.
    
    Examples:
        >>> warnings = validate_api_config()
        >>> if warnings['gee']:
        ...     logger.warning(f"GEE config issues: {warnings['gee']}")
    """
    return get_api_registry().validate_requirements()


if __name__ == '__main__':
    # Demo usage
    registry = get_api_registry()
    print(registry.report())
    print("\nValidation:", validate_api_config())
