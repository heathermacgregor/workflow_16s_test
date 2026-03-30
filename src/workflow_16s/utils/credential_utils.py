"""
Standardized Credential Access Utility.

This module provides a unified interface for accessing credentials from configuration
with environment variable fallback. All credentials should be accessed through the
helper functions in this module to ensure consistency and security.

STANDARDIZATION RULES:
    1. Try: config.credentials.<credential_name> first
    2. Fallback to: os.getenv("<CREDENTIAL_NAME_UPPER>")
    3. Same naming: snake_case in config = UPPER_CASE in env
    4. Never write to os.environ (security risk)
    5. Raise clear errors for REQUIRED credentials if not found
    6. Return None gracefully for OPTIONAL credentials
"""

import os
import logging
from typing import Any, Optional

logger = logging.getLogger("workflow_16s")


def get_credential(
    config: Any,
    name: str,
    env_name: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """
    Retrieve a credential from config or environment with standardized fallback.

    Args:
        config: AppConfig instance containing credentials
        name: Credential name in config (snake_case, e.g., 'ncbi_api_key')
        env_name: Environment variable name (auto-uppercased if not provided)
        required: If True, raise error if credential not found

    Returns:
        Credential value or None if not found and not required

    Raises:
        ValueError: If credential is required but not found
    """
    # Auto-generate env name if not provided
    if env_name is None:
        env_name = name.upper()

    # Try config first
    if hasattr(config, 'credentials'):
        value = getattr(config.credentials, name, None)
        if value:
            return value

    # Fall back to environment variable
    value = os.getenv(env_name)
    if value:
        return value

    # Handle missing required credentials
    if required:
        raise ValueError(
            f"Required credential '{name}' not found in config.credentials.{name} "
            f"or environment variable {env_name}"
        )

    return None


def get_email(config: Any, which: str = "email") -> str:
    """
    Retrieve email credential with smart fallback.

    Args:
        config: AppConfig instance
        which: Either 'email' (default) or 'ena_email'

    Returns:
        Email address from config or environment

    Raises:
        ValueError: If email not found
    """
    # Try the specific email first
    if hasattr(config, 'credentials'):
        value = getattr(config.credentials, which, None)
        if value and value != "default@example.com":
            return value

    # Fall back to generic 'email' if ena_email requested but not set
    if which == "ena_email" and hasattr(config, 'credentials'):
        value = getattr(config.credentials, "email", None)
        if value and value != "default@example.com":
            return value

    # Try environment variables
    env_name = which.upper()
    value = os.getenv(env_name)
    if value:
        return value

    # Final fallback with error
    raise ValueError(
        f"Email credential not found. Set config.credentials.{which} "
        f"or environment variable {env_name}"
    )


# CREDENTIAL MAPPING: Maps credential names to their environment variable equivalents
CREDENTIAL_MAPPING = {
    # Email credentials
    "email": "EMAIL",
    "ena_email": "ENA_EMAIL",
    # API keys
    "ncbi_api_key": "NCBI_API_KEY",
    "nrel_api_key": "NREL_API_KEY",
    "openaq_api_key": "OPENAQ_API_KEY",
    "mindat_api_key": "MINDAT_API_KEY",
    "airnow_api_key": "AIRNOW_API_KEY",
    "usgs_api_key": "USGS_API_KEY",
    "opentopography_api_key": "OPENTOPOGRAPHY_API_KEY",
    "springer_api_key": "SPRINGER_NATURE_API_KEY",
    "ieee_api_key": "IEEE_XPLORE_API_KEY",
    "mendeley_api_key": "MENDELEY_API_KEY",
    "dimensions_api_key": "DIMENSIONS_API_KEY",
    # GEE credentials
    "google_earth_engine_project": "GOOGLE_EARTH_ENGINE_PROJECT",
    "gee_service_account": "GEE_SERVICE_ACCOUNT",
    # CMEMS credentials
    "cmems_username": "CMEMS_USERNAME",
    "cmems_password": "CMEMS_PASSWORD",
}


def validate_credentials_standardized(config: Any, required_credentials: list) -> dict:
    """
    Validate that all required credentials are available.

    Args:
        config: AppConfig instance
        required_credentials: List of credential names to check

    Returns:
        Dict with credential name -> value for all valid credentials

    Raises:
        ValueError: If any required credential is missing
    """
    result = {}
    missing = []

    for cred_name in required_credentials:
        env_name = CREDENTIAL_MAPPING.get(cred_name, cred_name.upper())
        value = get_credential(config, cred_name, env_name=env_name, required=False)

        if value:
            result[cred_name] = value
        else:
            missing.append(cred_name)

    if missing:
        raise ValueError(
            f"Missing required credentials: {', '.join(missing)}. "
            f"Set in config.credentials or environment variables."
        )

    return result
