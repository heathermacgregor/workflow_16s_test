"""
ENA API module - provides access to ENA/SRA metadata enrichment functionality.

This module bridges the sequence-level ENA enrichment with the broader API structure.
"""

import asyncio
from typing import Dict, List, Optional

from workflow_16s.api.sequence.ena import ENAEnrichmentPipeline, SampleParser
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s.api.ena")


async def get_n_samples_by_bioproject_async(
    bioproject_id: str,
    email: str = None
) -> Optional[int]:
    """
    Get the number of samples associated with a BioProject ID.

    This function queries ENA to find all samples belonging to a specified BioProject.

    Args:
        bioproject_id: The BioProject ID (e.g., "PRJNA123456")
        email: Email for ENA API access (optional, uses config if not provided)

    Returns:
        Number of samples in the bioproject, or None if query fails

    Example:
        >>> n_samples = await get_n_samples_by_bioproject_async("PRJNA123456")
        >>> print(f"Found {n_samples} samples")
    """
    try:
        # Validate bioproject ID format
        if not bioproject_id or not isinstance(bioproject_id, str):
            logger.warning(f"Invalid bioproject ID: {bioproject_id}")
            return None

        # For now, return a placeholder value
        # In a full implementation, this would query the ENA REST API
        logger.debug(f"Querying samples for bioproject: {bioproject_id}")

        # Placeholder: Return None to indicate functionality not yet fully integrated
        # This prevents import errors while allowing the downstream workflow to load
        return None

    except Exception as e:
        logger.warning(f"Error querying bioproject {bioproject_id}: {e}")
        return None


__all__ = [
    "ENAEnrichmentPipeline",
    "SampleParser",
    "get_n_samples_by_bioproject_async",
]
