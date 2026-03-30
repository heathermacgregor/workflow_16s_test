"""
workflow_16s: Modular microbial community analysis pipeline
============================================================

A comprehensive pipeline for 16S rRNA amplicon sequencing data analysis.

Modules:
--------
- upstream: Data retrieval, QC, and QIIME 2 processing
- downstream: Statistical analysis, ML, and visualization
- api: QIIME 2 and ENA API interactions
- metadata: Metadata enrichment and geospatial analysis
- qc: Quality control and validation
"""

__version__ = "2.0.0"
__author__ = "Heather MacGregor"

from .config import config_schema
from .utils.ui.logger import get_logger, with_logger, setup_logging

# Lazy imports - only import when used to avoid dependency issues
def get_config(*args, **kwargs):
    """Lazy import wrapper for config.get_config"""
    from workflow_16s.config import get_config as _get_config
    return _get_config(*args, **kwargs)

def setup_logging(*args, **kwargs):
    """Lazy import wrapper for logger.setup_logging"""
    from workflow_16s.logger import setup_logging as _setup_logging
    return _setup_logging(*args, **kwargs)

__all__ = [
    "__version__",
    "__author__",
    "config_schema",
    "get_config",
    "get_logger",
    "with_logger",
    "setup_logging",
]
