# workflow_16s/downstream/utils/__init__.py

"""
Utility modules for the 16S Downstream Analysis Workflow.
Includes reporting, consolidation, and auxiliary helper functions.
"""

from .reporting import generate_synthesis_report

# Defining __all__ ensures that 'from workflow_16s.downstream.utils import *' 
# only exports the specific intended functions.
__all__ = [
    'generate_synthesis_report'
]