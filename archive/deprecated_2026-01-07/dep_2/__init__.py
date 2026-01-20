from collections import defaultdict
from typing import Any, Dict, List, Optional
from biom.table import Table
import pandas as pd

class Data:
    """A container for storing and managing datasets for the workflow."""

    def __init__(self):
        # Stores BIOM tables, e.g., self.tables['raw']['genus']
        self.tables: Dict[str, Dict[str, Table]] = defaultdict(dict)
        # Manages a single, unified metadata DataFrame for the project
        self.metadata: Optional[pd.DataFrame] = None
        self.nfc_facilities: Any = None
        self.env_data: Any = None
        self.analysis_columns: Dict[str, List[str]] = {}
        # This lambda function creates a defaultdict that, when a new key is
        # accessed, creates another defaultdict, allowing for arbitrarily deep nesting.
        tree = lambda: defaultdict(tree)
        self.analysis_results = tree()