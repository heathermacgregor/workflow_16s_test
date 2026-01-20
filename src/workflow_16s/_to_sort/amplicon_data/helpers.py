# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =============================== HELPER FUNCTIONS =================================== #

def _init_dict_level(a, b, c=None, d=None, e=None):
    if b not in a:
        a[b] = {}
    if c and c not in a[b]:
        a[b][c] = {}
    if d and d not in a[b][c]:
        a[b][c][d] = {}
    if e and e not in a[b][c][d]:
        a[b][c][d][e] = {}


class _ProcessingMixin:
    """
    Provides reusable methods for processing steps with progress tracking and logging.
    """
    
    def _run_processing_step(
        self,
        process_name: str,
        process_func: Callable,
        levels: List[str],
        func_args: tuple,
        get_source: Callable[[str], Table],
        log_template: Optional[str] = None,
        log_action: Optional[str] = None,
    ) -> Dict[str, Table]:
        processed: Dict[str, Table] = {}

        if getattr(self, "verbose", False):
            for level in levels:
                start_time = time.perf_counter() 
                processed[level] = process_func(get_source(level), level, *func_args)
                duration = time.perf_counter() - start_time
                if log_template or log_action:
                    self._log_level_action(level, log_template, log_action, duration)
        else:
            with get_progress_bar() as progress:
                parent_desc = f"{process_name}"
                parent_task = progress.add_task(_format_task_desc(parent_desc), total=len(levels))
                
                for level in levels:
                    level_desc = f"{parent_desc} ({level})"
                    progress.update(parent_task, description=_format_task_desc(level_desc))
                    
                    start_time = time.perf_counter()  
                    processed[level] = process_func(get_source(level), level, *func_args)
                    duration = time.perf_counter() - start_time
                    if log_template or log_action:
                        self._log_level_action(level, log_template, log_action, duration)

                    progress.update(parent_task, advance=1)
            progress.update(parent_task, description=_format_task_desc(parent_desc))
        return processed

    def _log_level_action(
        self,
        level: str,
        template: Optional[str] = None,
        action: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        message = ""
        if template:
            message = template.format(level=level)
        elif action:
            message = f"{level} {action}"

        if message and duration is not None:
            message += f" in {duration:.2f}s"

        if message:
            logger.debug(message)
