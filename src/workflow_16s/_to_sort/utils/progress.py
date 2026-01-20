# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from datetime import timedelta
from typing import Any, Optional

# Third-Party Imports
from rich.progress import (
    BarColumn, Column, Progress, ProgressColumn, SpinnerColumn, Task, TaskID,
    TextColumn, track
)
from rich.text import Text

# Local Imports
from workflow_16s import constants

# ============================== CUSTOM PROGRESS COLUMNS ============================= #

class MofNCompleteColumn(ProgressColumn):
    """Renders completed count/total (e.g., '3/10') with bold styling"""
    
    def render(self, task: Task) -> Text:
        """Render the progress count as 'completed/total'"""
        return Text(
            f"{task.completed}/{task.total}".rjust(10),
            style=constants.DEFAULT_M_OF_N_COMPLETE_STYLE,
            justify="right"
        )
        

class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""
    
    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style=constants.DEFAULT_TIME_ELAPSED_STYLE)
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(str(delta), style=constants.DEFAULT_TIME_ELAPSED_STYLE)
        

class TimeRemainingColumn(ProgressColumn):
    """Renders estimated time remaining."""
    max_refresh = 0.5 # Only refresh twice a second to prevent jitter

    def __init__(
        self,
        compact: bool = False,
        elapsed_when_finished: bool = False,
        table_column: Optional[Column] = None,
    ):
        self.compact = compact
        self.elapsed_when_finished = elapsed_when_finished
        super().__init__(table_column=table_column)


    def render(self, task: "Task") -> Text:
        """Show time remaining."""
        if self.elapsed_when_finished and task.finished:
            task_time = task.finished_time
            style = constants.DEFAULT_TIME_ELAPSED_STYLE
        else:
            task_time = task.time_remaining
            style = constants.DEFAULT_TIME_REMAINING_STYLE

        if task.total is None:
            return Text("", style=style)

        if task_time is None:
            return Text("--:--" if self.compact else "-:--:--", style=style)

        # Based on https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style=style)


# ===================================== FUNCTIONS ==================================== #

def get_progress_bar(transient: bool = False) -> Progress:
    """Return a customized progress bar with consistent styling"""
    return Progress(
        SpinnerColumn(
            "dots", 
            style=constants.DEFAULT_BAR_COLUMN_COMPLETE_STYLE, 
            speed=0.75
        ),
        TextColumn(
            "{task.description}".ljust(constants.DEFAULT_PROGRESS_TEXT_N), 
            style=constants.DEFAULT_DESCRIPTION_STYLE,
            justify="left"
        ),
        MofNCompleteColumn(),
        BarColumn(
            bar_width=constants.DEFAULT_BAR_WIDTH,
            style="black", # Background color
            complete_style=constants.DEFAULT_BAR_COLUMN_COMPLETE_STYLE,
            finished_style=constants.DEFAULT_FINISHED_STYLE
        ),
        TextColumn(
            "{task.percentage:>3.0f}%".rjust(5), 
            style=constants.DEFAULT_PROGRESS_PERCENTAGE_STYLE,
            justify="right"
        ),
        TextColumn(
            "E".rjust(2), 
            style=constants.DEFAULT_TIME_ELAPSED_STYLE,
            justify="right"
        ),
        TimeElapsedColumn(),
        TextColumn(
            "R".rjust(2), 
            style=constants.DEFAULT_TIME_REMAINING_STYLE,
            justify="right"
        ),
        TimeRemainingColumn(),
        transient=transient,
        expand=False
    )


def _format_task_desc(desc: str):
    return f"[white]{str(desc):<{constants.DEFAULT_N}}"
