# workflow_16s/src/workflow_16s/utils/progress.py

from datetime import timedelta
from typing import Optional

from rich.progress import (
    BarColumn, 
    Progress, 
    ProgressColumn, 
    SpinnerColumn, 
    Task, 
    TextColumn, 
)
from rich.table import Column
from rich.text import Text

# See supported colors at: https://www.w3schools.com/colors/colors_x11.asp

# Total character width of the progress bar text
DEFAULT_PROGRESS_TEXT_N: int = 65
DEFAULT_N: int = 65 
# Color of the progress bar description text
DEFAULT_DESCRIPTION_STYLE: str = "white"
# Width of the progress bar
DEFAULT_BAR_WIDTH: int = 40
# Color of the filled/complete portion of the progress bar
DEFAULT_BAR_COLUMN_COMPLETE_STYLE: str = "honeydew2"
# Color used when the progress bar is finished
DEFAULT_FINISHED_STYLE: str = "dark_cyan" 
# Color of the percentage complete text (e.g., "85%")
DEFAULT_PROGRESS_PERCENTAGE_STYLE: str = "honeydew2"
# Color of the "X of Y complete" text (e.g., "42 of 65")
DEFAULT_M_OF_N_COMPLETE_STYLE: str = "honeydew2"
# Color of the time elapsed display (e.g., "E: 00:01:25")
DEFAULT_TIME_ELAPSED_STYLE: str = "light_sky_blue1"
# Color of the estimated time remaining display (e.g., "R: 00:00:34")
DEFAULT_TIME_REMAINING_STYLE: str = "thistle1"

# ============================== CUSTOM PROGRESS COLUMNS ============================= #

class MofNCompleteColumn(ProgressColumn):
    """Renders completed count/total (e.g., '3/10') with bold styling"""
    def render(self, task: Task) -> Text:
        """Render the progress count as 'completed/total'"""
        return Text(f"{int(task.completed)}/{int(task.total)}".rjust(10), # type: ignore
                    style=DEFAULT_M_OF_N_COMPLETE_STYLE, justify="right")


class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""
    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None: return Text("-:--:--", style=DEFAULT_TIME_ELAPSED_STYLE)
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(str(delta), style=DEFAULT_TIME_ELAPSED_STYLE)


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
            style = DEFAULT_TIME_ELAPSED_STYLE
        else:
            task_time = task.time_remaining
            style = DEFAULT_TIME_REMAINING_STYLE

        if task.total is None: return Text("", style=style)
        if task_time is None: return Text("--:--" if self.compact else "-:--:--", style=style)

        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours: formatted = f"{minutes:02d}:{seconds:02d}"
        else: formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style=style)


class TaskDescriptionColumn(ProgressColumn):
    """A column for rendering the task description using a custom format."""
    def render(self, task: "Task") -> Text:
        """Render the task description by calling the formatting function."""
        return Text.from_markup(
            _format_task_desc(task.description), 
            style=DEFAULT_DESCRIPTION_STYLE, 
            justify="left"
        )


# ===================================== FUNCTIONS ==================================== #

def _format_task_desc(desc: str) -> str:
    """Formats the task description with padding and color."""
    return f"[white]{str(desc):<{DEFAULT_N}}[/white]"


def get_progress_bar(transient: bool = False) -> Progress:
    """Return a customized progress bar with consistent styling"""
    return Progress(
        SpinnerColumn("dots", style=DEFAULT_BAR_COLUMN_COMPLETE_STYLE, speed=0.75),
        TaskDescriptionColumn(),
        MofNCompleteColumn(),
        BarColumn(
            bar_width=DEFAULT_BAR_WIDTH, style="black", 
            complete_style=DEFAULT_BAR_COLUMN_COMPLETE_STYLE,
            finished_style=DEFAULT_FINISHED_STYLE
        ),
        TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%".rjust(5),
            style=DEFAULT_PROGRESS_PERCENTAGE_STYLE, 
            justify="right"
        ),
        TextColumn(
            "E".rjust(2), 
            style=DEFAULT_TIME_ELAPSED_STYLE, 
            justify="right"
        ),
        TimeElapsedColumn(),
        TextColumn(
            "R".rjust(2), 
            style=DEFAULT_TIME_REMAINING_STYLE, 
            justify="right"
        ),
        TimeRemainingColumn(),
        transient=transient,
        expand=False
    )
