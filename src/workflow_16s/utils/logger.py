# ===================================== IMPORTS ====================================== #

# Standard Library
from datetime import datetime
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

# 3rd‑party (Rich)
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.theme import Theme

# ==================================== FUNCTIONS ===================================== #

def setup_logging(log_dir_path: Union[str, Path], log_filename: Union[str, None] = None,
                  max_file_size: int = 5 * 1024 * 1024,  # 5 MB
                  backup_count: int = 3, console_level: int = logging.INFO, 
                  file_level: int = logging.DEBUG,
                  quiet_init: bool = False) -> Path: # <--- MODIFICATION 2: ADD quiet_init AND RETURN Path
    """Configure unified logging with:
        • Colorful Rich console output with consistent formatting
        • Rotating file handler for full DEBUG logs
        • Progress bars integrated into logging system
    """
    # ───────────────────── log‑file path ──────────────────────
    log_dir_path = Path(log_dir_path)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if log_filename is None: 
        log_filename = datetime.now().strftime("%Y-%m-%d_%H%M%S.log")
    log_file_path = log_dir_path / log_filename

    # ─────────────────── root / package logger ─────────────────
    logger = logging.getLogger("workflow_16s")
    logger.setLevel(logging.DEBUG)  # Keep everything
    logger.propagate = False   

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # ───────────────────────── FILE HANDLER ───────────────────
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf‑8",
    )
    file_handler.setLevel(file_level)
    file_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s:%(filename)s:%(funcName)s(): %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S",)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    # Custom theme for Rich console
    custom_theme = Theme({"logging.time": "dim cyan",
                          "logging.level.info": "bold white",
                          "logging.level.debug": "dim cyan",
                          "logging.level.warning": "bold yellow",
                          "logging.level.error": "bold red",
                          "logging.level.critical": "reverse bold bright_white on red",
                          "log.text": "white",
                          "repr.url": "underline blue",
                          "progress.description": "bold white",
                          "progress.percentage": "bold green",
                          "progress.bar": "blue",
            "path": "cyan underline",
        "number": "bold magenta",
        "keyword": "bold bright_yellow", # Style for your keywords
    })

    # CREATE A UNIFIED REGEX HIGHLIGHTER
    class CustomHighlighter(RegexHighlighter):
        """Apply style to numbers, file paths, and keywords."""
        highlights = [
            # Regex for your keywords (using word boundaries \b)
            r"(?P<keyword>\bBioProject\b|\bENA\b|\b16S\b)",
            # Regex for numbers (integers and floats)
            r"(?P<number>\b\d+\.?\d*\b)",
            # A simple regex for file paths
            r"(?P<path>([a-zA-Z]:)?(/|\\)([\w\.\-\_]+/)+[\w\.\-\_]+)"
        ]

    highlighter = CustomHighlighter()
    console = Console(theme=custom_theme)
    rich_handler = RichHandler(console=console, rich_tracebacks=True,
                               level=console_level, show_time=True,
                               show_level=True, show_path=False,
                               markup=True, highlighter=highlighter)
    
    logger.addHandler(rich_handler)

    if not quiet_init:
        logger.info("Logging initialised → %s", log_file_path)
        
    return log_file_path  # Return the path for the parent process


# --- NEW/MODIFIED FUNCTIONS ---

def initialize_logging(log_dir: Path) -> None:
    """
    Initializes logging for the main process.
    This MUST be called at the start of the main script.
    """
    # Check if already initialized (by a different call or process)
    if logging.getLogger("workflow_16s").hasHandlers():
        return

    # Set up the logger and get the timestamped file path
    try:
        log_file_path = setup_logging(
            log_dir_path=log_dir,
            log_filename=None # Creates the timestamp
        )
        
        # **CRITICAL:** Set the environment variable for all future children
        os.environ["WORKFLOW_LOG_FILE"] = str(log_file_path)
    
    except Exception as e:
        print(f"FATAL: Failed to initialize logger in {log_dir}: {e}", file=sys.stderr)
        # We can't use the logger, so just print and exit
        sys.exit(1)


def get_logger(name="workflow_16s") -> logging.Logger:
    """
    Retrieves the logger.
    If in a child process, it will auto-configure from the env var.
    """
    logger = logging.getLogger(name)
    
    # If handlers are *already* attached, we're good.
    if logger.hasHandlers():
        return logger

    # --- CHILD PROCESS PATH ---
    # Check if the main process has set the log file path
    log_file_from_env = os.environ.get("WORKFLOW_LOG_FILE")
    
    if log_file_from_env:
        # We are in a child process. Configure using the exact path.
        setup_logging(
            log_dir_path=Path(log_file_from_env).parent,
            log_filename=Path(log_file_from_env).name
        )
    else:
        # --- MAIN PROCESS NOT INITIALIZED ---
        # This is a fallback.
        # This will log to the CWD (e.g., .../src/logs/)
        print(f"Warning: get_logger() called before initialize_logging(). Defaulting to CWD.", file=sys.stderr)
        initialize_logging(Path.cwd() / "logs")

    return logger