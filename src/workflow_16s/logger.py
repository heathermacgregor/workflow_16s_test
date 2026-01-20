# ===================================== IMPORTS ====================================== #

# Standard Library
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

# 3rd‑party (Rich)
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ==================================== FUNCTIONS ===================================== #

def setup_logging(
    log_dir_path: Union[str, Path],
    log_filename: Union[str, None] = None,
    max_file_size: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
    console_level: int = logging.INFO,     # console shows INFO+
    file_level: int = logging.DEBUG        # file keeps DEBUG+
) -> logging.Logger:
    """
    Configure unified logging with:
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
    # Disable propagation to avoid duplicate logs from parent/root
    logger.propagate = False  # 🚀 Key fix

    # Only set up handlers if none exist (prevents repeated setup in parallel workers)
    if not logger.handlers:
        # Remove existing handlers to avoid duplicates (should be empty, but safe)
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
        file_fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s:%(filename)s:%(funcName)s(): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        # ────────────────────── CONSOLE HANDLER ────────────────────
        # Custom theme for Rich console
        custom_theme = Theme({
            "logging.time": "bold white",
            "logging.level.info": "bold white",
            "logging.level.debug": "dim cyan",
            "logging.level.warning": "bold yellow",
            "logging.level.error": "bold red",
            "logging.level.critical": "reverse bold bright_white on red",
            "progress.description": "bold white",
            "progress.percentage": "bold green",
            "progress.bar": "blue",
        })
        
        console = Console(theme=custom_theme)
        
        # Configure RichHandler for consistent console logging
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            level=console_level,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
            log_time_format="[%X]",
        )
        
        # Set formatter to match our desired format
        rich_handler.setFormatter(logging.Formatter(
            "%(message)s",
            datefmt="%H:%M:%S"
        ))
        
        logger.addHandler(rich_handler)

        # Only log initialization in the main process (not in child workers)
        import os
        # Main process usually has ppid != its own pid and != 1
        if os.getppid() != os.getpid() and os.getppid() != 1:
            logger.info("Logging initialised → %s", log_file_path)
    return logger

def get_logger(name="workflow_16s", log_dir: Union[str, Path, None] = None):
    """
    Retrieves the logger instance, initializing it if it hasn't been set up.
    This ensures that logging is automatically configured on its first use.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # If handlers are not configured, set them up automatically.
        # This makes the setup implicit and robust.
        if log_dir is None:
            # Default to a 'logs' directory in the current working directory
            # if no path is provided.
            log_dir = Path.cwd() / "logs"
            
        setup_logging(log_dir_path=log_dir)
        
    return logger