# ===================================== IMPORTS ====================================== #

# Standard Library
from datetime import datetime
import logging
import functools
from typing import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

# 3rd‑party (Rich)
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ==================================== FUNCTIONS ===================================== #

import inspect

def with_logger(func_or_class):
    """
    The 'Thar-Proof' Decorator. 
    Only injects 'logger' if the function signature allows for it.
    """
    if inspect.isclass(func_or_class):
        target = func_or_class.__init__
    else:
        target = func_or_class

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        # 1. Prepare the logger instance
        logger_instance = get_logger("workflow_16s")
        
        # 2. Inspect the recipient's signature
        sig = inspect.signature(target)
        
        # 3. Only add 'logger' to kwargs if the function can catch it
        # (Either it has a parameter named 'logger' or it has **kwargs)
        has_logger_param = 'logger' in sig.parameters
        has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if has_logger_param or has_var_kwargs:
            kwargs['logger'] = logger_instance
            
        # 4. Call the function - This will no longer throw a TypeError
        return target(*args, **kwargs)

    if inspect.isclass(func_or_class):
        func_or_class.__init__ = wrapper
        return func_or_class
    
    return wrapper

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

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
            
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

        logger.info("Logging initialised → %s", log_file_path)
    return logger

def get_logger(name="workflow_16s", log_dir: Union[str, Path, None] = None):
    """
    Retrieves the logger instance, initializing it if it hasn't been set up.
    This ensures that logging is automatically configured on its first use.
    """
    return logging.getLogger(name)
    '''
    if not logger.handlers:
        # If handlers are not configured, set them up automatically.
        # This makes the setup implicit and robust.
        if log_dir is None:
            # Default to a 'logs' directory in the current working directory
            # if no path is provided.
            log_dir = Path.cwd() / "logs"
            
        setup_logging(log_dir_path=log_dir)
        
    return logger
    '''