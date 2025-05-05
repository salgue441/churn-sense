# churnsense/utils/logging.py
"""Logging utilities for ChurnSense."""

import logging
import sys
from pathlib import Path
from typing import Optional

import colorlog


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file output.

    Args:
        name: Logger name (typically __name__).
        log_level: Logging level.
        log_file: Path to log file. If None, file logging is disabled.
        console: Whether to enable console logging.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:  
        return logger

    logger.setLevel(log_level)
    logger.propagate = False  

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if console:
        console_handler = colorlog.StreamHandler(stream=sys.stdout)
        colored_formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + log_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
