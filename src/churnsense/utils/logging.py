"""Logging utilities for ChurnSense."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import os
import json

import colorlog

from churnsense.config import config


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file output.

    Args:
        name: Logger name (typically __name__).
        log_level: Logging level.
        log_file: Path to log file. If None, uses default from config.
        console: Whether to enable console logging.
        log_format: Custom log format string.

    Returns:
        Configured logger.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file is None:
        log_file = config.log_path

    if isinstance(log_file, (str, Path)):
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

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
            secondary_log_colors={},
            style="%",
        )

        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class JsonLogger:
    """Logger for structured JSON output."""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        include_timestamp: bool = True,
    ):
        """
        Initialize JSON logger.

        Args:
            name: Logger name.
            log_dir: Directory for JSON log files. If None, uses config.
            include_timestamp: Whether to include timestamp in logs.
        """
        self.name = name
        self.include_timestamp = include_timestamp

        if log_dir is None:
            self.log_dir = Path(config.logs_dir) / "json"

        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"{name}_json")

        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"{name}_{today}.jsonl"

    def log(self, data: Dict[str, Any]) -> None:
        """
        Log data as JSON.

        Args:
            data: Data to log.
        """

        try:
            if self.include_timestamp:
                data["timestamp"] = datetime.now().isoformat()

            data["logger"] = self.name
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")

        except Exception as e:
            self.logger.error(f"Error logging JSON data: {str(e)}")

    def log_event(
        self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a structured event.

        Args:
            event_type: Type of event.
            message: Event message.
            data: Additional event data.
        """

        event_data = {
            "event_type": event_type,
            "message": message,
        }

        if data:
            event_data["data"] = data

        self.log(event_data)

    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an error.

        Args:
            error_type: Type of error.
            message: Error message.
            exception: Exception object.
            details: Additional error details.
        """

        error_data = {
            "error_type": error_type,
            "message": message,
        }

        if exception:
            error_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": self._get_traceback(exception),
            }

        if details:
            error_data["details"] = details

        self.log(error_data)
        self.logger.error(f"{error_type}: {message}")

    def _get_traceback(self, exception: Exception) -> str:
        """
        Get formatted traceback from exception.

        Args:
            exception: Exception object.

        Returns:
            Formatted traceback string.
        """

        import traceback

        return "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
