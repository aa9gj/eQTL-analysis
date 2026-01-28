"""Logging utilities for the eQTL analysis pipeline."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

# Custom log levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Package logger name
LOGGER_NAME = "eqtl_analysis"

# Format strings
CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        "TRACE": "\033[37m",  # White
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str | None = None, use_colors: bool = True) -> None:
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional colors."""
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    log_dir: str | Path | None = None,
    use_colors: bool = True,
    quiet: bool = False,
) -> Logger:
    """
    Set up logging for the eQTL analysis pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Specific log file path. If None, generates timestamped name.
        log_dir: Directory for log files. If None, no file logging.
        use_colors: Whether to use colored console output.
        quiet: If True, suppress console output.

    Returns:
        Configured logger instance.
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get the package logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(CONSOLE_FORMAT, DATE_FORMAT, use_colors)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir is not None or log_file is not None:
        if log_file is not None:
            file_path = Path(log_file)
        else:
            log_dir = Path(log_dir) if log_dir else Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = log_dir / f"eqtl_analysis_{timestamp}.log"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(FILE_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {file_path}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name. If None, returns the package logger.

    Returns:
        Logger instance.
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


class LogContext:
    """Context manager for temporarily changing log level."""

    def __init__(self, level: int | str, logger_name: str | None = None) -> None:
        """
        Initialize log context.

        Args:
            level: Temporary logging level.
            logger_name: Logger to modify. If None, uses package logger.
        """
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.logger = get_logger(logger_name)
        self.original_level: int | None = None

    def __enter__(self) -> Logger:
        """Enter context and set new level."""
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Exit context and restore original level."""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


def log_step(step_name: str, logger: Logger | None = None) -> None:
    """
    Log a pipeline step with visual separator.

    Args:
        step_name: Name of the step.
        logger: Logger to use. If None, uses package logger.
    """
    if logger is None:
        logger = get_logger()

    separator = "=" * 60
    logger.info(separator)
    logger.info(f"  {step_name}")
    logger.info(separator)


def log_summary(
    title: str,
    items: dict[str, str | int | float],
    logger: Logger | None = None,
) -> None:
    """
    Log a summary of key-value pairs.

    Args:
        title: Summary title.
        items: Dictionary of items to log.
        logger: Logger to use. If None, uses package logger.
    """
    if logger is None:
        logger = get_logger()

    logger.info(f"\n{title}:")
    logger.info("-" * 40)
    for key, value in items.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 40)
