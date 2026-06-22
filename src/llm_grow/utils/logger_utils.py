"""Logging utilities for distributed training and debugging.

This module provides enhanced logging functionality with color coding,
rank information display, and distributed training support.

Key features:
- Color-coded log messages based on severity level
- Automatic rank information display for distributed setups
- Configurable log levels and output formats
- Integration with both single-process and distributed environments
- Prevention of duplicate log handler initialization

Components:
- ColorfulFormatter: Adds ANSI color codes and rank info to log messages
- get_logger(): Factory function for properly configured loggers
- setup_logging(): Global logging configuration setup

Example usage:
    >>> from llm_grow.utils.logger_utils import get_logger, setup_logging
    >>>
    >>> # Setup global logging configuration
    >>> setup_logging(level=logging.INFO)
    >>>
    >>> # Get a logger for your module
    >>> logger = get_logger(__name__)
    >>> logger.info("This message will be color-coded with rank info")

The logger automatically displays format like:
[Rank 0] - module_name - LEVEL: message
"""

import logging
import os
import sys
import typing
from logging import Formatter, LogRecord
from pathlib import Path

import torch.distributed as dist
from colorama import Fore, Style

# Track which loggers have already been initialized to avoid duplicate handlers
logger_initialized: dict[str, bool] = {}


class ColorfulFormatter(Formatter):
    """Formatter that adds ANSI color codes to log messages based on their
    level and includes rank information for distributed training.

    Attributes:
        COLORS: Dictionary mapping log levels to their corresponding color codes

    Example:
        >>> formatter = ColorfulFormatter('%(levelname)s: %(message)s')
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)
    """

    COLORS: typing.ClassVar[dict[str, str]] = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.LIGHTGREEN_EX,
    }

    def format(self, record: LogRecord) -> str:
        """Format a log record with color codes and rank information.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message with color codes.
        """
        # Add rank information to the record (logging.LogRecord accepts dynamic attrs)
        record.rank = self._get_rank()
        record.is_main = record.rank == 0  # type: ignore[attr-defined]

        # Format the log message
        log_message = super().format(record)

        # Add color based on log level
        prefix = str(self.COLORS.get(record.levelname, ""))
        return prefix + log_message + str(Fore.RESET)

    def _get_rank(self) -> int:
        """Get the current process rank in a safe way.

        Attempts to retrieve rank from distributed training environment,
        falls back to environment variables, defaults to 0.

        Returns:
            int: The rank of the current process (0 for main process)
        """
        return _get_distributed_rank()


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    log_level: int = logging.INFO,
    file_mode: str = "w",
    force_main_process: bool = False,
) -> logging.Logger:
    """Initialize and get a logger by name with optional file output.

    This function creates or retrieves a logger with the specified configuration.
    It handles distributed training scenarios by managing log levels across different
    process ranks and prevents duplicate logging issues with PyTorch DDP.

    Args:
        name: Logger name for identification and hierarchy
        log_file: Path to the log file. If provided, logs
                 will also be written to this file
                 (only for rank 0 in distributed training)
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
                  Note: Only rank 0 process uses this level; others use ERROR level
        file_mode: File opening mode ('w' for write, 'a' for append)
        force_main_process: If True, only main process
                 (rank 0) will log regardless of log_level

    Returns:
        A configured logging.Logger instance

    Example:
        >>> logger = get_logger("my_model", "training.log", logging.DEBUG)
        >>> logger.info("Training started")
    """
    if file_mode not in ("w", "a"):
        raise ValueError("file_mode must be either 'w' or 'a'")

    # Get or create logger instance
    logger = logging.getLogger(name)

    # Return existing logger if already initialized with this exact name.
    # We intentionally do NOT match by prefix — a child logger (e.g.
    # llm_grow.safetensor.writer) must receive its own configured instance,
    # not the parent's (llm_grow.safetensor) logger, so that per-module
    # log-level and handler defaults are honoured.
    if name in logger_initialized:
        return logger

    # Get current rank safely
    rank = _get_distributed_rank()
    is_main_process = rank == 0

    # Fix PyTorch DDP duplicate logging issue
    # Clear existing handlers to prevent duplicate logging
    if logger.handlers:
        logger.handlers.clear()

    # Build handlers for this process.  By default only the main process emits
    # to stdout/file; non-main processes are throttled to ERROR (or disabled
    # entirely when force_main_process=True).
    handlers: list[logging.Handler] = []
    if is_main_process:
        handlers.append(logging.StreamHandler(sys.stdout))
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(log_file), file_mode))

    fmt = (
        "%(asctime)s - [Rank %(rank)d] - "
        "%(name)s.%(funcName)s:%(lineno)d - "
        "%(levelname)s - %(message)s"
        if is_main_process
        else "%(asctime)s - [Rank %(rank)d] - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = ColorfulFormatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # Set logger level based on rank and configuration
    if force_main_process and not is_main_process:
        logger.setLevel(logging.CRITICAL + 1)  # Disable logging for non-main processes
    else:
        logger.setLevel(log_level if is_main_process else logging.ERROR)

    # Prevent messages from being handled by both this logger and parent loggers
    logger.propagate = False

    # Mark logger as initialized
    logger_initialized[name] = True

    return logger


def _get_distributed_rank() -> int:
    """Safely get the current distributed rank.

    Returns:
        int: The current process rank (0 for main process)
    """
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except (RuntimeError, ValueError, OSError):
        # torch.distributed may not be initialized or available.
        # Do not log here: this function is called by the log formatter itself,
        # so emitting a log record could cause infinite recursion.
        pass

    # Fallback to environment variables (common distributed training variables)
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)

    return 0  # Default to main process
