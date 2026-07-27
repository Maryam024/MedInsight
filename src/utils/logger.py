"""
Centralized logging configuration for MedInsight.

Why this file exists
---------------------
In research code it is tempting to sprinkle `print()` statements everywhere.
This breaks down almost immediately once you have:
  - multiple modules (data loading, model training, retrieval, evaluation)
  - long-running experiments you need to audit after the fact
  - a need to separate "what happened" (INFO) from "something is wrong"
    (WARNING/ERROR) so you can grep logs for problems quickly.

`get_logger()` gives every module in the project a consistently formatted,
named logger that writes to both the console (for live feedback) and a
rotating log file (for permanent experiment records), configured in exactly
one place: configs/config.yaml.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Module-level cache so repeated calls to get_logger() with the same name
# don't attach duplicate handlers (which would print every message twice).
_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "experiments/logs",
    log_filename: str = "medinsight.log",
) -> logging.Logger:
    """
    Return a configured logger instance for the given module name.

    Parameters
    ----------
    name : str
        Name of the logger, conventionally `__name__` of the calling module.
        This shows up in every log line, so you can trace which part of the
        pipeline produced a given message.
    log_level : str
        One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        Read from configs/config.yaml -> logging.level in normal use.
    log_to_file : bool
        If True, also write logs to a rotating file on disk in addition to
        the console. Controlled by configs/config.yaml -> logging.log_to_file.
    log_dir : str
        Directory where the log file will be created if log_to_file is True.
    log_filename : str
        Name of the log file within log_dir.

    Returns
    -------
    logging.Logger
        A logger with console (and optionally file) handlers attached.

    Example
    -------
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting data preprocessing...")
    """
    logger = logging.getLogger(name)

    # Guard against attaching duplicate handlers if get_logger() is called
    # more than once for the same module (e.g., re-imports in notebooks).
    if name in _CONFIGURED_LOGGERS:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False  # avoid double-logging via the root logger

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler: what you see live while a script runs ---
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # --- File handler: permanent, rotating record for experiment audits ---
    if log_to_file:
        log_directory = Path(log_dir)
        log_directory.mkdir(parents=True, exist_ok=True)
        file_path = log_directory / log_filename

        # RotatingFileHandler caps file size so long-running experiments
        # don't produce unbounded log files. 5 MB per file, 5 backups kept.
        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
