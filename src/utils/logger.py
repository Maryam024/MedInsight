from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

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
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger  # avoid duplicate handlers on repeat calls

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False  # stop root logger from double-printing

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    if log_to_file:
        log_directory = Path(log_dir)
        log_directory.mkdir(parents=True, exist_ok=True)
        file_path = log_directory / log_filename

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
