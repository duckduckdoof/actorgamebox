"""
logging.py

Author: Caleb Scott

---

Configuration for logging.
Best to do this asap, so that we can cleanly organize atari game runs.

"""

# IMPORTS
import logging
import sys

from datetime import datetime, timezone

# FUNCTIONS
def configure_logger(
    logger_name: str,
    logger_dir: str,
    log_format_str: str,
    log_datetime_prefix: str,
    datetime_format: str,
    verbose: bool
):
    # Set up the logger first
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    log_fmt = logging.Formatter(
        log_format_str,
        datefmt=datetime_format
    )

    # Handlers (file + stdout if flag)
    today = datetime.now(tz=timezone.utc).strftime(log_datetime_prefix)
    file_handler = logging.FileHandler(f"{logger_dir}{logger_name}-{today}.log")
    file_handler.setFormatter(log_fmt)
    logger.addHandler(file_handler)

    if verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_fmt)
        logger.addHandler(stdout_handler)

    return logger
