"""
Logging configuration for TMT GraphRAG API.

Reads LOG_LEVEL env var (default: INFO) and configures the root logger with
a StreamHandler to stdout. The setup_logging() function is idempotent — safe
to call multiple times.
"""
import logging
import os
import sys

_LOGGING_CONFIGURED = False

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging() -> None:
    """Configure the root logger once. Subsequent calls are no-ops."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Avoid duplicate handlers if something already added one.
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    _LOGGING_CONFIGURED = True
