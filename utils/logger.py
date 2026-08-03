from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    console = Console()
    handler = RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)
    handler.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
