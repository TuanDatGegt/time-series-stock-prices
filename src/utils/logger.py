## src/utils/logger.py
from __future__ import annotations

import logging
import sys


def get_logger(
    name: str = "time_series_forecasting", level: str = "INFO"
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
