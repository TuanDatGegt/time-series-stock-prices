## src/ingestion/scheduler.py
"""
Module: src/ingestion/scheduler.py
Description: Automated Job Scheduler for Market Data Ingestion and Pipeline Refresh.
How it works:
    Uses APScheduler (BackgroundScheduler) to periodically trigger incremental market data syncing
    for monitored stock ticker symbols at configured time intervals (e.g., every 15 minutes).
    Executes `IncrementalIngestionService.sync_symbol()` for each target ticker and handles signal traps
    (SIGINT, SIGTERM) for graceful shutdown.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.ingestion.service import IncrementalIngestionService
from src.utils.logger import get_logger

logger = get_logger("ingestion_scheduler")


class IngestionScheduler:
    """
    Periodic job execution manager for automated market data ingestion.
    """

    def __init__(
        self,
        ingestion_service: IncrementalIngestionService,
        symbols: List[str],
        interval_minutes: int = 15,
    ):
        """
        Initialize the scheduler with ingestion service, target symbols, and interval frequency.

        Args:
            ingestion_service (IncrementalIngestionService): Service instance for data synchronization.
            symbols (List[str]): List of stock ticker symbols to monitor (e.g., ['INTC', 'AAPL']).
            interval_minutes (int): Ingestion interval frequency in minutes (default: 15).
        """
        self.ingestion_service = ingestion_service
        self.symbols = [s.upper().strip() for s in symbols if s.strip()]
        self.interval_minutes = interval_minutes
        self.scheduler = BackgroundScheduler()
        self._is_running = False

    def _sync_job(self) -> None:
        """
        Scheduled job callback that iterates through target symbols and syncs fresh data.
        """
        logger.info(
            "Starting scheduled incremental market data sync for symbols: %s",
            self.symbols,
        )
        for symbol in self.symbols:
            try:
                if hasattr(self.ingestion_service, "run_symbol"):
                    metrics = self.ingestion_service.run_symbol(symbol=symbol)
                else:
                    metrics = self.ingestion_service.sync_symbol(symbol=symbol)
                logger.info(
                    "Sync complete [%s]: fetched=%d, valid=%d, upserted=%d, last_ts=%s",
                    symbol,
                    metrics.get("fetched_rows", 0),
                    metrics.get("valid_rows", 0),
                    metrics.get("upserted_rows", 0),
                    metrics.get("last_timestamp"),
                )
            except Exception as exc:
                logger.error(
                    "Error during scheduled sync for symbol %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                )

    def start(self, run_immediately: bool = True) -> None:
        """
        Configure scheduled job triggers, start APScheduler background process,
        and attach signal handlers for graceful shutdown.

        Args:
            run_immediately (bool): If True, triggers an immediate sync job execution on startup.
        """
        if self._is_running:
            logger.warning("IngestionScheduler is already running.")
            return

        logger.info(
            "Initializing IngestionScheduler (Interval: %d minutes, Symbols: %s)...",
            self.interval_minutes,
            self.symbols,
        )

        # Register interval job
        self.scheduler.add_job(
            func=self._sync_job,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="market_data_sync_job",
            name="Market Data Incremental Sync",
            replace_existing=True,
            max_instances=1,
        )

        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.scheduler.start()
        self._is_running = True
        logger.info("IngestionScheduler background process started successfully.")

        # Run immediate sync on startup
        if run_immediately:
            self._sync_job()

    def _handle_shutdown(self, signum: int, frame: Optional[object]) -> None:
        """
        Signal handler callback for graceful scheduler shutdown.
        """
        logger.info(
            "Received termination signal (%d). Shutting down IngestionScheduler...",
            signum,
        )
        self.stop()
        sys.exit(0)

    def stop(self) -> None:
        """
        Stop the background job scheduler cleanly.
        """
        if self._is_running and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("IngestionScheduler stopped cleanly.")
