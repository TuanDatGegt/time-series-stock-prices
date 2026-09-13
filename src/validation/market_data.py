## src/validation/market_data.py

from __future__ import annotations
from typing import List, Tuple
import pandas as pd

REQUIRED_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


class MarketDataValidator:
    """Validate OHLCV market data before persistence to storage."""

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            raise ValueError("Market data cannot be empty.")

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        normalized = df.copy()
        normalized["timestamp"] = pd.to_datetime(
            normalized["timestamp"], errors="coerce"
        )
        normalized = normalized.sort_values(["symbol", "timestamp"]).reset_index(
            drop=True
        )

        normalized["issues"] = ""

        duplicate_keys = normalized.duplicated(
            subset=["symbol", "timestamp"], keep="first"
        )

        for idx, row in normalized.iterrows():
            issues: List[str] = []

            symbol = row.get("symbol")
            timestamp = row.get("timestamp")
            open_price = row.get("open")
            high_price = row.get("high")
            low_price = row.get("low")
            close_price = row.get("close")
            volume = row.get("volume")

            if pd.isna(symbol) or str(symbol).strip() == "":
                issues.append("Missing symbol")

            if pd.isna(timestamp):
                issues.append("Missing timestamp")

            for field_name, value in {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }.items():
                if pd.isna(value):
                    issues.append(f"Missing {field_name}")

            if pd.isna(volume) or volume < 0:
                issues.append("Volume must be non-negative")

            if (
                not pd.isna(open_price)
                and not pd.isna(high_price)
                and not pd.isna(low_price)
                and not pd.isna(close_price)
            ):
                if not (
                    high_price >= open_price
                    and high_price >= close_price
                    and high_price >= low_price
                    and low_price <= open_price
                    and low_price <= close_price
                    and low_price <= high_price
                ):
                    issues.append("OHLC consistency")

            if duplicate_keys.iloc[idx]:
                issues.append("Duplicate symbol/timestamp")

            if pd.notna(timestamp) and timestamp.tzinfo is None:
                # keep naive timestamps as valid for local tests, but allow a warning externally
                pass

            normalized.at[idx, "issues"] = "; ".join(issues)

        invalid_mask = normalized["issues"].str.len() > 0
        valid_df = normalized.loc[~invalid_mask, REQUIRED_COLUMNS].copy()
        invalid_df = normalized.loc[invalid_mask, REQUIRED_COLUMNS + ["issues"]].copy()

        warnings = self.detect_gaps(normalized)
        if warnings:
            invalid_df["warnings"] = "; ".join(warnings)

        return valid_df.reset_index(drop=True), invalid_df.reset_index(drop=True)

    def detect_gaps(self, df: pd.DataFrame) -> List[str]:
        warnings: List[str] = []

        if "symbol" not in df.columns or "timestamp" not in df.columns:
            return warnings

        for symbol, group in df.groupby("symbol", sort=True):
            ordered = group.sort_values("timestamp").dropna(subset=["timestamp"])
            if ordered.shape[0] < 2:
                continue

            deltas = ordered["timestamp"].diff().dropna()
            if deltas.empty:
                continue

            median_delta = deltas.median()
            if pd.isna(median_delta):
                continue

            suspicious = deltas[deltas > median_delta * 3]
            if not suspicious.empty:
                warnings.append(
                    f"Gap detected for {symbol}: {len(suspicious)} abnormal timestamp jumps"
                )

        return warnings
