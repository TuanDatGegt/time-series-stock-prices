# **scripts/download\_data.py**[1][7]

"""
Script: scripts/download_data.py
Description: CLI entrypoint for full historical market data acquisition.
How it works:
    Parses arguments (`--symbol`, `--start`, `--end`), loads project config files (`configs/data.yaml`),
    instantiates the data provider, and executes a full historical download to populate the database/lake storage.
"""

import argparse


def main():
    # TODO: Parse command-line flags (--symbol, --start, --end, --interval)
    # TODO: Load data configurations from configs/data.yaml
    # TODO: Instantiate target provider (e.g., YahooFinanceSource)
    # TODO: Run historical ingestion and save output to raw data lake/database
    pass


if __name__ == "__main__":
    main()
