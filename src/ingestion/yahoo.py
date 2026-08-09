#src/ingestion/yahoo.py

"""Yahoo Finance Ingestion Client.
Responsibility is strictly: fetch raw market data and normalize it into the project's canonical schema;
It does NOT know about the storage - that is the job of "src/storage". This keeps the provider 
swappable: a structure 'PolygonClient' or 'AlphaVantageClient' only needs to implement the same 'download()' contract.
"""

from __future__ import annotations
from typing import Optional

import pandas as pd
import yfinance as yf
