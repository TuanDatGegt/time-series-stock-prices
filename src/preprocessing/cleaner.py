## src/preprocessing/cleaner.py

from __future__ import annotations
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.clean_data()
        self.drop_duplicates()
        self.fill_missing_values()
        self.convert_to_datetime()
        self.drop_unnecessary_columns()
        self.normalize_data()
        self.standardize_data()

    def clean_data(self):
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data.fillna(method="ffill", inplace=True)

    def drop_duplicates(self):
        self.data.drop_duplicates(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data.fillna(method="ffill", inplace=True)
        self.data.dropna(inplace=True)

    def fill_missing_values(self):
        self.data.fillna(method="ffill", inplace=True)

    def convert_to_datetime(self):
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data.set_index("date", inplace=True)
        self.data.drop(columns=["date"], inplace=True)
        self.data.sort_index(inplace=True)
        self.data.reset_index(inplace=True)
        self.data.fillna(method="ffill", inplace=True)
        self.data.dropna(inplace=True)
