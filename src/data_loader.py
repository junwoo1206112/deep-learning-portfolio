import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generator import generate_sensor_data, SENSOR_COLUMNS


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        pass


class SyntheticDataLoader(BaseDataLoader):
    def __init__(self, config: dict):
        self.config = config

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        data, labels = generate_sensor_data(
            n_samples=self.config['data']['n_samples'],
            n_features=self.config['data']['n_features'],
            anomaly_ratio=self.config['data'].get('anomaly_ratio', 0.05),
            random_seed=self.config['data']['random_seed'],
            return_labels=True
        )
        return data, labels


class CSVDataLoader(BaseDataLoader):
    def __init__(self, config: dict):
        self.path = config['data']['source']['path']
        self.label_column = config['data']['source'].get('label_column')

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.path)

        if self.label_column and self.label_column in df.columns:
            labels = df[self.label_column].astype(int).values
            feature_cols = [c for c in df.columns if c != self.label_column]
            data = df[feature_cols].values
        else:
            labels = np.zeros(len(df), dtype=int)
            data = df.values

        return data, labels


class ParquetDataLoader(BaseDataLoader):
    def __init__(self, config: dict):
        self.path = config['data']['source']['path']
        self.label_column = config['data']['source'].get('label_column')

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_parquet(self.path)

        if self.label_column and self.label_column in df.columns:
            labels = df[self.label_column].astype(int).values
            feature_cols = [c for c in df.columns if c != self.label_column]
            data = df[feature_cols].values
        else:
            labels = np.zeros(len(df), dtype=int)
            data = df.values

        return data, labels


class SQLiteDataLoader(BaseDataLoader):
    def __init__(self, config: dict):
        self.path = config['data']['source']['path']
        self.table = config['data']['source']['table']
        self.label_column = config['data']['source'].get('label_column')

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        conn = sqlite3.connect(self.path)
        df = pd.read_sql_query(f"SELECT * FROM {self.table}", conn)
        conn.close()

        if self.label_column and self.label_column in df.columns:
            labels = df[self.label_column].astype(int).values
            feature_cols = [c for c in df.columns if c != self.label_column]
            data = df[feature_cols].values
        else:
            labels = np.zeros(len(df), dtype=int)
            data = df.values

        return data, labels


def get_data_loader(config: dict) -> BaseDataLoader:
    source = config['data'].get('source', {})
    source_type = source.get('type', 'synthetic').lower()

    if source_type == 'csv':
        return CSVDataLoader(config)
    if source_type == 'parquet':
        return ParquetDataLoader(config)
    if source_type == 'sqlite':
        return SQLiteDataLoader(config)

    return SyntheticDataLoader(config)
