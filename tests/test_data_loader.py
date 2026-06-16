import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.data_loader import SyntheticDataLoader, CSVDataLoader, ParquetDataLoader, SQLiteDataLoader, get_data_loader


@pytest.fixture
def sample_config():
    return {
        'data': {
            'n_samples': 1000,
            'n_features': 5,
            'anomaly_ratio': 0.05,
            'random_seed': 42,
            'source': {
                'type': 'synthetic'
            }
        }
    }


def test_synthetic_loader(sample_config):
    loader = SyntheticDataLoader(sample_config)
    data, labels = loader.load()
    assert data.shape == (1000, 5)
    assert labels.shape == (1000,)


def test_csv_loader(tmp_path, sample_config):
    df = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    df['anomaly'] = np.random.randint(0, 2, 100)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    sample_config['data']['source'] = {
        'type': 'csv',
        'path': str(path),
        'label_column': 'anomaly'
    }

    loader = CSVDataLoader(sample_config)
    data, labels = loader.load()
    assert data.shape == (100, 5)
    assert labels.shape == (100,)


def test_parquet_loader(tmp_path, sample_config):
    df = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    df['anomaly'] = np.random.randint(0, 2, 100)
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)

    sample_config['data']['source'] = {
        'type': 'parquet',
        'path': str(path),
        'label_column': 'anomaly'
    }

    loader = ParquetDataLoader(sample_config)
    data, labels = loader.load()
    assert data.shape == (100, 5)
    assert labels.shape == (100,)


def test_sqlite_loader(tmp_path, sample_config):
    path = tmp_path / "data.db"
    conn = sqlite3.connect(str(path))
    df = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    df['anomaly'] = np.random.randint(0, 2, 100)
    df.to_sql("sensor", conn, index=False)
    conn.close()

    sample_config['data']['source'] = {
        'type': 'sqlite',
        'path': str(path),
        'table': 'sensor',
        'label_column': 'anomaly'
    }

    loader = SQLiteDataLoader(sample_config)
    data, labels = loader.load()
    assert data.shape == (100, 5)
    assert labels.shape == (100,)


def test_get_data_loader(sample_config):
    sample_config['data']['source'] = {'type': 'synthetic'}
    loader = get_data_loader(sample_config)
    assert isinstance(loader, SyntheticDataLoader)
