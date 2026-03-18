"""
Загрузка датасета M4 и базовые операции с данными
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

def _load_m4_via_datasetsforecast(frequency: str) -> pd.DataFrame:
    """Загружает M4."""
    from datasetsforecast.m4 import M4

    freq_map = {
        "monthly":   "Monthly",
        "quarterly": "Quarterly",
        "yearly":    "Yearly",
    }
    group = freq_map.get(frequency.lower(), "Monthly")
    train_df, *_ = M4.load(directory="data/", group=group)
    # Колонки: unique_id, ds, y
    return train_df

def load_m4_data(frequency: str = "monthly") -> pd.DataFrame:
    """
    Загружает датасет M4.
    """
    try: #падает из-за библиотеки
        df = _load_m4_via_datasetsforecast(frequency)
        print(f"[data] Загружено M4 {frequency}: {df['unique_id'].nunique()} рядов")
    except Exception as e:
        print(f"[data] Не удалось загрузить")
    return df


def sample_series(data: pd.DataFrame,
                  n_samples: int = 150,
                  min_length: int = 60, 
                  random_seed: int = 42) -> pd.DataFrame:
    """
    Сэмплирование длинных временных рядов (с порогом по длине)
    """
    lengths = data.groupby("unique_id").size()
    
    long_series_ids = lengths[lengths >= min_length].index.unique()

    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(long_series_ids, size=n_samples, replace=False)
    
    return data[data["unique_id"].isin(chosen)].copy()

def train_test_split(data: pd.DataFrame,
                     horizon: int = 18) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает каждый ряд на обучение и тест
    """
    train_rows, test_rows = [], []

    for uid, grp in data.groupby("unique_id", sort=False):
        grp = grp.sort_values("ds")
        train_rows.append(grp.iloc[:-horizon])
        test_rows.append(grp.iloc[-horizon:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True) if test_rows else pd.DataFrame()
    return train_df, test_df


def series_to_list(data: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
    """
    Конвертирует long-format DataFrame в список numpy-массивов
    """
    series_list, uid_list = [], []
    for uid, grp in data.groupby("unique_id", sort=False):
        series_list.append(grp.sort_values("ds")["y"].values.astype(float))
        uid_list.append(uid)
    return series_list, uid_list