"""
Генерация признаков: лаги, скользящие окна, тренд, сезонность
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def create_lag_features(data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Добавляет колонки lag_<k>
    """
    df = data.copy()
    for k in lags:
        df[f"lag_{k}"] = df.groupby("unique_id")["y"].shift(k)
    return df


def create_rolling_features(data: pd.DataFrame,
                             windows: List[int],
                             agg_funcs: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Добавляет скользящие агрегации 
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max"]

    df = data.copy()
    for w in windows:
        shifted = df.groupby("unique_id")["y"].shift(1)
        for func in agg_funcs:
            col = f"roll_{w}_{func}"
            df[col] = (
                shifted
                .groupby(df["unique_id"])
                .transform(lambda x: x.rolling(w, min_periods=1).agg(func))
            )
    return df


def add_trend_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет порядковый номер шага t и его квадрат t² для каждого ряда
    """
    df = data.copy()
    # cumcount даёт 0-based порядковый номер внутри каждой группы
    t = df.groupby("unique_id").cumcount()
    df["t"]  = t
    df["t2"] = t ** 2
    return df


def add_seasonal_features(data: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """
    Добавляет синус/косинус Фурье для основного сезонного периода
    и категориальный признак 'month_in_period'
    """
    df = data.copy()
    t = df.groupby("unique_id").cumcount()
    df["sin_s1"] = np.sin(2 * np.pi * t / period)
    df["cos_s1"] = np.cos(2 * np.pi * t / period)
    df["sin_s2"] = np.sin(4 * np.pi * t / period)
    df["cos_s2"] = np.cos(4 * np.pi * t / period)
    df["month_in_period"] = (t % period).astype(int)
    return df

def engineer_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Применяет все шаги генерации признаков
    """
    df = data.copy()

    # Лаги
    lags = config.get("lags", list(range(1, 13)))
    df = create_lag_features(df, lags)

    # Скользящие окна
    windows   = config.get("rolling_windows", [3, 6, 12])
    agg_funcs = config.get("rolling_agg_funcs", ["mean", "std"])
    df = create_rolling_features(df, windows, agg_funcs)

    # трендовые признаки
    if config.get("add_trend", True):
        df = add_trend_features(df)

    # сезонные признаки
    if config.get("add_seasonal", True):
        period = config.get("seasonal_period", 12)
        df = add_seasonal_features(df, period)

    feature_cols = [c for c in df.columns
                    if c not in ("unique_id", "ds", "y")]
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)

    return df