"""
Метрики прогнозирования — все вычисляются в исходной шкале.
mae, rmse, mape, smape, mase
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
        )
    )


def mase(y_true: np.ndarray,
         y_pred: np.ndarray,
         y_train: np.ndarray,
         seasonality: int = 1) -> float:
    y_true  = np.asarray(y_true,  dtype=float)
    y_pred  = np.asarray(y_pred,  dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    m = int(seasonality)
    if len(y_train) > m:
        # сезонные разности: y[t] - y[t-m]
        naive_errors = np.abs(y_train[m:] - y_train[:-m])
    else:
        # ряд слишком короткий — падаем до lag-1
        naive_errors = np.abs(np.diff(y_train))

    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / (scale + 1e-8))

def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray,
             y_train: Optional[np.ndarray] = None,
             seasonality: int = 1) -> Dict[str, float]:
    """
    Считает все метрики для одного прогноза
    """
    result = {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
    if y_train is not None:
        result["MASE"] = mase(y_true, y_pred, y_train, seasonality)
    return result


def evaluate_all_models(
    predictions: Dict[str, np.ndarray],
    y_true_list: List[np.ndarray],
    y_train_list: Optional[List[np.ndarray]] = None,
    seasonality: int = 1,
) -> pd.DataFrame:
    """
    Считает средние метрики по всем рядам для каждой модели
    """
    rows = []
    for model_name, preds in predictions.items():
        accum: Dict[str, List[float]] = {}
        for i, (y_true, y_pred_i) in enumerate(zip(y_true_list, preds)):
            y_train = y_train_list[i] if y_train_list else None
            m = evaluate(y_true, y_pred_i, y_train, seasonality)
            for k, v in m.items():
                accum.setdefault(k, []).append(v)

        row = {"model": model_name}
        row.update({k: float(np.mean(v)) for k, v in accum.items()})
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")