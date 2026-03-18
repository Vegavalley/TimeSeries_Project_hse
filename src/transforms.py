"""
Трансформации временных рядов и их обратные функции
none, log1p, boxcox, diff
"""

import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox


def apply_transform(y: np.ndarray, kind: str):
    """
    Применяет трансформацию к ряду
    """
    y = np.asarray(y, dtype=float)

    if kind == "none":
        return y.copy(), None

    elif kind == "log1p":
        return np.log1p(y), None

    elif kind == "boxcox":
        y_tr, lam = boxcox(y + 1e-6)
        return y_tr, lam

    else:
        return np.diff(y), y[0]


def inverse_transform(y_pred: np.ndarray,
                      kind: str,
                      aux=None,
                      last_train_val=None) -> np.ndarray:
    """
    Обратная трансформация предсказаний в исходную шкалу
    """
    y_pred = np.asarray(y_pred, dtype=float)

    if kind == "none":
        return y_pred

    elif kind == "log1p":
        # np.expm1 при больших значениях даёт inf
        y_clipped = np.clip(y_pred, -500, 500)
        result    = np.expm1(y_clipped)
        return np.where(np.isfinite(result), result, 0.0)

    elif kind == "boxcox":
        lam = aux if aux is not None else 1.0
        if lam != 0:
            max_safe = (1e15 * abs(lam)) ** abs(lam) - 1.0 / abs(lam) if lam > 0 else 700
        else:
            max_safe = 700
        y_safe = np.clip(y_pred, -max_safe, max_safe)
        result  = inv_boxcox(y_safe, lam) - 1e-6
        return np.where(np.isfinite(result), result, 0.0)

    else:
        return np.cumsum(y_pred) + last_train_val
