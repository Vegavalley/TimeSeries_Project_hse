"""
Модели прогнозирования:
  - Статистические бейзлайны: Naive, SeasonalNaive, Mean, Median
  - Авто-модели: AutoTheta, AutoETS
  - CatBoost с нормализацией рядов, eval-set
"""
 
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from catboost import CatBoostRegressor
 
class BaselineForecaster:
    VALID_METHODS = ("naive", "seasonal_naive", "mean", "median")
 
    def __init__(self, method: str = "naive", seasonal_period: int = 12):
        self.method          = method
        self.seasonal_period = seasonal_period
        self._train: Optional[np.ndarray] = None
 
    def fit(self, y_train: np.ndarray) -> "BaselineForecaster":
        self._train = np.asarray(y_train, dtype=float)
        return self
 
    def predict(self, horizon: int) -> np.ndarray:
        y = self._train
        if self.method == "naive":
            return np.full(horizon, y[-1])
        if self.method == "seasonal_naive":
            p     = self.seasonal_period
            cycle = y[-p:] if len(y) >= p else y
            return np.tile(cycle, int(np.ceil(horizon / len(cycle))))[:horizon]
        if self.method == "mean":
            return np.full(horizon, np.mean(y))
        if self.method == "median":
            return np.full(horizon, np.median(y))
 

class StatsForecastBaseline:
    VALID_MODELS = ("AutoTheta", "AutoETS")
 
    def __init__(self, model_name: str = "AutoTheta", seasonal_period: int = 12):
        self.model_name      = model_name
        self.seasonal_period = seasonal_period
        self._model          = None
        self._fitted         = False
 
    def fit(self, y_train: np.ndarray) -> "StatsForecastBaseline":
        from statsforecast.models import AutoTheta, AutoETS
        y = np.asarray(y_train, dtype=float)
        self._model = (AutoTheta if self.model_name == "AutoTheta" else AutoETS)(
            season_length=self.seasonal_period
        ).fit(y)
        self._fitted = True
        return self
 
    def predict(self, horizon: int) -> np.ndarray:
        result = self._model.predict(h=horizon)
        if isinstance(result, dict):
            return np.asarray(list(result.values())[0], dtype=float)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0].values.astype(float)
        return np.asarray(result, dtype=float)
 
 
def train_statsforecast_models(
    series_list:     List[np.ndarray],
    model_names:     List[str],
    seasonal_period: int = 12,
) -> Dict[str, List[StatsForecastBaseline]]:
    result: Dict[str, List[StatsForecastBaseline]] = {}
    for name in model_names:
        fitted, ok = [], True
        for y in series_list:
            fitted.append(
                    StatsForecastBaseline(name, seasonal_period).fit(y))

        if ok:
            result[name] = fitted
    return result
 

class SeriesScaler:
    """Нормализует каждый ряд на (mean, std) для глобальных моделей"""
 
    def __init__(self):
        self.means_: List[float] = []
        self.stds_:  List[float] = []
 
    def fit_transform(self, series_list: List[np.ndarray]) -> List[np.ndarray]:
        self.means_, self.stds_ = [], []
        result = []
        for y in series_list:
            m = float(np.mean(y))
            s = float(np.std(y)) + 1e-8
            self.means_.append(m); self.stds_.append(s)
            result.append((y - m) / s)
        return result
 

CATBOOST_CAT_FEATURES = ["meta_cluster", "month_in_period"]
 
 
class CatBoostForecaster:
    """
    CatBoost с:
      - meta_cluster, month_in_period
      - eval-set для early stopping
      - SeriesScaler
    """
 
    def __init__(self, config: Optional[dict] = None):
        params = {
            "iterations":            2000,
            "learning_rate":         0.01,
            "depth":                 6,
            "l2_leaf_reg":           5.0,
            "loss_function":         "RMSE",
            "eval_metric":           "RMSE",
            "random_seed":           42,
            "early_stopping_rounds": 100,
            "verbose":               False,
        }
        if config:
            config.pop("verbose", None)
            params.update(config)
        self.model = CatBoostRegressor(**params)
        self._feature_names: List[str] = []
 
    def _resolve_cat_features(self, X: pd.DataFrame) -> List[str]:
        return [c for c in CATBOOST_CAT_FEATURES if c in X.columns]
 
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray]   = None) -> "CatBoostForecaster":
        self._feature_names = list(X_train.columns)
        cat_features        = self._resolve_cat_features(X_train)
 
        for col in cat_features:
            X_train = X_train.copy()
            X_train[col] = X_train[col].astype(int)
            if X_val is not None:
                X_val = X_val.copy()
                X_val[col] = X_val[col].astype(int)
 
        fit_kwargs: dict = {"cat_features": cat_features}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
 
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self
 
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        cat_features = self._resolve_cat_features(X_test)
        X_test = X_test.copy()
        for col in cat_features:
            X_test[col] = X_test[col].astype(int)
        return self.model.predict(X_test)
 
    def get_feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.get_feature_importance(),
            index=self._feature_names
        ).sort_values(ascending=False)
 
 
def train_catboost(X_train: pd.DataFrame, y_train: np.ndarray,
                   config: Optional[dict] = None,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[np.ndarray]   = None) -> CatBoostForecaster:
    return CatBoostForecaster(config=config).fit(X_train, y_train, X_val, y_val)
 

def train_baseline_models(
    series_list: List[np.ndarray], methods: List[str],
    seasonal_period: int = 12,
) -> Dict[str, List[BaselineForecaster]]:
    result: Dict[str, List[BaselineForecaster]] = {m: [] for m in methods}
    for y in series_list:
        for m in methods:
            result[m].append(
                BaselineForecaster(method=m, seasonal_period=seasonal_period).fit(y)
            )
    return result
 
 
def predict_baselines(baseline_models: Dict[str, List],
                      horizon: int) -> Dict[str, np.ndarray]:
    return {
        method: np.stack([m.predict(horizon) for m in models])
        for method, models in baseline_models.items()
    }
 