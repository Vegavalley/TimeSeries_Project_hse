"""
Конфигурация эксперимента
"""

import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_CONFIG = {
    "dataset":          "M4",
    "frequency":        "monthly",
    "forecast_horizon": 18,       #как в оригинальном соревновании
    "sample_size":      200,
    "train_test_split": 0.8,
    "random_seed":      RANDOM_SEED,
}

CLUSTERING_CONFIG = {
    "method":       "kmeans",
    "n_clusters":   None,
    "k_min":        3,
    "k_max":        8,
    "random_state": RANDOM_SEED,
}

TRANSFORM_CONFIG = {
    "candidates":    ["none", "log1p", "boxcox", "diff"],
    "log_transform": True,
    "box_cox":       True,
    "differencing":  True,
    "diff_order":    1,
}

FEATURE_CONFIG = {
    "lags":              list(range(1, 19)) + [24, 36],
    "rolling_windows":   [3, 6, 12],
    "rolling_agg_funcs": ["mean", "std", "min", "max"],
    "add_trend":         True,
    "add_seasonal":      True,
    "seasonal_period":   12,
}

MODEL_CONFIG = {
    "baseline_models":      ["naive", "seasonal_naive", "mean", "median"],
    "statsforecast_models": ["AutoTheta", "AutoETS"],
    "test_size":    18,
    "random_state": RANDOM_SEED,

    # CatBoost: Direct стратегия
    "catboost_val_frac": 0.15,
    "catboost": {
        "iterations":            5000,
        "learning_rate":         0.02,
        "depth":                 6,
        "l2_leaf_reg":           5.0,
        "loss_function":         "RMSE",
        "eval_metric":           "RMSE",
        "random_seed":           RANDOM_SEED,
        "early_stopping_rounds": 100,
    },
}

EXPERIMENT_CONFIG = {
    "output_dir":       "results/",
    "save_predictions": True,
    "random_seed":      RANDOM_SEED,
}