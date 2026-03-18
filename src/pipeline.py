"""
Полный пайплайн эксперимента
"""

try:
    from . import data as data_module
    from . import clustering
    from . import transforms
    from . import features as feat_module
    from . import models as model_module
    from . import metrics as metric_module
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src.data as data_module          # type: ignore
    import src.clustering as clustering     # type: ignore
    import src.transforms as transforms     # type: ignore
    import src.features as feat_module      # type: ignore
    import src.models as model_module       # type: ignore
    import src.metrics as metric_module     # type: ignore

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# Минимальная длина ряда для polyfit
_POLYFIT_MIN_LEN = 10


def _robust_trend(y: np.ndarray) -> float:
    """
    Робастная оценка тренда
    """
    n = len(y)
    if n < 2:
        return 0.0
    if n >= _POLYFIT_MIN_LEN:
        return float(np.polyfit(range(n), y, 1)[0])
    mid = n // 2
    return float(np.mean(y[mid:]) - np.mean(y[:mid]))

def _compute_series_meta(
    series_list:    List[np.ndarray],
    uid_list:       List[str],
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Статические признаки
    """
    rows = []
    for uid, y, cluster in zip(uid_list, series_list, cluster_labels):
        rows.append({
            "unique_id":    uid,
            "meta_mean":    float(np.mean(y)),
            "meta_std":     float(np.std(y)),
            "meta_cv":      float(np.std(y) / (np.mean(y) + 1e-8)),
            "meta_skew":    float(pd.Series(y).skew()),
            "meta_kurt":    float(pd.Series(y).kurt()),
            "meta_trend":   _robust_trend(y),
            "meta_cluster": int(cluster),
        })
    return pd.DataFrame(rows).set_index("unique_id")


def _attach_meta(X: pd.DataFrame,
                 uid_series: pd.Series,
                 meta_df: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["unique_id"] = uid_series.values
    X = X.merge(meta_df.reset_index(), on="unique_id", how="left")
    X = X.drop(columns=["unique_id"])
    return X

def _build_tabular(series_list: List[np.ndarray],
                   uid_list: List[str],
                   feature_config: dict) -> pd.DataFrame:
    rows = []
    for uid, y in zip(uid_list, series_list):
        for t, val in enumerate(y):
            rows.append({"unique_id": uid, "ds": t, "y": float(val)})
    return feat_module.engineer_features(pd.DataFrame(rows), feature_config)


def _split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    drop_cols  = ["unique_id", "ds", "y"]
    uid_col    = df["unique_id"] if "unique_id" in df.columns else pd.Series(
        dtype=str, name="unique_id"
    )
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, df["y"].values, uid_col


def _val_split(X: pd.DataFrame, y: np.ndarray,
               val_frac: float = 0.15) -> Tuple:
    n_val = max(1, int(len(y) * val_frac))
    return X.iloc[:-n_val], y[:-n_val], X.iloc[-n_val:], y[-n_val:]

def _build_direct_dataset(
    scaled_series:  List[np.ndarray],
    uid_list:       List[str],
    feature_config: dict,
    meta_df:        pd.DataFrame,
    step:           int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rows_feat, rows_target, rows_uid = [], [], []
    for uid, y_sc in zip(uid_list, scaled_series):
        n = len(y_sc)
        for t in range(n - step):
            rows_feat.append({"unique_id": uid, "ds": t, "y": float(y_sc[t])})
            rows_target.append(float(y_sc[t + step]))
            rows_uid.append(uid)

    feat_df          = feat_module.engineer_features(
        pd.DataFrame(rows_feat), feature_config
    )
    X, _, uid_series = _split_xy(feat_df)
    X                = _attach_meta(X, uid_series, meta_df)

    targets = np.array(rows_target)
    if len(targets) > len(X):
        targets = targets[-len(X):]
    return X, targets


def _build_prediction_features(
    scaled_series:  List[np.ndarray],
    uid_list:       List[str],
    feature_config: dict,
    meta_df:        pd.DataFrame,
) -> pd.DataFrame:
    context_rows = []
    for uid, y_sc in zip(uid_list, scaled_series):
        for t, val in enumerate(y_sc):
            context_rows.append({"unique_id": uid, "ds": t, "y": float(val)})

    full_df   = feat_module.engineer_features(
        pd.DataFrame(context_rows), feature_config
    )
    last_rows = full_df.groupby("unique_id").last().reset_index()
    X, _, uid_series = _split_xy(last_rows)
    X                = _attach_meta(X, uid_series, meta_df)
    return X

class ExperimentPipeline:

    def __init__(self, config: dict):
        self.config    = config
        self.results:  Dict = {}
        self.data_cfg  = config.get("data",       {})
        self.clust_cfg = config.get("clustering", {})
        self.trans_cfg = config.get("transforms", {})
        self.feat_cfg  = config.get("features",   {})
        self.model_cfg = config.get("models",     {})
        self.exp_cfg   = config.get("experiment", {})

    def load_and_prepare_data(self):
        raw = data_module.load_m4_data(
            frequency=self.data_cfg.get("frequency", "monthly")
        )
        sampled = data_module.sample_series(
            raw,
            n_samples=self.data_cfg.get("sample_size", 150),
            random_seed=self.data_cfg.get("random_seed", 42),
        )
        horizon = self.data_cfg.get("forecast_horizon", 18)
        train_df, test_df = data_module.train_test_split(sampled, horizon=horizon)
        train_list, uid_list = data_module.series_to_list(train_df)
        test_list,  _        = data_module.series_to_list(test_df)
        print(f"[pipeline] {len(train_list)} рядов | "
              f"средняя длина обучения = "
              f"{np.mean([len(y) for y in train_list]):.1f} | "
              f"длина теста = {horizon}")
        return train_list, test_list, uid_list

    def apply_clustering(self, series_list: List[np.ndarray]) -> np.ndarray:
        feat_df    = clustering.extract_features(series_list)
        n_clusters = self.clust_cfg.get("n_clusters", None)
        k_min      = self.clust_cfg.get("k_min", 3)
        k_max      = self.clust_cfg.get("k_max", 8)
        rs         = self.clust_cfg.get("random_state", 42)
        labels, score, best_k = clustering.cluster_series(
            feat_df, n_clusters=n_clusters,
            k_min=k_min, k_max=k_max, random_state=rs,
        )
        print(f"[pipeline] Выбрано k={best_k}")
        feat_df["cluster"] = labels
        print("\n[pipeline] Средние статпризнаки по кластерам:")
        print(feat_df.groupby("cluster").mean().round(3).to_string())
        print()
        return labels

    def _run_one_transform(
        self,
        kind:           str,
        train_list:     List[np.ndarray],
        test_list:      List[np.ndarray],
        uid_list:       List[str],
        cluster_labels: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        horizon  = self.data_cfg.get("forecast_horizon", 18)
        period   = self.feat_cfg.get("seasonal_period", 12)
        val_frac = self.model_cfg.get("catboost_val_frac", 0.15)

        tr_series, aux_list = [], []
        for y in train_list:
            y_tr, aux = transforms.apply_transform(y, kind)
            tr_series.append(np.asarray(y_tr, dtype=float))
            aux_list.append(aux)

        scaler        = model_module.SeriesScaler()
        scaled_series = scaler.fit_transform(tr_series)

        meta_df = _compute_series_meta(train_list, uid_list, cluster_labels)
        numeric_meta = ["meta_mean", "meta_std", "meta_cv",
                        "meta_skew", "meta_kurt", "meta_trend"]
        for col in numeric_meta:
            m_mean = meta_df[col].mean()
            m_std  = meta_df[col].std() + 1e-8
            meta_df[col] = (meta_df[col] - m_mean) / m_std

        direct_models = []
        cb_config     = self.model_cfg.get("catboost", {})

        for step in range(1, horizon + 1):
            X_step, y_step = _build_direct_dataset(
                scaled_series, uid_list, self.feat_cfg, meta_df, step
            )
            X_tr, y_tr, X_val, y_val = _val_split(X_step, y_step, val_frac)
            m = model_module.train_catboost(
                X_tr, y_tr, config=cb_config, X_val=X_val, y_val=y_val
            )
            direct_models.append(m)

        X_pred = _build_prediction_features(
            scaled_series, uid_list, self.feat_cfg, meta_df
        )
        cb_preds_scaled = np.stack(
            [m.predict(X_pred) for m in direct_models], axis=1
        )

        cb_preds_orig = np.empty_like(cb_preds_scaled, dtype=float)
        for i in range(len(train_list)):
            pred_tr = cb_preds_scaled[i] * scaler.stds_[i] + scaler.means_[i]
            pred_orig = transforms.inverse_transform(
                pred_tr, kind=kind, aux=aux_list[i],
                last_train_val=train_list[i][-1] if kind == "diff" else None,
            )
            cb_preds_orig[i] = np.clip(
                np.asarray(pred_orig, dtype=float), 0, None
            )

        bl_models = model_module.train_baseline_models(
            train_list,
            methods=self.model_cfg.get(
                "baseline_models", ["naive", "seasonal_naive", "mean", "median"]
            ),
            seasonal_period=period,
        )
        sf_models = model_module.train_statsforecast_models(
            train_list,
            model_names=self.model_cfg.get(
                "statsforecast_models", ["AutoTheta", "AutoETS"]
            ),
            seasonal_period=period,
        )
        all_preds = model_module.predict_baselines(bl_models, horizon)
        all_preds.update(model_module.predict_baselines(sf_models, horizon))
        all_preds["catboost"] = cb_preds_orig

        global_df = metric_module.evaluate_all_models(
            predictions=all_preds,
            y_true_list=test_list,
            y_train_list=train_list,
            seasonality=period,
        )

        cluster_dfs: Dict[int, pd.DataFrame] = {}
        for c in sorted(np.unique(cluster_labels)):
            idx = np.where(cluster_labels == c)[0]
            cluster_dfs[int(c)] = metric_module.evaluate_all_models(
                predictions={k: v[idx] for k, v in all_preds.items()},
                y_true_list=[test_list[i]  for i in idx],
                y_train_list=[train_list[i] for i in idx],
                seasonality=period,
            )

        return global_df, cluster_dfs

    def run(self) -> Dict:

        train_list, test_list, uid_list = self.load_and_prepare_data()
        self.results["n_series"] = len(train_list)

        cluster_labels = self.apply_clustering(train_list)
        self.results["cluster_labels"] = cluster_labels

        candidates = self.trans_cfg.get(
            "candidates", ["none", "log1p", "boxcox", "diff"]
        )

        per_transform:     Dict[str, pd.DataFrame]            = {}
        cluster_breakdown: Dict[str, Dict[int, pd.DataFrame]] = {}

        for kind in candidates:
                global_df, cluster_dfs = self._run_one_transform(
                    kind, train_list, test_list, uid_list, cluster_labels
                )
                per_transform[kind]     = global_df
                cluster_breakdown[kind] = cluster_dfs

                print("\n  Глобальные метрики (все ряды):")
                print(global_df.round(4).to_string())
                print("\n  Метрики по кластерам:")
                for c, df in cluster_dfs.items():
                    n_c = int(np.sum(cluster_labels == c))
                    print(f"\n    Кластер {c} (n={n_c}):")
                    print(df.round(4).to_string())

        self.results["per_transform"]     = per_transform
        self.results["cluster_breakdown"] = cluster_breakdown

        if per_transform:
            summary = pd.concat(per_transform, names=["transform", "model"])
            self.results["summary"] = summary
            print("\nСводная таблица: трансформация × модель (все ряды)")
            print(summary.round(4).to_string())

        if cluster_breakdown:
            print("\nИТОГ: лучшая трансформация для CatBoost по кластерам")
            rows = []
            for c in sorted(np.unique(cluster_labels)):
                row = {"cluster": c,
                       "n_series": int(np.sum(cluster_labels == c))}
                for metric in ["sMAPE", "MASE", "MAE"]:
                    best_transform, best_val = None, float("inf")
                    for kind, cdfs in cluster_breakdown.items():
                        if c not in cdfs:
                            continue
                        df = cdfs[c]
                        if "catboost" in df.index and metric in df.columns:
                            val = float(df.loc["catboost", metric])
                            if val < best_val:
                                best_val, best_transform = val, kind
                    row[f"best_{metric}"] = best_transform
                    row[f"{metric}"]      = round(best_val, 4)
                rows.append(row)

            conclusion_df = pd.DataFrame(rows).set_index("cluster")
            self.results["conclusion"] = conclusion_df
            print(conclusion_df.to_string())

        return self.results