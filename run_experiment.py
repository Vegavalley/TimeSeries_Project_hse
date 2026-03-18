"""
Эксперимент
"""

import os
import pandas as pd

from config import (
    DATA_CONFIG, CLUSTERING_CONFIG, TRANSFORM_CONFIG,
    FEATURE_CONFIG, MODEL_CONFIG, EXPERIMENT_CONFIG,
)
from src.pipeline import ExperimentPipeline


def main():
    config = {
        "data":       DATA_CONFIG,
        "clustering": CLUSTERING_CONFIG,
        "transforms": TRANSFORM_CONFIG,
        "features":   FEATURE_CONFIG,
        "models":     MODEL_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
    }

    pipeline = ExperimentPipeline(config)
    results  = pipeline.run()

    output_dir = EXPERIMENT_CONFIG.get("output_dir", "results/")
    os.makedirs(output_dir, exist_ok=True)

    # Сводная таблица
    if "summary" in results:
        summary_flat = (
            results["summary"]
            .reset_index()
            .rename(columns={"level_0": "transform", "level_1": "model"})
        )
        path = os.path.join(output_dir, "summary.csv")
        summary_flat.to_csv(path, index=False)
        print(f"\n[run] Сводная таблица → {path}")

    # Разбивка по кластерам
    if "cluster_breakdown" in results:
        rows = []
        for kind, cluster_dfs in results["cluster_breakdown"].items():
            for c, df in cluster_dfs.items():
                tmp = df.reset_index()
                tmp.insert(0, "cluster", c)
                tmp.insert(0, "transform", kind)
                rows.append(tmp)
        if rows:
            breakdown_df = pd.concat(rows, ignore_index=True)
            path = os.path.join(output_dir, "cluster_breakdown.csv")
            breakdown_df.to_csv(path, index=False)
            print(f"[run] Разбивка по кластерам → {path}")

    # Итог
    if "conclusion" in results:
        path = os.path.join(output_dir, "conclusion.csv")
        results["conclusion"].to_csv(path)
        print(f"[run] Итог гипотезы → {path}")

    # Лучшая трансформация
    if "summary" in results:
        summary_flat = (
            results["summary"]
            .reset_index()
            .rename(columns={"level_0": "transform", "level_1": "model"})
        )
        metrics = [c for c in summary_flat.columns
                   if c not in ("transform", "model")]
        cb = summary_flat[summary_flat["model"] == "catboost"]
        if not cb.empty:
            print("\n[run] Лучшая трансформация для CatBoost:")
            for m in metrics:
                best = cb.loc[cb[m].idxmin()]
                print(f"  {m:6s}: {best['transform']:8s}  ({best[m]:.4f})")

    return results


if __name__ == "__main__": #заглушка (проблемы с импортами)
    main()