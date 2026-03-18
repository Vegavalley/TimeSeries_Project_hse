"""
Кластеризация временных рядов по стат признакам.
Число кластеров - по силуэту
"""
 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
 
def extract_features(series_list) -> pd.DataFrame:
    """Извлекает статистические признаки"""
    rows = []
    for y in series_list:
        rows.append({
            "mean":  np.mean(y),
            "std":   np.std(y),
            "cv":    np.std(y) / (np.mean(y) + 1e-6),
            "skew":  pd.Series(y).skew(),
            "kurt":  pd.Series(y).kurt(),
            "trend": np.polyfit(range(len(y)), y, 1)[0],
        })
    return pd.DataFrame(rows)
 
 
def select_k_by_silhouette(X_scaled: np.ndarray,
                            k_min: int = 3,
                            k_max: int = 8,
                            random_state: int = 42) -> tuple:
    """
    Перебирает k от k_min до k_max, возвращает (best_k, scores_dict) (по максиму силуэта)
    """
    scores = {}
    for k in range(k_min, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        lab = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, lab)
 
    best_k = max(scores, key=scores.__getitem__)
    print("[clustering] силуэт:")
    for k, s in scores.items():
        print(f"  k={k}: {s:.4f}")
 
    return best_k, scores
 
 
def cluster_series(features: pd.DataFrame,
                   n_clusters: int = None,
                   k_min: int = 3,
                   k_max: int = 8,
                   random_state: int = 42) -> tuple:
    """
    Кластеризует ряды
    """
    X = StandardScaler().fit_transform(features)
 
    if n_clusters is None:
        best_k, _ = select_k_by_silhouette(X, k_min, k_max, random_state)
    else:
        best_k = n_clusters
        print(f"[clustering] k = {best_k}")
 
    km     = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels)
 
    return labels, score, best_k