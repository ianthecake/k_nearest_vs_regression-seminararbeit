from __future__ import annotations

from typing import Dict, Iterable, List
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def evaluate_knn_over_k(
    data: Dict[str, np.ndarray],
    k_values: Iterable[int],
    *,
    weights: str = "uniform",
    metric: str = "minkowski",
    n_jobs: int | None = None,
) -> Dict[str, List[float]]:
    """
    Train kNN on (X_train, y_train) and evaluate on X_test for each k.

    Returns dict:
      k
      mse_vs_true_mean  (vs y_mean_test)
      mse_vs_noisy_y    (vs y_test)
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]

    y_mean_test = data["y_mean_test"]
    y_test = data["y_test"]

    results = {"k": [], "mse_vs_true_mean": [], "mse_vs_noisy_y": []}

    for k in k_values:
        model = KNeighborsRegressor(
            n_neighbors=int(k),
            weights=weights,
            metric=metric,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results["k"].append(int(k))
        results["mse_vs_true_mean"].append(float(mean_squared_error(y_mean_test, y_pred)))
        results["mse_vs_noisy_y"].append(float(mean_squared_error(y_test, y_pred)))

    return results


def evaluate_knn_train_test(
    data: dict,
    k_values,
    *,
    weights="uniform",
    metric="minkowski",
    n_jobs=None,
):
    """
    Computes:
      - train MSE vs noisy y_train
      - test MSE vs noisy y_test
      - test MSE vs true mean y_mean_test
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_mean_test = data["y_mean_test"]

    out = {
        "k": [],
        "mse_train": [],
        "mse_test_noisy": [],
        "mse_test_true": [],
    }

    for k in k_values:
        model = KNeighborsRegressor(
            n_neighbors=int(k),
            weights=weights,
            metric=metric,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        out["k"].append(int(k))
        out["mse_train"].append(mean_squared_error(y_train, y_pred_train))
        out["mse_test_noisy"].append(mean_squared_error(y_test, y_pred_test))
        out["mse_test_true"].append(mean_squared_error(y_mean_test, y_pred_test))

    return out


def summarize_best_k(results: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Finds k that minimizes mse_vs_true_mean and returns summary numbers.
    """
    k = np.array(results["k"])
    mse_true = np.array(results["mse_vs_true_mean"])
    mse_noisy = np.array(results["mse_vs_noisy_y"])

    idx = int(np.argmin(mse_true))

    return {
        "best_k": float(k[idx]),
        "min_mse_vs_true": float(mse_true[idx]),
        "mse_vs_noisy_at_best_k": float(mse_noisy[idx]),
    }