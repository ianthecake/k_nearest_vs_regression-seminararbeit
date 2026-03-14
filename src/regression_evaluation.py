from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def _fit_predict_linear(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _fit_predict_polynomial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    degree: int,
    include_bias: bool = False,
) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("linreg", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_parametric_regressions(
    data: Dict[str, np.ndarray],
    polynomial_degrees: List[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluates:
      - linear regression
      - polynomial regression for each degree in polynomial_degrees

    Metrics:
      - mse_vs_true_mean
      - mse_vs_noisy

    Returns:
      DataFrame with columns:
        model, degree, mse_vs_true_mean, mse_vs_noisy
    """
    if polynomial_degrees is None:
        polynomial_degrees = [2, 3]

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_mean_test = data["y_mean_test"]

    rows = []

    # Linear regression
    pred_linear = _fit_predict_linear(X_train, y_train, X_test)
    rows.append(
        {
            "model": "linear",
            "degree": 1,
            "mse_vs_true_mean": float(mean_squared_error(y_mean_test, pred_linear)),
            "mse_vs_noisy": float(mean_squared_error(y_test, pred_linear)),
        }
    )

    # Polynomial regression
    for degree in polynomial_degrees:
        pred_poly = _fit_predict_polynomial(X_train, y_train, X_test, degree=int(degree))
        rows.append(
            {
                "model": f"polynomial_deg_{int(degree)}",
                "degree": int(degree),
                "mse_vs_true_mean": float(mean_squared_error(y_mean_test, pred_poly)),
                "mse_vs_noisy": float(mean_squared_error(y_test, pred_poly)),
            }
        )

    return pd.DataFrame(rows)


def summarize_best_parametric_model(res: pd.DataFrame) -> dict:
    """
    Picks best parametric model by smallest mse_vs_true_mean.
    """
    best_idx = res["mse_vs_true_mean"].idxmin()
    best_row = res.loc[best_idx]

    return {
        "best_model": str(best_row["model"]),
        "best_degree": int(best_row["degree"]),
        "min_mse_vs_true": float(best_row["mse_vs_true_mean"]),
        "mse_vs_noisy_at_best_model": float(best_row["mse_vs_noisy"]),
    }