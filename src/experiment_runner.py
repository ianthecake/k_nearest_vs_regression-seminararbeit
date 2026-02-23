from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .data_generation import DatasetConfig, generate_dataset, MEAN_FUNCTIONS, NOISE_FN_TYPE
from .knn_evaluation import evaluate_knn_over_k, summarize_best_k


def run_multiple_seeds(
    experiment_name: str,
    base_cfg: DatasetConfig,
    seeds: List[int],
    k_values: List[int],
    *,
    weights: str = "uniform",
    metric: str = "minkowski",
    n_jobs: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Runs:
      for seed in seeds:
        data = generate_dataset(base_cfg with seed)
        res  = evaluate_knn_over_k(data, k_values)
        store (best_k, min_mse_vs_true, mse_vs_noisy_at_best_k)
        store full mse curve vs true mean

    Returns:
      df_runs: one row per seed
      curves_true: shape (len(seeds), len(k_values)) -> mse vs true mean
    """
    rows = []
    curves_true = []

    for seed in seeds:
        cfg = replace(base_cfg, seed=seed)
        data = generate_dataset(cfg)

        res = evaluate_knn_over_k(data, k_values, weights=weights, metric=metric, n_jobs=n_jobs)
        best = summarize_best_k(res)

        rows.append(
            {
                "experiment": experiment_name,
                "seed": seed,
                "best_k": int(best["best_k"]),
                "min_mse_vs_true": best["min_mse_vs_true"],
                "mse_vs_noisy_at_best_k": best["mse_vs_noisy_at_best_k"],
            }
        )
        curves_true.append(res["mse_vs_true_mean"])

    df_runs = pd.DataFrame(rows)
    curves_true = np.array(curves_true, dtype=float)
    return df_runs, curves_true


def summarize_across_seeds(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates mean/std across seeds per experiment.
    """
    out = (
        df_runs.groupby("experiment")
        .agg(
            n_runs=("seed", "count"),
            best_k_mean=("best_k", "mean"),
            best_k_std=("best_k", "std"),
            min_mse_true_mean=("min_mse_vs_true", "mean"),
            min_mse_true_std=("min_mse_vs_true", "std"),
            mse_noisy_mean=("mse_vs_noisy_at_best_k", "mean"),
            mse_noisy_std=("mse_vs_noisy_at_best_k", "std"),
        )
        .reset_index()
    )

    # nicer rounding
    out["best_k_mean"] = out["best_k_mean"].round(2)
    out["best_k_std"] = out["best_k_std"].round(2)
    for c in ["min_mse_true_mean", "min_mse_true_std", "mse_noisy_mean", "mse_noisy_std"]:
        out[c] = out[c].round(6)

    return out


def save_outputs(
    df_runs: pd.DataFrame,
    df_summary: pd.DataFrame,
    curves_by_experiment: Dict[str, np.ndarray],
    k_values: List[int],
    out_dir: str = "results",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df_runs.to_csv(Path(out_dir) / "runs.csv", index=False)
    df_summary.to_csv(Path(out_dir) / "summary.csv", index=False)

    # Save curves in one npz
    # Each experiment stored as array (n_runs, n_k)
    np.savez(
        Path(out_dir) / "curves_true_mean.npz",
        k_values=np.array(k_values, dtype=int),
        **{name: arr for name, arr in curves_by_experiment.items()},
    )

class OnlineMeanVariance:
    """
    Online mean/variance (Welford) for vectors.
    Stores mean and M2 per component.
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.n = 0
        self.mean = np.zeros(shape, dtype=float)
        self.M2 = np.zeros(shape, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.M2 / (self.n - 1)

def run_bias_variance_decomposition(
        experiment_name: str,
        base_cfg: DatasetConfig,
        seeds: List[int],
        k_values: List[int],
        *,
        fixed_test_seed: int = 123456,
        weights: str = "uniform",
        metric: str = "minkowski",
        n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Approximates bias^2 and variance over repeated runs (different seeds).

    Key idea:
      - Use a FIXED test set X_test across all runs to make predictions comparable.
      - For each k, aggregate predictions across runs using online mean/variance.

    Output columns per k:
      experiment, k, bias2, variance, noise, mse_est (bias2+var+noise)

    Note:
      noise is estimated as mean(sigma_test^2) on the fixed test set.
      (Works for homo + hetero noise models in your generator.)
    """
    # 1) Create a fixed test set + corresponding true mean + noise sigma
    test_cfg = replace(base_cfg, seed=fixed_test_seed)
    fixed = generate_dataset(test_cfg)
    X_test = fixed["X_test"]
    f_true = fixed["y_mean_test"]
    sigma_test = fixed["sigma_test"]
    noise = float(np.mean(sigma_test ** 2))

    # sanity: we need the true mean function for this experiment
    # (already available in fixed["y_mean_test"])

    # 2) Online aggregators per k
    agg_by_k: Dict[int, OnlineMeanVariance] = {
        int(k): OnlineMeanVariance(shape=(X_test.shape[0],))
        for k in k_values
    }

    # 3) For each run: generate training data with that seed, fit/predict for each k
    from sklearn.neighbors import KNeighborsRegressor

    for seed in seeds:
        train_cfg = replace(base_cfg, seed=seed)
        data = generate_dataset(train_cfg)
        X_train = data["X_train"]
        y_train = data["y_train"]

        for k in k_values:
            model = KNeighborsRegressor(
                n_neighbors=int(k),
                weights=weights,
                metric=metric,
                n_jobs=n_jobs,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            agg_by_k[int(k)].update(pred)

    # 4) Compute bias^2 and variance per k
    rows = []
    for k in k_values:
        k = int(k)
        mean_pred = agg_by_k[k].mean
        var_pred = agg_by_k[k].variance()

        bias2 = float(np.mean((mean_pred - f_true) ** 2))
        variance = float(np.mean(var_pred))
        mse_est = bias2 + variance + noise

        rows.append(
            {
                "experiment": experiment_name,
                "k": k,
                "bias2": bias2,
                "variance": variance,
                "noise": noise,
                "mse_est": mse_est,
            }
        )

    return pd.DataFrame(rows)

def run_dimension_sweep(
        sweep_name: str,
        base_cfg: DatasetConfig,
        d_values: List[int],
        seeds: List[int],
        k_values: List[int],
        *,
        n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    For each dimension d in d_values:
      - run multiple seeds
      - pick best_k by min MSE vs true mean per run
      - aggregate mean/std of best_k and min MSE vs true mean

    Returns a summary table by d.
    """
    all_runs = []

    for d in d_values:
        cfg_d = replace(base_cfg, d=int(d))
        # run like your earlier multi-seed logic, but inline for compactness:
        for seed in seeds:
            cfg = replace(cfg_d, seed=seed)
            data = generate_dataset(cfg)
            res = evaluate_knn_over_k(data, k_values, n_jobs=n_jobs)
            best = summarize_best_k(res)

            all_runs.append(
                {
                    "sweep": sweep_name,
                    "d": int(d),
                    "seed": seed,
                    "best_k": int(best["best_k"]),
                    "min_mse_vs_true": float(best["min_mse_vs_true"]),
                }
            )

    df_runs = pd.DataFrame(all_runs)

    df_summary = (
        df_runs.groupby(["sweep", "d"])
        .agg(
            n_runs=("seed", "count"),
            best_k_mean=("best_k", "mean"),
            best_k_std=("best_k", "std"),
            min_mse_true_mean=("min_mse_vs_true", "mean"),
            min_mse_true_std=("min_mse_vs_true", "std"),
        )
        .reset_index()
    )

    df_summary["best_k_mean"] = df_summary["best_k_mean"].round(2)
    df_summary["best_k_std"] = df_summary["best_k_std"].round(2)
    df_summary["min_mse_true_mean"] = df_summary["min_mse_true_mean"].round(6)
    df_summary["min_mse_true_std"] = df_summary["min_mse_true_std"].round(6)

    return df_summary


def run_variance_sweep(
    sweep_name: str,
    base_cfg: DatasetConfig,
    seeds: list[int],
    k_values: list[int],
    *,
    mode: str,
    sigma_values: list[float] | None = None,
    scale_values: list[float] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Variance sweeps:

    mode="homo":
      - vary sigma over sigma_values
      - cfg.noise_type forced to HOMO

    mode="hetero_scale":
      - keep heteroskedastic SHAPE (sigma_min..sigma_max),
        but multiply both by scale in scale_values:
          sigma_min' = scale*sigma_min
          sigma_max' = scale*sigma_max
      - cfg.noise_type kept as HETERO_LINEAR or HETERO_RADIAL (from base_cfg)

    Returns summary per noise setting:
      sweep, mode, noise_type, param, n_runs,
      best_k_mean/std, min_mse_true_mean/std
    """
    if mode not in {"homo", "hetero_scale"}:
        raise ValueError("mode must be 'homo' or 'hetero_scale'")

    rows = []

    if mode == "homo":
        if sigma_values is None:
            raise ValueError("sigma_values required for mode='homo'")
        for sigma in sigma_values:
            cfg_noise = replace(base_cfg, noise_type=NOISE_FN_TYPE.HOMO, sigma=float(sigma))
            for seed in seeds:
                cfg = replace(cfg_noise, seed=int(seed))
                data = generate_dataset(cfg)
                res = evaluate_knn_over_k(data, k_values, n_jobs=n_jobs)
                best = summarize_best_k(res)
                rows.append(
                    {
                        "sweep": sweep_name,
                        "mode": "homo",
                        "noise_type": cfg_noise.noise_type.value,
                        "param": float(sigma),  # sigma
                        "seed": int(seed),
                        "best_k": int(best["best_k"]),
                        "min_mse_vs_true": float(best["min_mse_vs_true"]),
                    }
                )

    else:  # hetero_scale
        if scale_values is None:
            raise ValueError("scale_values required for mode='hetero_scale'")
        if base_cfg.noise_type not in {NOISE_FN_TYPE.HETERO_LINEAR, NOISE_FN_TYPE.HETERO_RADIAL}:
            raise ValueError("For hetero_scale, base_cfg.noise_type must be HETERO_LINEAR or HETERO_RADIAL")

        for scale in scale_values:
            cfg_noise = replace(
                base_cfg,
                sigma_min=float(base_cfg.sigma_min) * float(scale),
                sigma_max=float(base_cfg.sigma_max) * float(scale),
            )
            for seed in seeds:
                cfg = replace(cfg_noise, seed=int(seed))
                data = generate_dataset(cfg)
                res = evaluate_knn_over_k(data, k_values, n_jobs=n_jobs)
                best = summarize_best_k(res)
                rows.append(
                    {
                        "sweep": sweep_name,
                        "mode": "hetero_scale",
                        "noise_type": cfg_noise.noise_type.value,
                        "param": float(scale),  # scale factor
                        "seed": int(seed),
                        "best_k": int(best["best_k"]),
                        "min_mse_vs_true": float(best["min_mse_vs_true"]),
                    }
                )

    df_runs = pd.DataFrame(rows)

    df_summary = (
        df_runs.groupby(["sweep", "mode", "noise_type", "param"])
        .agg(
            n_runs=("seed", "count"),
            best_k_mean=("best_k", "mean"),
            best_k_std=("best_k", "std"),
            min_mse_true_mean=("min_mse_vs_true", "mean"),
            min_mse_true_std=("min_mse_vs_true", "std"),
        )
        .reset_index()
    )

    df_summary["best_k_mean"] = df_summary["best_k_mean"].round(2)
    df_summary["best_k_std"] = df_summary["best_k_std"].round(2)
    df_summary["min_mse_true_mean"] = df_summary["min_mse_true_mean"].round(6)
    df_summary["min_mse_true_std"] = df_summary["min_mse_true_std"].round(6)

    return df_summary