from __future__ import annotations
import pandas as pd

from src.data_generation import DatasetConfig, MEAN_FN_TYPE, NOISE_FN_TYPE
from src.experiment_runner import (
    run_multiple_seeds,
    summarize_across_seeds,
    save_outputs,
    run_multiple_seeds_parametric,
    summarize_parametric_across_seeds,
    save_parametric_outputs,
    build_model_comparison_table,
)


def main():
    seeds = list(range(1, 31))  # 30 runs
    k_values = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200]
    polynomial_degrees = [2, 3]  # keep modest, especially for d=5 and d=10

    experiments = {
        "sine_1d_homo": DatasetConfig(
            d=1,
            mean_fn=MEAN_FN_TYPE.SINE_1D,
            noise_type=NOISE_FN_TYPE.HOMO,
            sigma=0.2,
            seed=0,
        ),
        "piecewise_1d_hetero_linear": DatasetConfig(
            d=1,
            mean_fn=MEAN_FN_TYPE.PIECEWISE_1D,
            noise_type=NOISE_FN_TYPE.HETERO_LINEAR,
            sigma_min=0.05,
            sigma_max=0.5,
            seed=0,
        ),
        "product_sine_5d_homo": DatasetConfig(
            d=5,
            mean_fn=MEAN_FN_TYPE.PRODUCT_SINE,
            noise_type=NOISE_FN_TYPE.HOMO,
            sigma=0.15,
            seed=0,
        ),
        "linear_sum_10d_hetero_radial": DatasetConfig(
            d=10,
            mean_fn=MEAN_FN_TYPE.SUM_LINEAR,
            noise_type=NOISE_FN_TYPE.HETERO_RADIAL,
            sigma_min=0.05,
            sigma_max=0.6,
            seed=0,
        ),
    }

    # -------------------------
    # kNN experiments
    # -------------------------
    all_runs_knn = []
    curves = {}

    for name, cfg in experiments.items():
        df_runs, curves_true = run_multiple_seeds(
            experiment_name=name,
            base_cfg=cfg,
            seeds=seeds,
            k_values=k_values,
            n_jobs=-1,
        )
        all_runs_knn.append(df_runs)
        curves[name] = curves_true

    df_runs_knn = pd.concat(all_runs_knn, ignore_index=True)
    df_summary_knn = summarize_across_seeds(df_runs_knn)

    save_outputs(
        df_runs=df_runs_knn,
        df_summary=df_summary_knn,
        curves_by_experiment=curves,
        k_values=k_values,
        out_dir="results",
    )

    # -------------------------
    # linear / polynomial baselines
    # -------------------------
    all_runs_param = []

    for name, cfg in experiments.items():
        df_runs_param = run_multiple_seeds_parametric(
            experiment_name=name,
            base_cfg=cfg,
            seeds=seeds,
            polynomial_degrees=polynomial_degrees,
        )
        all_runs_param.append(df_runs_param)

    df_runs_param = pd.concat(all_runs_param, ignore_index=True)
    df_summary_param = summarize_parametric_across_seeds(df_runs_param)

    save_parametric_outputs(
        df_runs=df_runs_param,
        df_summary=df_summary_param,
        out_dir="results",
        prefix="parametric",
    )

    # -------------------------
    # direct comparison table
    # -------------------------
    df_comparison = build_model_comparison_table(df_summary_knn, df_summary_param)
    df_comparison.to_csv("results/knn_vs_parametric_summary.csv", index=False)

    print("\n=== kNN Summary (mean ± std across seeds) ===")
    print(df_summary_knn.to_string(index=False))

    print("\n=== Parametric Summary (mean ± std across seeds) ===")
    print(df_summary_param.to_string(index=False))

    print("\n=== Direct Comparison: kNN vs Parametric ===")
    print(df_comparison.to_string(index=False))

    print("\nSaved to:")
    print("  results/runs.csv")
    print("  results/summary.csv")
    print("  results/curves_true_mean.npz")
    print("  results/parametric_runs.csv")
    print("  results/parametric_summary.csv")
    print("  results/knn_vs_parametric_summary.csv")


if __name__ == "__main__":
    main()