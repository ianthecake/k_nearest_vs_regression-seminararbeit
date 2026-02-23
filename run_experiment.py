from __future__ import annotations
import pandas as pd

from src.data_generation import DatasetConfig, MEAN_FN_TYPE, NOISE_FN_TYPE
from src.experiment_runner import run_multiple_seeds, summarize_across_seeds, save_outputs


def main():
    seeds = list(range(1, 31))  # 30 runs
    k_values = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200]

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

    all_runs = []
    curves = {}

    for name, cfg in experiments.items():
        df_runs, curves_true = run_multiple_seeds(
            experiment_name=name,
            base_cfg=cfg,
            seeds=seeds,
            k_values=k_values,
            n_jobs=-1,  # uses all  cpu cores
        )
        all_runs.append(df_runs)
        curves[name] = curves_true


    df_runs_all = pd.concat(all_runs, ignore_index=True)
    df_summary = summarize_across_seeds(df_runs_all)

    save_outputs(
        df_runs=df_runs_all,
        df_summary=df_summary,
        curves_by_experiment=curves,
        k_values=k_values,
        out_dir="results",
    )

    print("\n=== Summary (mean ± std across seeds) ===")
    print(df_summary.to_string(index=False))
    print("\nSaved to: results/runs.csv, results/summary.csv, results/curves_true_mean.npz")


if __name__ == "__main__":
    main()