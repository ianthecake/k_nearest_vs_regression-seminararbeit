from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation import DatasetConfig, MEAN_FN_TYPE, NOISE_FN_TYPE
from src.experiment_runner import run_variance_sweep


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_variance_sweep(df: pd.DataFrame, title: str, x_label: str, filename_prefix: str):
    # Plot min MSE vs param
    plt.figure()
    plt.errorbar(
        df["param"],
        df["min_mse_true_mean"],
        yerr=df["min_mse_true_std"].fillna(0),
        marker="o",
        linestyle="-",
    )
    plt.xlabel(x_label)
    plt.ylabel("min MSE vs true (mean ± std)")
    plt.title(f"{title}: error vs noise")
    plt.tight_layout()
    out1 = PLOTS_DIR / f"{filename_prefix}_mse_vs_noise.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # Plot best k vs param
    plt.figure()
    plt.errorbar(
        df["param"],
        df["best_k_mean"],
        yerr=df["best_k_std"].fillna(0),
        marker="o",
        linestyle="-",
    )
    plt.xlabel(x_label)
    plt.ylabel("best k (mean ± std)")
    plt.title(f"{title}: best k vs noise")
    plt.tight_layout()
    out2 = PLOTS_DIR / f"{filename_prefix}_bestk_vs_noise.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


def print_terminal_summary(df: pd.DataFrame, title: str, x_label: str):
    print("\n================ VARIANCE SWEEP ================\n")
    print(title)
    print(df.to_string(index=False))

    # quick qualitative interpretation
    low = df.sort_values("param").iloc[0]
    high = df.sort_values("param").iloc[-1]

    print("\nInterpretation (auto):")
    print(
        f"- As {x_label} increases from {low['param']} to {high['param']}, "
        f"min MSE vs true changes from {low['min_mse_true_mean']} to {high['min_mse_true_mean']}."
    )
    print(
        f"- Best k (mean) shifts from {low['best_k_mean']} to {high['best_k_mean']} "
        "(often increases because more averaging helps against noise)."
    )
    print("=================================================\n")


def main():
    seeds = list(range(1, 31))
    k_values = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200]

    RESULTS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------
    # A) Homoskedastic variance sweep (sigma)
    # Keep function + dimension fixed to isolate sigma effect.
    # Recommended: 1D sine or 1D piecewise (both informative).
    # ---------------------------------------------
    homo_base = DatasetConfig(
        d=1,
        mean_fn=MEAN_FN_TYPE.SINE_1D,
        noise_type=NOISE_FN_TYPE.HOMO,
        sigma=0.2,   # overwritten by sweep
        seed=0,
    )

    sigma_values = [0.05, 0.1, 0.2, 0.4, 0.8]

    df_homo = run_variance_sweep(
        sweep_name="sine_1d_homo_sigma",
        base_cfg=homo_base,
        seeds=seeds,
        k_values=k_values,
        mode="homo",
        sigma_values=sigma_values,
        n_jobs=-1,
    )

    df_homo.to_csv(RESULTS_DIR / "variance_sweep_homo_sigma.csv", index=False)
    print_terminal_summary(df_homo, "Homoskedastic sweep: sine_1d", "sigma")
    plot_variance_sweep(df_homo, "Homoskedastic sweep: sine_1d", "sigma", "homo_sigma_sine_1d")

    # ---------------------------------------------
    # B) Heteroskedastic sweep (scale sigma_min/max)
    # Keep heteroskedastic SHAPE fixed, scale amplitude.
    # You can do this for linear or radial hetero.
    # ---------------------------------------------
    hetero_base = DatasetConfig(
        d=1,
        mean_fn=MEAN_FN_TYPE.PIECEWISE_1D,
        noise_type=NOISE_FN_TYPE.HETERO_LINEAR,
        sigma_min=0.05,
        sigma_max=0.5,
        seed=0,
    )

    scale_values = [0.5, 1.0, 2.0, 4.0]

    df_hetero = run_variance_sweep(
        sweep_name="piecewise_1d_hetero_linear_scale",
        base_cfg=hetero_base,
        seeds=seeds,
        k_values=k_values,
        mode="hetero_scale",
        scale_values=scale_values,
        n_jobs=-1,
    )

    df_hetero.to_csv(RESULTS_DIR / "variance_sweep_hetero_scale.csv", index=False)
    print_terminal_summary(df_hetero, "Heteroskedastic sweep: piecewise_1d (linear)", "scale")
    plot_variance_sweep(df_hetero, "Heteroskedastic sweep: piecewise_1d (linear)", "scale", "hetero_scale_piecewise_1d_linear")


if __name__ == "__main__":
    main()