from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation import DatasetConfig, MEAN_FN_TYPE, NOISE_FN_TYPE
from src.experiment_runner import run_bias_variance_decomposition, run_dimension_sweep


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_dimension_summary(df_dim: pd.DataFrame, title_prefix: str):
    # Plot: best_k_mean vs d
    plt.figure()
    plt.plot(df_dim["d"], df_dim["best_k_mean"], marker="o")
    plt.fill_between(
        df_dim["d"],
        df_dim["best_k_mean"] - df_dim["best_k_std"].fillna(0),
        df_dim["best_k_mean"] + df_dim["best_k_std"].fillna(0),
        alpha=0.25,
    )
    plt.xlabel("dimension d")
    plt.ylabel("best k (mean ± std)")
    plt.title(f"{title_prefix}: best k vs dimension")
    plt.tight_layout()
    out1 = PLOTS_DIR / f"{title_prefix}_bestk_vs_d.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # Plot: min MSE vs d
    plt.figure()
    plt.plot(df_dim["d"], df_dim["min_mse_true_mean"], marker="o")
    plt.fill_between(
        df_dim["d"],
        df_dim["min_mse_true_mean"] - df_dim["min_mse_true_std"].fillna(0),
        df_dim["min_mse_true_mean"] + df_dim["min_mse_true_std"].fillna(0),
        alpha=0.25,
    )
    plt.xlabel("dimension d")
    plt.ylabel("min MSE vs true (mean ± std)")
    plt.title(f"{title_prefix}: min MSE vs dimension")
    plt.tight_layout()
    out2 = PLOTS_DIR / f"{title_prefix}_minmse_vs_d.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


def plot_bias_variance(df_bv: pd.DataFrame, title: str):
    k = df_bv["k"].to_numpy()

    plt.figure()
    plt.plot(k, df_bv["bias2"].to_numpy(), marker="o", label="bias^2")
    plt.plot(k, df_bv["variance"].to_numpy(), marker="o", label="variance")
    plt.plot(k, df_bv["noise"].to_numpy(), marker="o", label="noise (mean sigma^2)")
    plt.plot(k, df_bv["mse_est"].to_numpy(), marker="o", label="bias^2 + var + noise")

    plt.xscale("log")
    plt.xlabel("k (log scale)")
    plt.ylabel("error components")
    plt.title(f"{title}: bias-variance decomposition")
    plt.legend()
    plt.tight_layout()

    out = PLOTS_DIR / f"{title}_bias_variance.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def print_terminal_summary_dimension(df_dim: pd.DataFrame, title: str):
    print("\n================ DIMENSION ANALYSIS ================\n")
    print(title)
    print(df_dim.to_string(index=False))

    # simple qualitative statement:
    d_min = df_dim.sort_values("min_mse_true_mean").iloc[0]
    d_max = df_dim.sort_values("min_mse_true_mean", ascending=False).iloc[0]

    print("\nInterpretation (auto):")
    print(
        f"- Best average performance at d={int(d_min['d'])} "
        f"(min MSE ≈ {d_min['min_mse_true_mean']})."
    )
    print(
        f"- Worst average performance at d={int(d_max['d'])} "
        f"(min MSE ≈ {d_max['min_mse_true_mean']})."
    )
    print(
        "- Typically, min MSE grows with dimension because neighborhoods become less 'local' "
        "(curse of dimensionality).\n"
    )
    print(
        "- Normalized MSE (dividing by noise variance) makes results comparable and highlights "
        "how much worse kNN gets beyond the irreducible noise level."
    )
    print("====================================================\n")


def print_terminal_summary_bias_variance(df_bv: pd.DataFrame, title: str):
    print("\n============== BIAS–VARIANCE ANALYSIS ==============\n")
    print(title)

    # Find k minimizing mse_est
    best = df_bv.sort_values("mse_est").iloc[0]
    print(
        f"- k that minimizes (bias^2+var+noise): k={int(best['k'])}, "
        f"MSE_est≈{best['mse_est']:.6f} "
        f"(bias^2≈{best['bias2']:.6f}, var≈{best['variance']:.6f}, noise≈{best['noise']:.6f})"
    )

    # show trend at small vs large k
    small = df_bv.sort_values("k").iloc[0]
    large = df_bv.sort_values("k").iloc[-1]
    print("\nTypical pattern:")
    print(
        f"- Small k (k={int(small['k'])}): bias^2={small['bias2']:.6f}, variance={small['variance']:.6f}"
    )
    print(
        f"- Large k (k={int(large['k'])}): bias^2={large['bias2']:.6f}, variance={large['variance']:.6f}"
    )
    print(
        "\nInterpretation (auto):\n"
        "• As k increases, variance usually decreases (more averaging).\n"
        "• As k increases, bias^2 usually increases (more smoothing / loss of local structure).\n"
        "• The optimal k balances both.\n"
    )
    print("====================================================\n")



def add_dimension_normalizations(df_dim: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """
    Adds:
      - noise_var (sigma^2)
      - min_mse_true_mean_norm_noise = min_mse_true_mean / noise_var
      - min_mse_true_mean_rel_d1 = min_mse_true_mean / min_mse_true_mean(d=1)

    Assumes the sweep uses homoskedastic noise with fixed sigma.
    """
    df = df_dim.copy()
    noise_var = float(sigma ** 2)
    df["noise_var"] = noise_var
    df["min_mse_true_mean_norm_noise"] = df["min_mse_true_mean"] / noise_var

    # relative to d=1
    if (df["d"] == 1).any():
        d1_val = float(df.loc[df["d"] == 1, "min_mse_true_mean"].iloc[0])
        df["min_mse_true_mean_rel_d1"] = df["min_mse_true_mean"] / d1_val
    else:
        df["min_mse_true_mean_rel_d1"] = np.nan

    # rounding for nicer tables
    df["min_mse_true_mean_norm_noise"] = df["min_mse_true_mean_norm_noise"].round(6)
    df["min_mse_true_mean_rel_d1"] = df["min_mse_true_mean_rel_d1"].round(6)
    return df


def fit_loglog_slope(x, y, *, eps=1e-12):
    """
    Fits y ≈ C * x^a  by linear regression in log-log space:
      log(y) = a*log(x) + b
    Returns (a, C, r2), where C = exp(b) in natural log space.
    Uses log10 for numerical stability.

    eps avoids log(0) if y is extremely small (shouldn't happen for MSE).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Guard against non-positive values (log undefined)
    y = np.maximum(y, eps)
    x = np.maximum(x, eps)

    lx = np.log10(x)
    ly = np.log10(y)

    # linear fit: ly = a*lx + b
    a, b = np.polyfit(lx, ly, 1)

    # R^2 in log space (how well a straight line explains log-log trend)
    ly_hat = a * lx + b
    ss_res = np.sum((ly - ly_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Convert intercept back: y ≈ 10^b * x^a
    C = 10 ** b
    return float(a), float(C), float(r2)


def plot_dimension_loglog(df_dim: pd.DataFrame, title_prefix: str):
    """
    Makes log-log plots + fits slopes:
      - min MSE vs d (log-log)
      - normalized MSE vs d (log-log), both normalizations
    Prints fitted slope alpha and R^2 to terminal.
    """
    d = df_dim["d"].to_numpy()

    # -------- 1) raw MSE log-log --------
    y_raw = df_dim["min_mse_true_mean"].to_numpy()
    alpha_raw, C_raw, r2_raw = fit_loglog_slope(d, y_raw)

    plt.figure()
    plt.plot(d, y_raw, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dimension d (log)")
    plt.ylabel("min MSE vs true (log)")
    plt.title(f"{title_prefix}: min MSE vs d (log-log)")
    plt.tight_layout()
    out1 = PLOTS_DIR / f"{title_prefix}_minmse_loglog.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    print(f"\n[log-log fit] {title_prefix} raw MSE:")
    print(f"  min_mse ≈ {C_raw:.6g} * d^{alpha_raw:.3f}   (R^2 in log space: {r2_raw:.3f})")
    print(f"  Saved: {out1}")

    # -------- 2) normalized-by-noise log-log --------
    if "min_mse_true_mean_norm_noise" in df_dim.columns:
        y_norm_noise = df_dim["min_mse_true_mean_norm_noise"].to_numpy()
        alpha_nn, C_nn, r2_nn = fit_loglog_slope(d, y_norm_noise)

        plt.figure()
        plt.plot(d, y_norm_noise, marker="o")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dimension d (log)")
        plt.ylabel("min MSE / noise_var (log)")
        plt.title(f"{title_prefix}: normalized MSE vs d (log-log)")
        plt.tight_layout()
        out2 = PLOTS_DIR / f"{title_prefix}_minmse_norm_noise_loglog.png"
        plt.savefig(out2, dpi=200)
        plt.close()

        print(f"[log-log fit] {title_prefix} MSE normalized by noise variance:")
        print(f"  (min_mse/noise) ≈ {C_nn:.6g} * d^{alpha_nn:.3f}   (R^2: {r2_nn:.3f})")
        print(f"  Saved: {out2}")
    else:
        out2 = None

    # -------- 3) relative-to-d1 log-log --------
    if "min_mse_true_mean_rel_d1" in df_dim.columns:
        y_rel = df_dim["min_mse_true_mean_rel_d1"].to_numpy()
        alpha_rel, C_rel, r2_rel = fit_loglog_slope(d, y_rel)

        plt.figure()
        plt.plot(d, y_rel, marker="o")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dimension d (log)")
        plt.ylabel("min MSE / min MSE(d=1) (log)")
        plt.title(f"{title_prefix}: relative MSE vs d (log-log)")
        plt.tight_layout()
        out3 = PLOTS_DIR / f"{title_prefix}_minmse_rel_d1_loglog.png"
        plt.savefig(out3, dpi=200)
        plt.close()

        print(f"[log-log fit] {title_prefix} MSE relative to d=1:")
        print(f"  (min_mse/min_mse_d1) ≈ {C_rel:.6g} * d^{alpha_rel:.3f}   (R^2: {r2_rel:.3f})")
        print(f"  Saved: {out3}")
    else:
        out3 = None

    print(f"Saved: {out1}")
    if out2 is not None:
        print(f"Saved: {out2}")
    if out3 is not None:
        print(f"Saved: {out3}")


def main():
    # shared controls
    seeds = list(range(1, 31))
    k_values = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200]

    # -------------------------------------------------
    # A) DIMENSION SWEEP (recommend: simple mean function)
    # -------------------------------------------------
    d_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]

    # Keep it simple: sum_linear is dimension-friendly.
    base_dim_cfg = DatasetConfig(
        d=1,  # overwritten
        mean_fn=MEAN_FN_TYPE.SUM_LINEAR,
        noise_type=NOISE_FN_TYPE.HOMO,
        sigma=0.2,
        seed=0,  # overwritten
    )

    df_dim = run_dimension_sweep(
        sweep_name="sum_linear_homo",
        base_cfg=base_dim_cfg,
        d_values=d_values,
        seeds=seeds,
        k_values=k_values,
        n_jobs=-1,
    )
    df_dim = add_dimension_normalizations(df_dim, sigma=base_dim_cfg.sigma)

    RESULTS_DIR.mkdir(exist_ok=True)
    df_dim.to_csv(RESULTS_DIR / "dimension_summary_sum_linear_homo.csv", index=False)
    print_terminal_summary_dimension(df_dim, "Sweep: sum_linear + homoskedastic noise")
    plot_dimension_summary(df_dim, "sum_linear_homo")
    plot_dimension_loglog(df_dim, "sum_linear_homo")

    # Fit slopes and save a tiny slope table too
    alpha_raw, C_raw, r2_raw = fit_loglog_slope(df_dim["d"], df_dim["min_mse_true_mean"])

    alpha_nn = C_nn = r2_nn = np.nan
    if "min_mse_true_mean_norm_noise" in df_dim.columns:
        alpha_nn, C_nn, r2_nn = fit_loglog_slope(df_dim["d"], df_dim["min_mse_true_mean_norm_noise"])

    alpha_rel = C_rel = r2_rel = np.nan
    if "min_mse_true_mean_rel_d1" in df_dim.columns:
        alpha_rel, C_rel, r2_rel = fit_loglog_slope(df_dim["d"], df_dim["min_mse_true_mean_rel_d1"])

    df_slopes = pd.DataFrame(
        [
            {"metric": "min_mse_true_mean", "alpha": alpha_raw, "C": C_raw, "r2_logspace": r2_raw},
            {"metric": "min_mse_true_mean_norm_noise", "alpha": alpha_nn, "C": C_nn, "r2_logspace": r2_nn},
            {"metric": "min_mse_true_mean_rel_d1", "alpha": alpha_rel, "C": C_rel, "r2_logspace": r2_rel},
        ]
    )

    df_slopes.to_csv(RESULTS_DIR / "dimension_loglog_slopes.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'dimension_loglog_slopes.csv'}")

    # -------------------------------------------------
    # B) BIAS–VARIANCE (pick a few representative datasets)
    # -------------------------------------------------
    bv_experiments = {
        "sine_1d_homo": DatasetConfig(
            d=1, mean_fn=MEAN_FN_TYPE.SINE_1D, noise_type=NOISE_FN_TYPE.HOMO, sigma=0.2, seed=0
        ),
        "piecewise_1d_hetero_linear": DatasetConfig(
            d=1, mean_fn=MEAN_FN_TYPE.PIECEWISE_1D, noise_type=NOISE_FN_TYPE.HETERO_LINEAR,
            sigma_min=0.05, sigma_max=0.5, seed=0
        ),
        "product_sine_5d_homo": DatasetConfig(
            d=5, mean_fn=MEAN_FN_TYPE.PRODUCT_SINE, noise_type=NOISE_FN_TYPE.HOMO, sigma=0.15, seed=0
        ),
        "linear_sum_10d_hetero_radial": DatasetConfig(
            d=10, mean_fn=MEAN_FN_TYPE.SUM_LINEAR, noise_type=NOISE_FN_TYPE.HETERO_RADIAL,
            sigma_min=0.05, sigma_max=0.6, seed=0
        ),
    }

    for name, cfg in bv_experiments.items():
        df_bv = run_bias_variance_decomposition(
            experiment_name=name,
            base_cfg=cfg,
            seeds=seeds,
            k_values=k_values,
            fixed_test_seed=9999,
            n_jobs=-1,
        )
        df_bv.to_csv(RESULTS_DIR / f"bias_variance_{name}.csv", index=False)
        print_terminal_summary_bias_variance(df_bv, name)
        plot_bias_variance(df_bv, name)


if __name__ == "__main__":
    main()