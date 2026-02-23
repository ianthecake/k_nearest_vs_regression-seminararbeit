from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation import DatasetConfig, MEAN_FN_TYPE, NOISE_FN_TYPE, generate_dataset
from src.knn_evaluation import evaluate_knn_train_test


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    curves = np.load(RESULTS_DIR / "curves_true_mean.npz")
    k_values = curves["k_values"]
    experiment_names = [key for key in curves.files if key != "k_values"]
    return summary, curves, k_values, experiment_names


def plot_curves(curves, k_values, experiment_names):
    print("\nGenerating mean±std MSE-vs-k plots...\n")

    for name in experiment_names:
        arr = curves[name]  # shape (n_runs, n_k)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        plt.figure()
        plt.plot(k_values, mean, marker="o", label="mean MSE vs true mean")
        plt.fill_between(k_values, mean - std, mean + std, alpha=0.25, label="±1 std")
        plt.xscale("log")
        plt.xlabel("k (log scale)")
        plt.ylabel("MSE vs true mean")
        plt.title(name)
        plt.legend()
        plt.tight_layout()

        filepath = PLOTS_DIR / f"{name}.png"
        plt.savefig(filepath, dpi=200)
        plt.close()

        print(f"Saved plot: {filepath}")


def print_summary(summary):
    print("\n================ EXPERIMENT SUMMARY ================\n")

    for _, row in summary.iterrows():
        print(f"Dataset: {row['experiment']}")
        print(f"  Runs: {int(row['n_runs'])}")
        print(f"  Best k (mean ± std): {row['best_k_mean']} ± {row['best_k_std']}")
        print(
            f"  Min MSE vs true (mean ± std): "
            f"{row['min_mse_true_mean']} ± {row['min_mse_true_std']}"
        )
        print(
            f"  MSE vs noisy at best k (mean ± std): "
            f"{row['mse_noisy_mean']} ± {row['mse_noisy_std']}"
        )
        print("")

    print("====================================================\n")

    # automatic qualitative interpretation
    print("Automatic qualitative interpretation:\n")

    worst = summary.sort_values("min_mse_true_mean", ascending=False).iloc[0]
    best = summary.sort_values("min_mse_true_mean").iloc[0]

    print(
        f"- Lowest overall error achieved for: {best['experiment']} "
        f"(mean MSE ≈ {best['min_mse_true_mean']})."
    )
    print(
        f"- Highest error observed for: {worst['experiment']} "
        f"(mean MSE ≈ {worst['min_mse_true_mean']})."
    )

    print(
        "\nGeneral pattern observed:\n"
        "• Low-dimensional smooth functions are approximated very well.\n"
        "• Piecewise or heteroskedastic settings make the bias–variance trade-off more delicate.\n"
        "• High-dimensional settings show strong performance degradation (curse of dimensionality).\n"
    )


def plot_overfitting_diagnostic(
    *,
    k_values,
    cfg: DatasetConfig,
    filename: str,
    title: str,
):
    """
    Generates one classic overfitting plot:
      train MSE (noisy) vs k
      test MSE (noisy) vs k
      test MSE (true mean) vs k
    """
    data = generate_dataset(cfg)
    res = evaluate_knn_train_test(data, k_values, n_jobs=-1)

    k = np.array(res["k"])
    mse_train = np.array(res["mse_train"])
    mse_test_noisy = np.array(res["mse_test_noisy"])
    mse_test_true = np.array(res["mse_test_true"])

    plt.figure()
    plt.plot(k, mse_train, marker="o", label="train MSE (noisy)")
    plt.plot(k, mse_test_noisy, marker="o", label="test MSE (noisy)")
    plt.plot(k, mse_test_true, marker="o", label="test MSE (true mean)")

    plt.xscale("log")
    plt.xlabel("k (log scale)")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"\nSaved overfitting diagnostic plot: {out}")

    # tiny terminal summary
    k1_idx = 0  # assumes first k is 1 (true for our defaults)
    print("\nOverfitting diagnostic (quick read):")
    print(f"- Small k={int(k[k1_idx])}: train MSE ≈ {mse_train[k1_idx]:.5f}, test MSE vs true ≈ {mse_test_true[k1_idx]:.5f}")
    best_idx = int(np.argmin(mse_test_true))
    print(f"- Best generalization (vs true mean) at k={int(k[best_idx])}: test MSE vs true ≈ {mse_test_true[best_idx]:.5f}")
    print("Interpretation: small k tends to overfit (low train error, higher test error); larger k reduces variance until bias dominates.\n")


def main():
    summary, curves, k_values_from_file, experiment_names = load_data()

    # --- 1) Mean±std curves from multi-run output ---
    plot_curves(curves, k_values_from_file, experiment_names)

    # --- 2) Summary to terminal ---
    print_summary(summary)

    # --- 3) Overfitting diagnostic (single representative run) ---
    # Use a k grid appropriate for overfitting visualization:
    k_values_overfit = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200]

    # Representative dataset config (same family as your experiments)
    # You can switch to PIECEWISE_1D to show a sharper bias/variance effect.
    cfg_overfit = DatasetConfig(
        d=1,
        mean_fn=MEAN_FN_TYPE.SINE_1D,
        noise_type=NOISE_FN_TYPE.HOMO,
        sigma=0.2,
        seed=1,
    )

    plot_overfitting_diagnostic(
        k_values=k_values_overfit,
        cfg=cfg_overfit,
        filename="overfitting_diagnostic_sine_1d.png",
        title="Overfitting diagnostic: 1D sine (homoskedastic)",
    )


if __name__ == "__main__":
    main()