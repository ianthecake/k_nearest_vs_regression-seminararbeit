from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data_generation import DatasetConfig, MEAN_FN_TYPE, NOISE_FN_TYPE, generate_dataset
from src.knn_evaluation import evaluate_knn_train_test


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    k_values = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200]

    # Simple illustrative case
    cfg = DatasetConfig(
        d=1,
        mean_fn=MEAN_FN_TYPE.SINE_1D,
        noise_type=NOISE_FN_TYPE.HOMO,
        sigma=0.2,
        seed=1,
    )

    data = generate_dataset(cfg)
    res = evaluate_knn_train_test(data, k_values, n_jobs=-1)

    k = np.array(res["k"])

    plt.figure()
    plt.plot(k, res["mse_train"], marker="o", label="train MSE (noisy)")
    plt.plot(k, res["mse_test_noisy"], marker="o", label="test MSE (noisy)")
    plt.plot(k, res["mse_test_true"], marker="o", label="test MSE (true mean)")

    plt.xscale("log")
    plt.xlabel("k (log)")
    plt.ylabel("MSE")
    plt.title("Overfitting diagnostic (1D sine, homoskedastic)")
    plt.legend()
    plt.tight_layout()

    out = PLOTS_DIR / "overfitting_sine_1d.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")

    # tiny terminal summary
    print("\nOverfitting check:")
    print(f"- k=1 train MSE ≈ {res['mse_train'][0]:.4f}")
    print(f"- k=1 test MSE vs true ≈ {res['mse_test_true'][0]:.4f}")
    print("Small k fits noise (train error very small) but generalizes poorly.")
    print("Larger k increases bias but improves test error until optimum.\n")


if __name__ == "__main__":
    main()