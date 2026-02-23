from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict
import numpy as np


class MEAN_FN_TYPE(str, Enum):
    SINE_1D = "sine_1d"
    SUM_LINEAR = "sum_linear"
    PRODUCT_SINE = "product_sine"
    PIECEWISE_1D = "piecewise_1d"


class NOISE_FN_TYPE(str, Enum):
    HOMO = "homo"
    HETERO_LINEAR = "hetero_linear"
    HETERO_RADIAL = "hetero_radial"


# -----------------------------
# Mean functions
# -----------------------------
def mean_sine_1d(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    return np.sin(2 * np.pi * x)


def mean_sum_linear(X: np.ndarray) -> np.ndarray:
    return np.sum(X, axis=1)


def mean_product_sine(X: np.ndarray) -> np.ndarray:
    return np.prod(np.sin(np.pi * X), axis=1)


def mean_piecewise_1d(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    return np.where(x < 0.5, 0.0, 1.0)


MEAN_FUNCTIONS: Dict[MEAN_FN_TYPE, Callable[[np.ndarray], np.ndarray]] = {
    MEAN_FN_TYPE.SINE_1D: mean_sine_1d,
    MEAN_FN_TYPE.SUM_LINEAR: mean_sum_linear,
    MEAN_FN_TYPE.PRODUCT_SINE: mean_product_sine,
    MEAN_FN_TYPE.PIECEWISE_1D: mean_piecewise_1d,
}


# -----------------------------
# Noise (sigma) functions
# -----------------------------
def sigma_constant(X: np.ndarray, sigma: float) -> np.ndarray:
    return np.full(X.shape[0], float(sigma))


def sigma_linear_in_first_dim(X: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
    x0 = X[:, 0]
    return sigma_min + (sigma_max - sigma_min) * x0


def sigma_radial(X: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
    center = 0.5
    r = np.linalg.norm(X - center, axis=1)
    r_norm = r / (np.sqrt(X.shape[1]) * 0.5)  # ~normalize to [0,1]
    r_norm = np.clip(r_norm, 0.0, 1.0)
    return sigma_min + (sigma_max - sigma_min) * r_norm


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class DatasetConfig:
    # sizes
    n_train: int = 500
    n_test: int = 2000
    d: int = 1

    mean_fn: MEAN_FN_TYPE = MEAN_FN_TYPE.SINE_1D
    noise_type: NOISE_FN_TYPE = NOISE_FN_TYPE.HOMO

    sigma: float = 0.2         # for homo
    sigma_min: float = 0.05    # for hetero
    sigma_max: float = 0.5     # for hetero

    x_dist: str = "uniform01"

    seed: int = 1


def generate_dataset(cfg: DatasetConfig) -> Dict[str, np.ndarray]:
    """
    Generates a synthetic regression dataset:
      X ~ Uniform([0,1]^d)
      y = f(X) + eps, eps ~ N(0, sigma(X)^2)

    Returns dict with arrays:
      X_train, y_train
      X_test, y_test
      y_mean_train, y_mean_test
      sigma_train, sigma_test
    """
    rng = np.random.default_rng(cfg.seed)

    if cfg.mean_fn in (MEAN_FN_TYPE.SINE_1D, MEAN_FN_TYPE.PIECEWISE_1D) and cfg.d != 1:
        raise ValueError(f"{cfg.mean_fn.value} requires d=1, but got d={cfg.d}")

    if cfg.x_dist == "uniform01":
        X_train = rng.uniform(0.0, 1.0, size=(cfg.n_train, cfg.d))
        X_test = rng.uniform(0.0, 1.0, size=(cfg.n_test, cfg.d))
    else:
        raise ValueError(f"Unknown x_dist: {cfg.x_dist}")

    f = MEAN_FUNCTIONS[cfg.mean_fn]
    y_mean_train = f(X_train)
    y_mean_test = f(X_test)

    if cfg.noise_type == NOISE_FN_TYPE.HOMO:
        sigma_train = sigma_constant(X_train, cfg.sigma)
        sigma_test = sigma_constant(X_test, cfg.sigma)
    elif cfg.noise_type == NOISE_FN_TYPE.HETERO_LINEAR:
        sigma_train = sigma_linear_in_first_dim(X_train, cfg.sigma_min, cfg.sigma_max)
        sigma_test = sigma_linear_in_first_dim(X_test, cfg.sigma_min, cfg.sigma_max)
    elif cfg.noise_type == NOISE_FN_TYPE.HETERO_RADIAL:
        sigma_train = sigma_radial(X_train, cfg.sigma_min, cfg.sigma_max)
        sigma_test = sigma_radial(X_test, cfg.sigma_min, cfg.sigma_max)
    else:
        raise ValueError(f"Unknown noise_type: {cfg.noise_type}")

    eps_train = rng.normal(0.0, sigma_train, size=cfg.n_train)
    eps_test = rng.normal(0.0, sigma_test, size=cfg.n_test)

    y_train = y_mean_train + eps_train
    y_test = y_mean_test + eps_test

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_mean_train": y_mean_train,
        "y_mean_test": y_mean_test,
        "sigma_train": sigma_train,
        "sigma_test": sigma_test,
    }