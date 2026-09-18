"""
normalize.py — z-score normalization, fit on the train split only.

Why this file exists: fitting mean/std on train+val+test combined lets
val/test statistics leak into the numbers used to train the model —
even though the model never directly sees val/test labels, it's
indirectly shaped by them. The fix is mechanical: compute stats from
TRAIN ONLY, then apply those exact same numbers everywhere else —
val, test, and later, any brand-new variant at inference time.

compute_zscore_stats() is the only function allowed to look at data
and produce numbers. Every other function in this file just applies
numbers it's given — that split is what makes leakage impossible by
construction, not just by convention.
"""

import numpy as np


def compute_zscore_stats(train_signals: np.ndarray) -> dict:
    """Compute per-channel mean/std from the TRAIN split ONLY.

    train_signals shape: (n_train_variants, n_channels) — match this to
    whatever shape your actual signals_*.npz arrays use (e.g. after
    max-pooling, before flattening into the 2,048-dim model input).

    Never call this on val, test, or combined data — that's the exact
    leakage bug being fixed.
    """
    mean = train_signals.mean(axis=0)
    std = train_signals.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero on constant channels
    return {"mean": mean, "std": std}


def apply_zscore(signals: np.ndarray, stats: dict) -> np.ndarray:
    """Apply PRE-COMPUTED train stats to any signal array.

    This function does not compute anything — it only applies numbers
    it's handed. Use it for val, test, AND for a single new variant's
    signal at inference time (that's Step 2 in tools.py's
    predict_pathogenicity).
    """
    return (signals - stats["mean"]) / stats["std"]


def save_stats(stats: dict, path: str = "zscore_stats.npy") -> None:
    np.save(path, stats, allow_pickle=True)


def load_stats(path: str = "zscore_stats.npy") -> dict:
    return np.load(path, allow_pickle=True).item()


if __name__ == "__main__":
    # Minimal usage example — replace the random arrays with your real
    # train/val signal arrays (e.g. loaded from signals_liver.npz).
    rng = np.random.default_rng(42)
    train_signals = rng.normal(loc=5, scale=2, size=(100, 2))
    val_signals = rng.normal(loc=5, scale=2, size=(20, 2))

    stats = compute_zscore_stats(train_signals)  # fit: TRAIN ONLY
    save_stats(stats)

    train_norm = apply_zscore(train_signals, stats)  # apply to train
    val_norm = apply_zscore(val_signals, stats)      # apply SAME stats to val

    print("train_norm mean/std (should be ~0 / ~1):", train_norm.mean(axis=0), train_norm.std(axis=0))
    print("val_norm mean (should NOT be ~0 — proves no leakage):", val_norm.mean(axis=0))
