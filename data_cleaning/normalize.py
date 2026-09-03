"""
Key design: normalization stats are computed INCREMENTALLY via Welford's
online algorithm so that we never load more than one year into RAM at a time.

Welford's algorithm maintains a running count, mean, and M2 (sum of squared
deviations) and can be updated with new batches of data without holding the
full history in memory. After all training years are streamed, the final
population mean and std are derived from the accumulators.

Reference: Welford, B.P. (1962). "Note on a Method for Calculating
Corrected Sums of Squares and Products." Technometrics 4 (3): 419–420.
"""
import json
import numpy as np
import xarray as xr
from config import (
    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, OUT_DIR,
)


# ── Incremental normalization stats ───────────────────────────────────

class WelfordAccumulator:
    """Tracks running mean and variance via Welford's online algorithm.
    Processes flat arrays of values (NaN excluded) incrementally.
    """
    def __init__(self):
        self.n   = 0       # total finite samples seen
        self.mean = 0.0
        self.M2   = 0.0    # sum of squared deviations from running mean

    def update(self, values: np.ndarray):
        """Update with a 1-D array of finite float values."""
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        # Batch Welford update
        batch_n    = len(values)
        batch_mean = float(np.mean(values))
        batch_var  = float(np.var(values, ddof=0))

        combined_n    = self.n + batch_n
        delta         = batch_mean - self.mean
        new_mean      = self.mean + delta * batch_n / combined_n
        new_M2        = (self.M2
                         + batch_var * batch_n
                         + delta ** 2 * self.n * batch_n / combined_n)
        self.n    = combined_n
        self.mean = new_mean
        self.M2   = new_M2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        var = self.M2 / self.n   # population variance
        return float(np.sqrt(max(var, 1e-20)))


def init_accumulators(var_names: list) -> dict:
    return {v: WelfordAccumulator() for v in var_names}


def update_accumulators(accumulators: dict, year_ds: xr.Dataset,
                        mask: xr.DataArray):
    """Stream one year's data into the Welford accumulators.
    Only ocean cells (mask == 1) and finite values are included.
    """
    ocean = mask.values == 1  # (81, 101) bool
    for var, acc in accumulators.items():
        data = year_ds[var].values  # (T_year, 81, 101)
        ocean_vals = data[:, ocean].ravel()
        acc.update(ocean_vals)


def accumulators_to_stats(accumulators: dict) -> dict:
    stats = {}
    for var, acc in accumulators.items():
        std = acc.std
        if std < 1e-10:
            std = 1.0
        stats[var] = {"mean": acc.mean, "std": std}
        print(f"     {var:>16s}  μ={acc.mean:+11.5f}  σ={std:10.5f}  "
              f"n={acc.n:,}")
    return stats


def save_norm_stats(stats: dict):
    path = OUT_DIR / "normalization_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"     Stats saved → {path}")


def load_norm_stats() -> dict:
    path = OUT_DIR / "normalization_stats.json"
    with open(path) as f:
        return json.load(f)


# ── apply: Z-score ─────────────────────────────────────────────────────

def apply_zscore_inplace(ds: xr.Dataset, stats: dict) -> xr.Dataset:
    """Apply Z-score in-place to avoid doubling RAM usage."""
    for var in ds.data_vars:
        if var not in stats:
            continue
        mu  = stats[var]["mean"]
        std = stats[var]["std"]
        ds[var].values -= mu
        ds[var].values /= std
    return ds


# ── Zero-fill inputs ─────────────────────────────────────────────────

def zero_fill_inputs_inplace(ds: xr.Dataset, input_vars: list) -> xr.Dataset:
    """Zero-fill NaN in input channels in-place (post-normalization).
    0.0 = climatological mean in Z-score space.
    Target NaN stays NaN for masked loss computation.
    """
    for var in input_vars:
        if var not in ds.data_vars:
            continue
        arr   = ds[var].values
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            np.nan_to_num(arr, nan=0.0, copy=False)
            print(f"     {var}: {n_nan} NaN → 0.0")
    return ds
