"""
Train / val / test membership, defined by CALENDAR YEAR rather than by index.

Why this replaces SPLIT_RANGES
------------------------------
SPLIT_RANGES was a dict of contiguous (start, stop) index tuples, which forces
the held-out period to be one unbroken block at a fixed position. Two things
made that untenable:

1. The record is about to run 1993-2024 instead of 2005-2022, so every index
   in a hardcoded range changes meaning.
2. The domain-mean subsurface anomaly carries a +0.86 degC/decade warming
   trend that explains ~87% of its variance. Whether the held-out years sit at
   the END of the record (the model must EXTRAPOLATE that trend) or are spread
   through it (the model INTERPOLATES) is a real experimental choice, and it
   should be a switch rather than a decision baked into an index tuple.

So membership is expressed as years, and indices are derived from the actual
date array. Nothing here hardcodes a record length.

The leakage trade, stated plainly
---------------------------------
Interleaved holdout is leakier than chronological. A test year sitting between
two training years shares low-frequency ocean state with its neighbours -
ENSO and IOD run 12-18 months, and subsurface heat content has multi-year
memory. BUFFER_DAYS drops training days either side of each held-out year,
which removes the most egregious near-duplicates but cannot remove the shared
low-frequency state; the anomaly e-folding time is 85-307 days by depth, and
buffering that far would consume the record.

Chronological is therefore the default: it is the honest harder test, it is
what a reviewer expects, and expanding the record to 28 training years already
defuses most of the trend-extrapolation penalty that motivated interleaving.

Run both and report both. The gap between them is the measurement of how much
the interleaved number owes to temporal leakage.

A note on lag history, which is NOT a problem
---------------------------------------------
X.npy is one contiguous record and the lag stack reads it by ABSOLUTE index,
independent of split membership. A validation sample on 1 January reads its
t-7 input from late December of a training year. Those are PAST SURFACE
OBSERVATIONS, never targets, so this is legitimate - and it is exactly what
the existing 2021/2022 chronological split already does. The only real
constraint is the global one: a sample needs `max_lag` days of record in front
of it, which is why every accessor takes max_lag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C

TRAIN, VAL, TEST, BUFFER = 0, 1, 2, 3
CODES = {"train": TRAIN, "val": VAL, "test": TEST, "buffer": BUFFER}
NAMES = {v: k for k, v in CODES.items()}

_CACHE: dict = {}


def _config(mode: str | None = None) -> dict:
    mode = mode or C.SPLIT_MODE
    if mode not in C.SPLIT_CONFIGS:
        raise KeyError(f"unknown split mode {mode!r}; "
                       f"have {sorted(C.SPLIT_CONFIGS)}")
    return C.SPLIT_CONFIGS[mode]


def record_years() -> np.ndarray:
    """Calendar year of every day in the record, read from the date array.

    Deliberately derived rather than declared: this is what makes the module
    work unchanged whether the record is 6574 days or 11672.
    """
    if "years" not in _CACHE:
        d = pd.DatetimeIndex(np.load(C.DATES_NPY))
        _CACHE["years"] = d.year.values.astype(np.int32)
        _CACHE["dates"] = d
    return _CACHE["years"]


def n_days() -> int:
    return len(record_years())


def day_labels(mode: str | None = None) -> np.ndarray:
    """(n_days,) int8 of TRAIN/VAL/TEST/BUFFER for every day in the record."""
    key = ("labels", mode or C.SPLIT_MODE)
    if key in _CACHE:
        return _CACHE[key]

    cfg = _config(mode)
    years = record_years()
    lab = np.full(years.size, TRAIN, dtype=np.int8)

    val_years = set(cfg["val_years"])
    test_years = set(cfg["test_years"])
    overlap = val_years & test_years
    if overlap:
        raise ValueError(f"years {sorted(overlap)} are in both val and test")

    in_val = np.isin(years, list(val_years))
    in_test = np.isin(years, list(test_years))
    lab[in_val] = VAL
    lab[in_test] = TEST

    # Buffer: training days within BUFFER_DAYS of any held-out day are dropped
    # from training entirely. They are not reassigned - they belong to no
    # split, which is why index_to_split has to answer "buffer".
    buf = int(cfg.get("buffer_days", 0))
    # A buffer exists to stop a training day from reading surface state that
    # belongs to a held-out year. It was sized against a 7-day lag reach; a
    # trailing W-day mean reaches W-1 days back, so a buffer narrower than
    # that no longer does what it claims. Raise rather than silently widening
    # it - the right buffer is a modelling decision, not a repair.
    #
    # The default 'chronological' mode has buffer_days=0 and puts the held-out
    # years at the END of the record, so it has no interior boundaries and is
    # unaffected. This guard is for 'interleaved'.
    need = C.required_history()
    if 0 < buf < need:
        raise ValueError(
            f"buffer_days={buf} is smaller than the {need} days of history a "
            f"sample needs (lags {C.LAGS}, running means "
            f"{C.RUNNING_MEAN_WINDOWS or 'none'}). A training day adjacent to "
            f"the buffer would average surface state from a held-out year. "
            f"Widen buffer_days to at least {need} or set RUNNING_MEAN_WINDOWS "
            f"back to a reach the buffer covers.")
    if buf > 0:
        held = in_val | in_test
        near = np.zeros_like(held)
        idx = np.flatnonzero(held)
        if idx.size:
            for s in (-buf, buf):
                shifted = np.clip(idx + s, 0, held.size - 1)
                near[shifted] = True
            # fill the interior of each shifted window
            near = np.convolve(held.astype(np.int32),
                               np.ones(2 * buf + 1, np.int32),
                               mode="same") > 0
        lab[near & ~held] = BUFFER

    _CACHE[key] = lab
    return lab


def split_indices(name: str, max_lag: int = 0,
                  mode: str | None = None) -> np.ndarray:
    """Sorted target indices belonging to `name`, usable given `max_lag`.

    `name` may also be "any", which returns every servable day regardless of
    membership - used for serving arbitrary dates, including buffer days.
    """
    labels = day_labels(mode)
    if name == "any":
        sel = np.ones(labels.size, dtype=bool)
    else:
        if name not in CODES:
            raise KeyError(f"unknown split {name!r}; "
                           f"use one of {sorted(CODES)} or 'any'")
        sel = labels == CODES[name]
    sel[:max_lag] = False          # no lag history available at the very start
    return np.flatnonzero(sel).astype(np.int64)


def index_to_split(i: int, mode: str | None = None) -> str:
    return NAMES[int(day_labels(mode)[int(i)])]


def split_years(name: str, mode: str | None = None) -> list[int]:
    if name == "train":
        cfg = _config(mode)
        held = set(cfg["val_years"]) | set(cfg["test_years"])
        return sorted(set(record_years().tolist()) - held)
    return sorted(_config(mode).get(f"{name}_years", []))


def split_blocks(name: str, max_lag: int = 0,
                 mode: str | None = None) -> list[tuple[int, int]]:
    """Contiguous (start, stop) runs of a split. For display only - never use
    these as a membership test; that is the bug this module exists to remove."""
    idx = split_indices(name, max_lag, mode)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]] + 1
    return list(zip(starts.tolist(), stops.tolist()))


def describe(mode: str | None = None) -> str:
    mode = mode or C.SPLIT_MODE
    d = _CACHE.get("dates")
    if d is None:
        record_years()
        d = _CACHE["dates"]
    out = [f"split mode: {mode}   record {d[0].date()} .. {d[-1].date()} "
           f"({n_days()} days)"]
    for name in ("train", "val", "test", "buffer"):
        idx = split_indices(name, max_lag=C.required_history(), mode=mode)
        yrs = split_years(name, mode) if name != "buffer" else []
        span = (f"{d[idx[0]].date()}..{d[idx[-1]].date()}" if idx.size else "-")
        extra = ""
        if name in ("val", "test") and yrs:
            extra = f"  years={yrs}"
        elif name == "train" and yrs:
            extra = f"  {len(yrs)} years"
        out.append(f"  {name:7s} {idx.size:6d} days  {span}{extra}")
    return "\n".join(out)


if __name__ == "__main__":
    for m in sorted(C.SPLIT_CONFIGS):
        print(describe(m), "\n")
