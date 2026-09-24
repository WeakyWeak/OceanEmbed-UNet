#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import splits as S

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(
        f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else "")
    )
    return cond


print("=" * 78)
print("  test_splits.py")
print("=" * 78)

# Days of history a sample needs: lag reach or running-mean reach, whichever
# is larger. With no running means this is max(C.LAGS), as it always was.
max_lag = C.required_history()
dates = pd.DatetimeIndex(np.load(C.DATES_NPY))

# ── 1. legacy equivalence: the whole point ───────────────────────────────────
print("\n1. legacy mode reproduces SPLIT_RANGES exactly")
if hasattr(C, "SPLIT_RANGES"):
    for name, (lo, hi) in C.SPLIT_RANGES.items():
        want = np.arange(max(lo, max_lag), hi, dtype=np.int64)
        got = S.split_indices(name, max_lag, mode="legacy")
        check(
            f"{name}: identical index set",
            np.array_equal(want, got),
            f"n={got.size} (old {want.size})",
        )
else:
    print("  SPLIT_RANGES already removed - migration complete, skipping")

# ── 2. disjointness and coverage ─────────────────────────────────────────────
# A mode whose buffer is narrower than the history a sample needs is UNSOUND,
# not merely untested: a training day beside the buffer would average surface
# state out of a held-out year. splits.py refuses such a mode outright, which
# is the correct behaviour - assert the refusal here, once, then exclude the
# mode from the loops below rather than routing around the guard.
# 'interleaved' (buffer_days=45) lands here whenever RUNNING_MEAN_WINDOWS
# reaches further than 45 days.
print("\n2a. modes whose buffer cannot cover the required history are refused")
_need = C.required_history()
USABLE_MODES = []
for mode in sorted(C.SPLIT_CONFIGS):
    _buf = int(C.SPLIT_CONFIGS[mode].get("buffer_days", 0))
    if 0 < _buf < _need:
        try:
            S.split_indices("train", 0, mode)
        except ValueError:
            check(
                f"{mode}: refused, buffer {_buf}d < history {_need}d",
                True,
                "unsound under the current windows - correctly rejected",
            )
        else:
            check(
                f"{mode}: refused, buffer {_buf}d < history {_need}d",
                False,
                "splits.py ACCEPTED an unsound buffer",
            )
    else:
        USABLE_MODES.append(mode)
if not USABLE_MODES:
    raise SystemExit("no usable split mode under the current configuration")
print(f"        usable modes: {USABLE_MODES}")

print("\n2. partition integrity, every usable mode")
for mode in USABLE_MODES:
    sets = {
        n: set(S.split_indices(n, 0, mode).tolist())
        for n in ("train", "val", "test", "buffer")
    }
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
        ("train", "buffer"),
        ("val", "buffer"),
        ("test", "buffer"),
    ]
    disjoint = all(not (sets[a] & sets[b]) for a, b in pairs)
    total = sum(len(v) for v in sets.values())
    check(f"{mode}: splits are pairwise disjoint", disjoint)
    check(
        f"{mode}: every day belongs to exactly one split",
        total == S.n_days(),
        f"{total} of {S.n_days()}",
    )

# ── 3. no held-out year leaks into training ──────────────────────────────────
print("\n3. no training day falls in a held-out calendar year")
years = S.record_years()
for mode in USABLE_MODES:
    held = set(S.split_years("val", mode)) | set(S.split_years("test", mode))
    tr = S.split_indices("train", 0, mode)
    bad = sorted(set(years[tr].tolist()) & held)
    check(
        f"{mode}: training years exclude {sorted(held)}",
        not bad,
        f"leaked: {bad}" if bad else "",
    )

# ── 4. lag history is available for every sample ─────────────────────────────
print("\n4. every usable sample has max_lag days of record in front of it")
for mode in USABLE_MODES:
    ok = all(
        S.split_indices(n, max_lag, mode).min(initial=max_lag) >= max_lag
        for n in ("train", "val", "test")
    )
    check(f"{mode}: no index below max_lag={max_lag}", ok)

# ── 5. buffers sit between train and held-out, and only there ────────────────
print("\n5. buffer semantics")
for mode in USABLE_MODES:
    cfg = C.SPLIT_CONFIGS[mode]
    buf = S.split_indices("buffer", 0, mode)
    if cfg.get("buffer_days", 0) == 0:
        check(f"{mode}: no buffer days when buffer_days=0", buf.size == 0)
    else:
        held = set(S.split_years("val", mode)) | set(S.split_years("test", mode))
        # A buffer day must be in a TRAINING year (it was taken out of train),
        # and must be within buffer_days of a held-out day.
        in_train_year = not (set(years[buf].tolist()) & held)
        held_idx = np.concatenate(
            [S.split_indices("val", 0, mode), S.split_indices("test", 0, mode)]
        )
        near = np.abs(buf[:, None] - held_idx[None, :]).min(axis=1)
        check(f"{mode}: buffer days lie in training years", in_train_year)
        check(
            f"{mode}: buffer days within {cfg['buffer_days']} of a held-out day",
            bool((near <= cfg["buffer_days"]).all()),
            f"max distance {int(near.max())}",
        )

# ── 6. THE BUG THIS MODULE EXISTS TO PREVENT ─────────────────────────────────
print("\n6. membership is a set test, not a range test")
# dataset_att.py validated indices with `idx.min() >= lo and idx.max() < hi`.
# On a NON-CONTIGUOUS split that accepts anything between the first and last
# day of the split - including training days sitting in the gap between two
# held-out years. Demonstrated with a throwaway two-year config so the test
# does not depend on which years happen to fall inside the current record.
_probe = "___probe___"
C.SPLIT_CONFIGS[_probe] = {
    "val_years": [2010, 2015],
    "test_years": [2012],
    "buffer_days": 0,
}
try:
    val = S.split_indices("val", 0, _probe)
    tr = S.split_indices("train", 0, _probe)
    gap = tr[(tr > val.min()) & (tr < val.max())]
    check(
        "a range test WOULD accept training indices as val",
        gap.size > 0,
        f"{gap.size} training days sit inside the val min/max span "
        f"({val.min()}..{val.max()})",
    )
    check("set membership rejects every one of them", not np.isin(gap, val).any())
finally:
    del C.SPLIT_CONFIGS[_probe]

# ── 7. derived, not hardcoded ────────────────────────────────────────────────
print("\n7. record length is derived from the data, not declared")
check("n_days matches the date array", S.n_days() == len(dates), f"{S.n_days()}")

# Behavioural, not a text grep: point the module at a shorter date array and
# confirm every split follows. This is what guarantees the same code works on
# both the 6574-day record and the 11672-day one.
import tempfile

_short = dates[: len(dates) // 2]
with tempfile.TemporaryDirectory() as td:
    fake = Path(td) / "dates.npy"
    np.save(fake, _short.values)
    real_path, real_cache = C.DATES_NPY, dict(S._CACHE)
    try:
        C.DATES_NPY = fake
        S._CACHE.clear()
        check(
            "n_days follows a different record length",
            S.n_days() == len(_short),
            f"{S.n_days()} == {len(_short)}",
        )
        check(
            "splits still partition the shorter record",
            sum(
                S.split_indices(n, 0, "legacy").size
                for n in ("train", "val", "test", "buffer")
            )
            == len(_short),
        )
    finally:
        C.DATES_NPY = real_path
        S._CACHE.clear()
        S._CACHE.update(real_cache)

print("\n" + "=" * 78)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"    FAILED: {f}")
    sys.exit(1)
print("=" * 78)
