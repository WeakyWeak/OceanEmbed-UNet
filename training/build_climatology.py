#!/usr/bin/env python3

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import splits as S


def _circular_smooth(clim: np.ndarray, half: int) -> np.ndarray:
    """Centred circular moving average over the day-of-year axis (1..366)."""
    if half <= 0:
        return clim
    body = clim[1:]  # (366, ...)
    n = body.shape[0]
    acc = np.zeros_like(body, dtype=np.float64)
    for off in range(-half, half + 1):
        acc += np.roll(body, off, axis=0)
    body = (acc / (2 * half + 1)).astype(np.float32)
    out = np.zeros_like(clim)
    out[1:] = body
    return out


def _doy_mean(arr, rows, doy, n_ch, half):
    """Per-cell per-day-of-year mean over `rows`, then circular smoothing."""
    clim = np.zeros((367, C.N_LAT, C.N_LON, n_ch), dtype=np.float32)
    counts = np.zeros(367, dtype=np.int32)
    for d in range(1, 367):
        r = rows[doy[rows] == d]
        counts[d] = len(r)
        if len(r):
            clim[d] = np.asarray(arr[r], dtype=np.float32).mean(axis=0)
    thin = [d for d in range(1, 367) if counts[d] < 5]
    if thin:
        print(
            f"      day-of-year values backed by <5 train years: {thin} "
            f"(smoothing fills these from neighbours)"
        )
    return _circular_smooth(clim, half)


def build_running_means(force: bool = False) -> None:
    """Write rm{W}.npy for every window in C.RUNNING_MEAN_WINDOWS.

    Each array is (N_DAYS, N_LAT, N_LON, N_INPUT_VARS) float16 holding the
    CLIMATOLOGY-SUBTRACTED trailing W-day mean, NOT yet standardised. The
    standardisation constants go into anom_stats.json so the arrays and the
    statistics can be regenerated independently; baking the standardisation
    into a 910 MiB binary would let a later stats rebuild disagree with the
    array silently.

    ORDER MATTERS: climatology is subtracted BEFORE the window average, never
    after. A trailing W-day mean attenuates a 365-day sinusoid by
    sin(wW/2)/(W sin(w/2)) and lags it by (W-1)/2 days: at W=90 that is a
    factor 0.903 and 44.5 days, so clim(t) minus the windowed mean of clim
    leaves a residual of 0.717*A. For sss (92% seasonal, z-space amplitude
    1.36) that residual is 0.98 against an anomaly std of 0.164 - a 5.9x
    contamination. Subtracting afterwards does not give a slightly wrong
    anomaly, it gives a channel that is mostly a phase-shifted season.

    Running means are ALWAYS of the anomaly, whatever ANOMALISE_INPUTS says: a
    running mean of raw z is dominated by the seasonal cycle and is not a
    useful feature under any flag setting. All five flags are True today, so
    this is currently moot - it is written down so flipping one later does not
    silently require rebuilding 1.8 GiB.

    Rejected alternative, so nobody "optimises" it back in: one prefix-sum
    array would serve every window from 910 MiB instead of 910 MiB per window,
    but partial sums reach ~4700 in z units and recovering a difference of
    order 0.2 needs ~8 significant digits. float32 has ~7. It works only in
    float64, at 3.6 GiB, which defeats the point.
    """
    windows = [int(w) for w in C.RUNNING_MEAN_WINDOWS]
    if not windows:
        print("[running_means] RUNNING_MEAN_WINDOWS is empty - nothing to do.")
        return

    todo = [w for w in windows if force or not C.rm_npy(w).exists()]
    if not todo:
        print(f"[running_means] rm{windows} already built - use --force to rebuild.")
    else:
        X = np.load(C.X_NPY, mmap_mode="r")
        clim_X = np.load(C.CLIM_X_NPY, mmap_mode="r")
        doy = pd.DatetimeIndex(np.load(C.DATES_NPY)).dayofyear.values
        n_days = X.shape[0]
        H = max(windows) - 1
        BLOCK = 128

        outs = {
            w: np.lib.format.open_memmap(
                C.rm_npy(w),
                mode="w+",
                dtype=np.float16,
                shape=(n_days, C.N_LAT, C.N_LON, C.N_INPUT_VARS),
            )
            for w in todo
        }

        print(
            f"[running_means] windows={todo}  history={H}  "
            f"{n_days} days in {BLOCK}-day blocks"
        )
        for a in range(0, n_days, BLOCK):
            b = min(a + BLOCK, n_days)
            lo = max(0, a - H)
            # cumsum is reset per block, so rounding error never accumulates
            # across the record; at most 217 terms of magnitude <= 0.8 are
            # summed, giving ~1e-5 absolute error in float32 - three orders
            # below the float16 storage step at these magnitudes.
            raw = np.asarray(X[lo:b], np.float32) - np.asarray(
                clim_X[doy[lo:b]], np.float32
            )
            cs = np.cumsum(raw, axis=0)
            t = np.arange(a, b)
            j = t - lo
            for w in todo:
                hi_c = cs[j]
                k = j - w
                lo_c = np.where(
                    (k >= 0)[:, None, None, None], cs[np.clip(k, 0, None)], 0.0
                )
                m = (hi_c - lo_c) / w
                # Rows with less than w days of history in front of them get
                # EXACT ZERO. They are excluded from every split by
                # splits.py once required_history() covers this window, so
                # nothing reads them - and zero is the post-standardisation
                # channel mean, so a leaked one presents as neutral rather
                # than wild. Note the zero band is PER WINDOW: rm30 rows
                # 0..28, rm90 rows 0..88.
                m[t < w - 1] = 0.0
                outs[w][a:b] = m.astype(np.float16)
            del raw, cs
            for w in todo:
                outs[w].flush()
            gc.collect()
            if (a // BLOCK) % 20 == 0:
                print(f"    {b}/{n_days} days")
        del outs
        gc.collect()
        print(f"[running_means] wrote {[str(C.rm_npy(w).name) for w in todo]}")

    # ── per-window standardisation statistics, train rows only ──────────────
    land = np.load(C.LAND_MASK_NPY)
    ocean = land == 1
    doy = pd.DatetimeIndex(np.load(C.DATES_NPY)).dayofyear.values

    # required_history(), NOT max_lag=0: the warm-up rows are exact zeros and
    # including them would pull the mean toward zero and deflate the std.
    rows = S.split_indices("train", C.required_history())
    held = set(S.split_years("val")) | set(S.split_years("test"))
    leaked = sorted(set(S.record_years()[rows].tolist()) & held)
    if leaked:
        raise RuntimeError(f"training rows include held-out years {leaked}")
    print(
        f"[running_means] statistics over {rows.size} train rows "
        f"(history {C.required_history()})"
    )

    rm_mean, rm_std = {}, {}
    for w in windows:
        arr = np.load(C.rm_npy(w), mmap_mode="r")
        s = np.zeros(C.N_INPUT_VARS, np.float64)
        ss = np.zeros(C.N_INPUT_VARS, np.float64)
        n = np.zeros(C.N_INPUT_VARS, np.float64)
        for b in range(0, rows.size, 512):
            idx = rows[b : b + 512]
            a = np.asarray(arr[idx], dtype=np.float32)
            for c in range(C.N_INPUT_VARS):
                v = a[:, :, :, c][:, ocean]
                s[c] += v.sum(dtype=np.float64)
                ss[c] += np.square(v, dtype=np.float64).sum()
                n[c] += v.size
        mu = s / n
        sd = np.sqrt(np.maximum(ss / n - mu**2, 1e-20))
        rm_mean[w] = mu
        rm_std[w] = sd
        del arr
        gc.collect()

    # Merge, never overwrite: the served models depend on keys already here
    # (target_phys_scale_degC above all), so a windows-only rebuild must not
    # disturb them.
    out = json.load(open(C.ANOM_STATS_JSON))
    out["input_rm_anom_mean"] = {
        str(w): {v: float(rm_mean[w][i]) for i, v in enumerate(C.INPUT_VARS)}
        for w in windows
    }
    out["input_rm_anom_std"] = {
        str(w): {v: float(rm_std[w][i]) for i, v in enumerate(C.INPUT_VARS)}
        for w in windows
    }
    out["running_mean_windows"] = windows
    out["running_mean_warmup_rows"] = {str(w): w - 1 for w in windows}
    out["running_mean_note"] = (
        "rm{W}.npy holds the climatology-subtracted trailing W-day mean, "
        "pre-standardisation, float16. Channel = (stored - mean)/std with the "
        "per-window constants here. Rows below running_mean_warmup_rows are "
        "exact zeros and are excluded from every split."
    )
    json.dump(out, open(C.ANOM_STATS_JSON, "w"), indent=2)

    print("\n  per-window anomaly std (z-space), vs the instantaneous field:")
    base = out["input_anom_std"]
    print(f"    {'window':>7} " + " ".join(f"{v:>8}" for v in C.INPUT_VARS))
    print(f"    {'inst':>7} " + " ".join(f"{base[v]:>8.4f}" for v in C.INPUT_VARS))
    for w in windows:
        print(
            f"    {('rm%d' % w):>7} "
            + " ".join(f"{rm_std[w][i]:>8.4f}" for i, _ in enumerate(C.INPUT_VARS))
        )
        print(
            f"    {'ratio':>7} "
            + " ".join(
                f"{rm_std[w][i] / base[v]:>8.3f}" for i, v in enumerate(C.INPUT_VARS)
            )
        )


def build(force: bool = False) -> None:
    if C.CLIM_Y_NPY.exists() and not force:
        print(f"[build_climatology] {C.CLIM_Y_NPY} exists - use --force to rebuild.")
        # The running means depend on clim_X, which is already here, and may
        # still be missing or stale. Never skip them just because the
        # climatology is current.
        build_running_means(force=False)
        return

    X = np.load(C.X_NPY, mmap_mode="r")
    Y = np.load(C.Y_NPY, mmap_mode="r")
    dates = pd.DatetimeIndex(np.load(C.DATES_NPY))
    doy = dates.dayofyear.values
    land = np.load(C.LAND_MASK_NPY)
    tm = np.load(C.TARGET_MASK_NPY)
    ocean = land == 1
    meta = json.load(open(C.META_JSON))

    # Training rows come from splits.py, which resolves membership by calendar
    # YEAR against the actual date axis. The old `lo, hi = SPLIT_RANGES["train"]`
    # assumed the training days were one contiguous block at the front of the
    # record - true for 2005-2022, and false for any configurable split.
    #
    # This matters more here than almost anywhere else: the climatology and the
    # anomaly stats are the definition of the training target. Computing them
    # over rows that include a held-out year leaks that year into every sample,
    # and nothing downstream would raise.
    rows = S.split_indices("train", max_lag=0)
    yrs = S.split_years("train")
    print(
        f"[build_climatology] split={C.SPLIT_MODE}  "
        f"{rows.size} training rows over {len(yrs)} years "
        f"({dates[rows[0]].date()} .. {dates[rows[-1]].date()})"
    )
    held = set(S.split_years("val")) | set(S.split_years("test"))
    leaked = sorted(set(S.record_years()[rows].tolist()) & held)
    if leaked:
        raise RuntimeError(f"training rows include held-out years {leaked}")

    print("  inputs  ...")
    clim_X = _doy_mean(X, rows, doy, C.N_INPUT_VARS, C.CLIM_SMOOTH_HALF_WINDOW)
    print("  targets ...")
    clim_Y = _doy_mean(Y, rows, doy, C.N_DEPTHS, C.CLIM_SMOOTH_HALF_WINDOW)

    # ── anomaly statistics, streamed over the train split ───────────────────
    print("  anomaly statistics ...")

    def stats(arr, clim, n_ch, mask3d):
        """(mean, std) of arr - clim over train ocean cells, per channel."""
        s = np.zeros(n_ch, np.float64)
        ss = np.zeros(n_ch, np.float64)
        n = np.zeros(n_ch, np.float64)
        # Iterate the training ROW SET, not a contiguous lo..hi range. The
        # range form only worked while the training days were one unbroken
        # block; under a configurable split it would sweep straight through
        # held-out years and fold them into the very statistics that define
        # the model's target.
        for b in range(0, rows.size, 512):
            idx = rows[b : b + 512]
            a = np.asarray(arr[idx], dtype=np.float32) - clim[doy[idx]]
            for c in range(n_ch):
                v = a[:, :, :, c][:, mask3d[:, :, c]]
                s[c] += v.sum(dtype=np.float64)
                ss[c] += np.square(v, dtype=np.float64).sum()
                n[c] += v.size
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean**2, 1e-20))
        return mean, std

    in_mask = np.repeat(ocean[:, :, None], C.N_INPUT_VARS, axis=2)
    xa_mean, xa_std = stats(X, clim_X, C.N_INPUT_VARS, in_mask)
    ya_mean, ya_std = stats(Y, clim_Y, C.N_DEPTHS, tm)

    # degC per unit of normalised anomaly = anomaly_std(z) * sigma_absolute
    sig_abs = np.array([meta["sigmas"][v] for v in C.TARGET_VARS])
    phys_scale = ya_std * sig_abs

    out = {
        "input_anom_mean": {v: float(xa_mean[i]) for i, v in enumerate(C.INPUT_VARS)},
        "input_anom_std": {v: float(xa_std[i]) for i, v in enumerate(C.INPUT_VARS)},
        "target_anom_mean": ya_mean.tolist(),
        "target_anom_std": ya_std.tolist(),
        "target_sigma_abs": sig_abs.tolist(),
        "target_phys_scale_degC": phys_scale.tolist(),
        "clim_smooth_half_window": C.CLIM_SMOOTH_HALF_WINDOW,
        # Record WHICH split these were computed under. A model trained on one
        # split and evaluated against another split's climatology produces a
        # wrong answer that looks entirely plausible, so the provenance has to
        # travel with the artefact.
        "split_mode": C.SPLIT_MODE,
        "train_years": S.split_years("train"),
        "n_train_rows": int(rows.size),
        "clim_source": (
            f"{C.SPLIT_MODE} train split, {rows.size} rows, "
            f"{dates[rows[0]].date()}..{dates[rows[-1]].date()}"
        ),
        "note": (
            "Anomaly a = (z - clim_z); normalised target = (a - mean)/std. "
            "Physical degC = target_phys_scale_degC * normalised anomaly. "
            "Absolute degC = clim_physical + that."
        ),
    }
    np.save(C.CLIM_X_NPY, clim_X)
    np.save(C.CLIM_Y_NPY, clim_Y)
    json.dump(out, open(C.ANOM_STATS_JSON, "w"), indent=2)

    print("\n  per-depth anomaly statistics (train):")
    print(
        f"    {'depth':>7} {'sigma_abs':>10} {'anom_std(z)':>12} {'degC/unit':>10} "
        f"{'seasonal share':>15}"
    )
    for k, lbl in enumerate(C.DEPTH_LABELS):
        share = 100 * (1 - ya_std[k] ** 2)  # z-space total variance is ~1
        print(
            f"    {lbl:>7} {sig_abs[k]:>10.4f} {ya_std[k]:>12.4f} "
            f"{phys_scale[k]:>10.4f} {share:>14.1f}%"
        )

    print("\n  per-input anomaly std (z-space):")
    for v in C.INPUT_VARS:
        flag = "ANOMALISED" if C.ANOMALISE_INPUTS[v] else "raw"
        print(f"    {v:>5} {xa_std[C.INPUT_VARS.index(v)]:>8.4f}   [{flag}]")

    print(
        f"\n[build_climatology] wrote {C.CLIM_X_NPY.name}, {C.CLIM_Y_NPY.name}, "
        f"{C.ANOM_STATS_JSON.name}"
    )

    # Running means read clim_X, so they can only be built after it exists.
    build_running_means(force=force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--force", action="store_true", help="rebuild everything, climatology included"
    )
    ap.add_argument(
        "--running-means-only",
        action="store_true",
        help="rebuild only rm{W}.npy and their statistics, "
        "leaving the climatology untouched",
    )
    args = ap.parse_args()
    if args.running_means_only:
        build_running_means(force=args.force)
    else:
        build(force=args.force)
