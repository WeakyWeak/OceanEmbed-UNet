#!/usr/bin/env python3
"""
One-time conversion: data/processed/*.nc  ->  training/arrays/*.npy

Why this exists
---------------
The NetCDF files are chunked [1, 81, 101] per variable. That is fast for
sequential reads but costs ~11 ms of decompressed chunk reads per training
sample across 22 variables - roughly 180 ms per batch of 16, which would dwarf
the GPU work. Training must read from memory.

Building the temporal stack by materialising 21+ channels on disk would cost
4.5 GB for the inputs alone. Instead we store ONE 7-channel array and let
dataset.py construct the stack by indexing at batch time.

Outputs (see spec section 1.2)
------------------------------
  X.npy            (6574, 81, 101,  7) float32   ~1.40 GiB
  Y.npy            (6574, 81, 101, 15) float32   ~3.01 GiB   NaN already -> 0.0
  target_mask.npy  (      81, 101, 15) bool
  pair_mask.npy    (      81, 101, 14) bool
  land_mask.npy    (      81, 101)     float32
  dates.npy        (6574,)             datetime64[D]
  meta.json        mus, sigmas, depths, split indices, variable order

Usage:
    .venv/bin/python training/build_arrays.py [--force]
"""
import argparse
import gc
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import train_config as C


def _open(path):
    """Open a split, reading variables by explicit name only.

    Never iterate ds.data_vars: the files carry a vestigial scalar `number`
    variable (ERA5 ensemble id) that xarray surfaces as a 23rd data variable.
    """
    return crop_to_grid(xr.open_dataset(path))


def crop_to_grid(ds):
    """Cut `ds` down to this version's grid, as a contiguous index slice.

    A no-op for v1 and v2, whose files already are the grid. For v3 it takes
    the lon >= 75 E block out of v2's 81x201 record - the same cells, values
    untouched. A positional slice (not .sel with an array) keeps xarray on
    basic indexing, so the lazy netCDF read stays a plain chunk read. The
    caller's assert_allclose on lat/lon remains the real check that the slice
    landed where intended.
    """
    lat, lon = ds.lat.values, ds.lon.values
    i = int(np.argmin(np.abs(lat - C.TARGET_LAT[0])))
    j = int(np.argmin(np.abs(lon - C.TARGET_LON[0])))
    return ds.isel(lat=slice(i, i + C.N_LAT), lon=slice(j, j + C.N_LON))


def build(force: bool = False) -> None:
    C.ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    if C.X_NPY.exists() and not force:
        print(f"[build_arrays] {C.X_NPY} exists - use --force to rebuild.")
        return

    stats = json.load(open(C.STATS_JSON))
    chunks = C.record_chunks()

    # float16 storage. At v2 size a float32 X+Y is 15.2 GiB against 15 GB of
    # RAM, with random access across the whole record every epoch - the kernel
    # would thrash. float16 halves that to 7.6 GiB, which caches comfortably.
    #
    # Precision is not a concern: every value is z-scored, so |z| < ~6 and
    # float16's relative step is ~5e-4. The tightest case is the training
    # target, formed as (y_abs_z - clim_z) - a difference of two same-magnitude
    # numbers, the classic cancellation risk. Even there the error lands near
    # 7e-4 in z units, which at the deepest level's ~0.25 degC anomaly sd is
    # ~0.0006 degC against a 0.23 degC RMSE. The read path casts back to
    # float32, so only storage is affected.
    STORE = np.float16

    # ── allocate memmapped outputs so peak RAM stays at one variable-slab ────
    X = np.lib.format.open_memmap(
        C.X_NPY, mode="w+", dtype=STORE,
        shape=(C.N_DAYS, C.N_LAT, C.N_LON, C.N_INPUT_VARS))
    Y = np.lib.format.open_memmap(
        C.Y_NPY, mode="w+", dtype=STORE,
        shape=(C.N_DAYS, C.N_LAT, C.N_LON, C.N_DEPTHS))

    all_dates, offset = [], 0
    target_mask = None

    # Streamed in time blocks. The previous version did `ds[v].values`, which
    # materialises ONE WHOLE VARIABLE - at v2 size that is 11672*81*201*4 =
    # 760 MB per variable - and never flushed the memmaps until the very end,
    # so 7.6 GiB of dirty pages accumulated alongside it. On a 10 GB machine
    # that is an OOM kill, which is exactly what happened.
    #
    # Reading BLOCK days at a time caps each read at ~8 MB per variable, and
    # flushing after every block lets the kernel write back and reclaim. The
    # record is chunked [1, 81, 201] on disk, so consecutive days are a
    # sequential read and this costs nothing in speed.
    BLOCK = 128

    for name, path, expect_n in chunks:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run the data_cleaning pipeline first.")
        ds = _open(path)
        n = ds.sizes["time"]
        if expect_n is not None:
            assert n == expect_n, f"{name}: expected {expect_n} days, found {n}"
        print(f"[build_arrays] {name}: {n} days -> rows {offset}..{offset+n-1}")

        np.testing.assert_allclose(ds.lat.values, C.TARGET_LAT, atol=1e-6)
        np.testing.assert_allclose(ds.lon.values, C.TARGET_LON, atol=1e-6)

        if target_mask is None:
            target_mask = np.empty((C.N_LAT, C.N_LON, C.N_DEPTHS), bool)

        for a in range(0, n, BLOCK):
            b = min(a + BLOCK, n)

            for c, v in enumerate(C.INPUT_VARS):
                blk = ds[v].isel(time=slice(a, b)).values
                # Inputs were zero-filled in pipeline phase 4, so a NaN here
                # means the fill did not happen - catch it now rather than
                # letting it reach a gradient.
                if not np.isfinite(blk).all():
                    raise ValueError(
                        f"{v}: non-finite input at rows {offset+a}..{offset+b-1}")
                X[offset + a:offset + b, :, :, c] = blk
                del blk

            for c, v in enumerate(C.TARGET_VARS):
                blk = ds[v].isel(time=slice(a, b)).values
                m = np.isfinite(blk[0])
                if offset == 0 and a == 0:
                    # Validity geography is time-invariant; establish it once.
                    target_mask[:, :, c] = m
                else:
                    # Checked once per block rather than at four sampled
                    # timesteps - strictly more thorough and effectively free.
                    assert np.array_equal(target_mask[:, :, c], m), \
                        f"{v}: validity geography changes at row {offset+a}"
                Y[offset + a:offset + b, :, :, c] = np.nan_to_num(blk, nan=0.0)
                del blk

            X.flush()
            Y.flush()
            gc.collect()

            done = offset + b
            if (a // BLOCK) % 8 == 0 or b == n:
                pct = 100.0 * done / C.N_DAYS
                print(f"  rows {done:6d}/{C.N_DAYS}  ({pct:5.1f}%)",
                      flush=True)

        all_dates.append(pd.DatetimeIndex(ds.time.values))
        offset += n
        ds.close()

    assert offset == C.N_DAYS, f"wrote {offset} rows, expected {C.N_DAYS}"

    # ── date axis: strictly daily and contiguous across split boundaries ─────
    dates = all_dates[0]
    for d in all_dates[1:]:
        dates = dates.append(d)
    gaps = np.unique(np.diff(dates.asi8))
    assert gaps.tolist() == [86_400_000_000_000], f"non-daily time axis: {gaps}"
    print(f"[build_arrays] dates {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)} days, contiguous)")

    # ── derived masks ───────────────────────────────────────────────────────
    pair_mask = target_mask[:, :, :-1] & target_mask[:, :, 1:]
    land = crop_to_grid(xr.open_dataset(C.MASK_NC))["land_mask"].values.astype(np.float32)
    assert land.shape == (C.N_LAT, C.N_LON), f"land mask {land.shape}"

    counts = target_mask.sum(axis=(0, 1)).tolist()
    print("[build_arrays] valid ocean cells per depth:", counts)
    if C.EXPECTED_VALID_CELLS is None:
        print("[build_arrays] EXPECTED_VALID_CELLS is not set for this dataset "
              "version - recording, not asserting.")
        tag = C.DATASET_VERSION.upper()
        print("[build_arrays] paste this into train_config.py as "
              f"EXPECTED_VALID_CELLS_{tag} to lock it in:")
        print(f"    EXPECTED_VALID_CELLS_{tag} = {counts}")
    else:
        assert counts == C.EXPECTED_VALID_CELLS, \
            f"valid-cell counts {counts} != expected {C.EXPECTED_VALID_CELLS}"
        assert int(land.sum()) == C.EXPECTED_VALID_CELLS[0], "land mask mismatch"

    # Integrity was checked per block as the data was written (inputs asserted
    # finite, targets NaN-filled to 0.0 with validity captured in target_mask),
    # so a second full pass over 7.6 GiB here would re-read everything to learn
    # nothing new.
    X.flush(); Y.flush()

    np.save(C.TARGET_MASK_NPY, target_mask)
    np.save(C.PAIR_MASK_NPY, pair_mask)
    np.save(C.LAND_MASK_NPY, land)
    np.save(C.DATES_NPY, dates.values.astype("datetime64[D]"))

    meta = {
        "n_days": C.N_DAYS,
        "input_vars": C.INPUT_VARS,
        "target_vars": C.TARGET_VARS,
        "depth_labels": C.DEPTH_LABELS,
        "actual_depth_m": C.ACTUAL_DEPTH_M,
        "split_ranges": getattr(C, "SPLIT_RANGES", None),
        "dataset_version": C.DATASET_VERSION,
        "storage_dtype": str(np.dtype(STORE)),
        "source_record": [str(p) for _, p, _ in chunks],
        "lon_range": [float(C.TARGET_LON[0]), float(C.TARGET_LON[-1])],
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "mus":    {v: stats[v]["mean"] for v in C.INPUT_VARS + C.TARGET_VARS},
        "sigmas": {v: stats[v]["std"]  for v in C.INPUT_VARS + C.TARGET_VARS},
        "target_valid_cells": counts,
        "note": ("Y has NaN replaced by 0.0; validity lives in target_mask.npy. "
                 "This is what keeps NaN out of the gradients."),
    }
    json.dump(meta, open(C.META_JSON, "w"), indent=2)

    gib = lambda p: p.stat().st_size / 1024 ** 3
    print(f"[build_arrays] done. X={gib(C.X_NPY):.2f} GiB  Y={gib(C.Y_NPY):.2f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if X.npy exists")
    build(force=ap.parse_args().force)
