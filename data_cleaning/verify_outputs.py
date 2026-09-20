#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc4
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import (
    OUT_DIR, YEARS_DIR, YEARS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    TARGET_LAT, TARGET_LON, DEPTH_LABELS,
)

# ── Constants ─────────────────────────────────────────────────────────────────
# The v1 seven-variable input set, ON PURPOSE: this script checks the legacy
# three-file layout, which was built with these. The current record.nc carries
# the five satellite inputs instead - sst, sla, sss, ugos, vgos - so anything
# retargeting this script at record.nc must change this list too.
INPUT_VARS  = ["sst", "sla", "sss", "uo", "vo", "u10", "v10"]
TARGET_VARS = [f"thetao_{lbl}" for lbl in DEPTH_LABELS]
ALL_VARS    = INPUT_VARS + TARGET_VARS

EXPECTED_DAYS = {
    y: 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    for y in YEARS
}
EXPECTED_TRAIN_DAYS = sum(EXPECTED_DAYS[y] for y in TRAIN_YEARS)  # 5844
EXPECTED_VAL_DAYS   = sum(EXPECTED_DAYS[y] for y in VAL_YEARS)    # 365
EXPECTED_TEST_DAYS  = sum(EXPECTED_DAYS[y] for y in TEST_YEARS)    # 365

# Z-score tolerance (sampling, not full-population, so allow generous margin)
ZSCORE_MEAN_TOL = 0.10
ZSCORE_STD_TOL  = 0.10

# ── Failure tracking ──────────────────────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES   = []


def ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS  {msg}")


def fail(msg: str, detail: str = ""):
    global FAIL_COUNT
    FAIL_COUNT += 1
    full = msg + (f"\n        -> {detail}" if detail else "")
    FAILURES.append(full)
    print(f"  FAIL  {msg}")
    if detail:
        print(f"        -> {detail}")


def check(condition: bool, pass_msg: str, fail_msg: str, detail: str = ""):
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg, detail)


def section(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ── A: File inventory ─────────────────────────────────────────────────────────

def check_file_inventory():
    section("A. File Inventory & Sizes")

    required = {
        OUT_DIR / "land_mask.nc":             "land_mask.nc",
        OUT_DIR / "normalization_stats.json": "normalization_stats.json",
        OUT_DIR / "train.nc":                 "train.nc",
        OUT_DIR / "val.nc":                   "val.nc",
        OUT_DIR / "test.nc":                  "test.nc",
    }
    for path, name in required.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 ** 2
            ok(f"{name} exists ({size_mb:.0f} MB)")
        else:
            fail(f"{name} MISSING", str(path))

    for year in YEARS:
        p = YEARS_DIR / f"{year}.nc"
        if p.exists():
            size_mb = p.stat().st_size / 1024 ** 2
            ok(f"years/{year}.nc ({size_mb:.0f} MB)")
        else:
            fail(f"years/{year}.nc MISSING")


# ── B: Land mask ──────────────────────────────────────────────────────────────

def check_land_mask():
    section("B. Land Mask Integrity")
    mask_path = OUT_DIR / "land_mask.nc"
    if not mask_path.exists():
        fail("Skipping land mask -- file missing")
        return

    ds   = xr.open_dataset(mask_path)
    mask = ds["land_mask"]

    check("land_mask" in ds.data_vars,
          "Variable 'land_mask' present",
          "'land_mask' not found in land_mask.nc")

    check(mask.dims == ("lat", "lon"),
          f"Dims correct: {mask.dims}",
          f"Unexpected dims: {mask.dims}", "expected (lat, lon)")

    check(mask.shape == (81, 101),
          f"Shape correct: {mask.shape}",
          f"Shape wrong: {mask.shape}", "expected (81, 101)")

    vals   = mask.values
    unique = np.unique(vals[~np.isnan(vals)])
    check(set(unique).issubset({0, 1}),
          f"Values are binary 0/1 -- unique={unique}",
          f"Non-binary values in mask: {unique}")

    ocean_count = int((vals == 1).sum())
    total       = vals.size
    ocean_pct   = ocean_count / total * 100
    # Logged: 4545 ocean / 8181 total = 55.6%
    check(4400 <= ocean_count <= 4700,
          f"Ocean cell count plausible: {ocean_count}/{total} ({ocean_pct:.1f}%)",
          f"Ocean cell count suspect: {ocean_count} (expected ~4545)",
          "Mask may not have been generated from GLORYS correctly")

    ds.close()


# ── C: Per-year checkpoint files ─────────────────────────────────────────────

def check_year_checkpoints():
    section("C. Per-Year Checkpoint Files")

    for year in YEARS:
        p = YEARS_DIR / f"{year}.nc"
        if not p.exists():
            fail(f"{year}: file missing -- skipping year")
            continue

        ds = xr.open_dataset(p)

        # Variable completeness
        missing_vars = [v for v in ALL_VARS if v not in ds.data_vars]
        check(len(missing_vars) == 0,
              f"{year}: all 22 variables present",
              f"{year}: missing variables",
              str(missing_vars))

        # Time size (leap-year aware)
        n_days   = ds.sizes.get("time", 0)
        expected = EXPECTED_DAYS[year]
        check(n_days == expected,
              f"{year}: time={n_days} days (correct)",
              f"{year}: time={n_days} days (expected {expected})")

        # Spatial dims
        got_lat = ds.sizes.get("lat", 0)
        got_lon = ds.sizes.get("lon", 0)
        check(got_lat == 81 and got_lon == 101,
              f"{year}: spatial dims 81x101 correct",
              f"{year}: spatial dims wrong",
              f"lat={got_lat}, lon={got_lon}")

        # dtype float32
        wrong_dtype = [v for v in ALL_VARS
                       if v in ds.data_vars and ds[v].dtype != np.float32]
        check(len(wrong_dtype) == 0,
              f"{year}: all vars float32",
              f"{year}: dtype not float32 for some vars",
              str({v: str(ds[v].dtype) for v in wrong_dtype}))

        # First and last day
        times          = pd.DatetimeIndex(ds.time.values)
        expected_first = pd.Timestamp(f"{year}-01-01")
        expected_last  = pd.Timestamp(f"{year}-12-31")
        check(times[0] == expected_first,
              f"{year}: first day = {times[0].date()}",
              f"{year}: first day wrong",
              f"got {times[0].date()}, expected {expected_first.date()}")
        check(times[-1] == expected_last,
              f"{year}: last day = {times[-1].date()}",
              f"{year}: last day wrong",
              f"got {times[-1].date()}, expected {expected_last.date()}")

        # No duplicates
        dupes = int(pd.Series(times).duplicated().sum())
        check(dupes == 0,
              f"{year}: no duplicate timestamps",
              f"{year}: {dupes} duplicate timestamps")

        # Monotonic increasing
        check(times.is_monotonic_increasing,
              f"{year}: time is monotonically increasing",
              f"{year}: time is NOT monotonically increasing")

        # Year checkpoints are PRE-normalization; inputs CAN have NaN here.
        # Check thetao_0m has a reasonable number of valid ocean cells.
        thetao_vals = ds["thetao_0m"].isel(time=0).values
        n_ocean     = int(np.isfinite(thetao_vals).sum())
        check(n_ocean > 3000,
              f"{year}: thetao_0m has {n_ocean} finite ocean cells on day 0",
              f"{year}: thetao_0m suspiciously few finite cells: {n_ocean}")

        # Raw SST should be physically plausible for Bay of Bengal
        sst_sample = ds["sst"].isel(time=0).values
        sst_ocean  = sst_sample[np.isfinite(sst_sample)]
        if len(sst_ocean) > 0:
            sst_min = float(sst_ocean.min())
            sst_max = float(sst_ocean.max())
            check(20.0 <= sst_min and sst_max <= 36.0,
                  f"{year}: SST range physical [{sst_min:.2f}, {sst_max:.2f}] C",
                  f"{year}: SST range implausible [{sst_min:.2f}, {sst_max:.2f}] C",
                  "Bay of Bengal SST should be 20-36 C")

        ds.close()


# ── D: Normalization stats ────────────────────────────────────────────────────

def check_norm_stats():
    section("D. Normalization Statistics")
    stats_path = OUT_DIR / "normalization_stats.json"
    if not stats_path.exists():
        fail("normalization_stats.json missing -- skipping")
        return {}

    with open(stats_path) as f:
        stats = json.load(f)

    check(set(stats.keys()) == set(ALL_VARS),
          "All 22 variables have stats entries",
          "Stats keys mismatch",
          f"Missing: {set(ALL_VARS) - set(stats.keys())}")

    # Physical plausibility bounds from domain knowledge
    plausibility = {
        "sst":          ((26.0, 32.0), (0.5, 3.0)),
        "sla":          ((-0.5,  0.5), (0.01, 0.5)),
        "sss":          ((28.0, 36.0), (0.5, 3.0)),
        "thetao_0m":    ((26.0, 32.0), (0.5, 3.0)),
        "thetao_100m":  ((18.0, 28.0), (0.5, 5.0)),
        "thetao_1000m": ((4.0,  10.0), (0.1, 1.0)),
    }
    for var, (mean_range, std_range) in plausibility.items():
        if var not in stats:
            fail(f"{var}: not in stats")
            continue
        mu  = stats[var]["mean"]
        std = stats[var]["std"]
        check(mean_range[0] <= mu <= mean_range[1],
              f"{var}: mean={mu:.4f} in plausible range {mean_range}",
              f"{var}: mean={mu:.4f} OUTSIDE plausible range {mean_range}")
        check(std_range[0] <= std <= std_range[1],
              f"{var}: std={std:.4f} in plausible range {std_range}",
              f"{var}: std={std:.4f} OUTSIDE plausible range {std_range}")

    zero_stds = [v for v in stats if stats[v]["std"] <= 0]
    check(len(zero_stds) == 0,
          "All stds positive",
          "Zero or negative std found",
          str(zero_stds))

    return stats


# ── E: Final split checks ─────────────────────────────────────────────────────

def check_split(name: str, path: Path, expected_years: list,
                expected_days: int, stats: dict):
    print(f"\n  [ {name.upper()} -> {path.name} ]")

    if not path.exists():
        fail(f"{name}.nc MISSING -- skipping all checks")
        return None

    ds = xr.open_dataset(path)

    # E1: Dimensions
    n_time = ds.sizes.get("time", 0)
    check(n_time == expected_days,
          f"time={n_time} (expected {expected_days})",
          f"time={n_time} WRONG (expected {expected_days})")
    check(ds.sizes.get("lat", 0) == 81,
          "lat=81 correct",
          f"lat={ds.sizes.get('lat')} WRONG")
    check(ds.sizes.get("lon", 0) == 101,
          "lon=101 correct",
          f"lon={ds.sizes.get('lon')} WRONG")

    # E2: Variables & dtype
    missing = [v for v in ALL_VARS if v not in ds.data_vars]
    check(len(missing) == 0,
          "All 22 variables present",
          "Missing variables", str(missing))
    wrong_dtype = [v for v in ALL_VARS
                   if v in ds.data_vars and ds[v].dtype != np.float32]
    check(len(wrong_dtype) == 0,
          "All vars float32",
          "dtype not float32",
          str({v: str(ds[v].dtype) for v in wrong_dtype}))

    # E3: Grid coordinate values
    try:
        np.testing.assert_allclose(ds.lat.values, TARGET_LAT, atol=1e-4)
        ok("lat coordinates match TARGET_LAT")
    except AssertionError as e:
        fail("lat coordinates do not match TARGET_LAT", str(e))
    try:
        np.testing.assert_allclose(ds.lon.values, TARGET_LON, atol=1e-4)
        ok("lon coordinates match TARGET_LON")
    except AssertionError as e:
        fail("lon coordinates do not match TARGET_LON", str(e))

    # E4: No NaN in input variables (zero-filled in phase 4)
    for var in INPUT_VARS:
        if var not in ds.data_vars:
            continue
        sample = ds[var].isel(time=0).values
        n_nan  = int(np.isnan(sample).sum())
        check(n_nan == 0,
              f"{var} day-0: no NaN (zero-filled confirmed)",
              f"{var} day-0: {n_nan} NaN found -- zero-fill may have failed")

    # E5: Target variables must retain NaN (land cells masked)
    for var in TARGET_VARS[:3]:
        if var not in ds.data_vars:
            continue
        sample = ds[var].isel(time=0).values
        n_nan  = int(np.isnan(sample).sum())
        check(n_nan > 0,
              f"{var} day-0: {n_nan} NaN present (land mask intact)",
              f"{var} day-0: NO NaN -- land mask may be absent")

    # E6: Z-score sanity -- train only
    # Val/test are normalized with train stats so they won't be exactly N(0,1)
    if name == "train" and stats:
        rng = np.random.default_rng(seed=42)
        for var in INPUT_VARS + ["thetao_0m", "thetao_100m"]:
            if var not in ds.data_vars:
                continue
            t_indices = rng.integers(0, n_time, size=5)
            slices    = [ds[var].isel(time=int(t)).values for t in t_indices]
            all_vals  = np.concatenate([s.ravel() for s in slices])
            ocean     = all_vals[np.isfinite(all_vals)]
            if len(ocean) < 100:
                continue
            z_mean = float(np.mean(ocean))
            z_std  = float(np.std(ocean))
            check(abs(z_mean) < ZSCORE_MEAN_TOL,
                  f"{var}: z-score mean={z_mean:+.3f} (within +/-{ZSCORE_MEAN_TOL})",
                  f"{var}: z-score mean={z_mean:+.3f} -- normalization suspect",
                  f"Expected |mean| < {ZSCORE_MEAN_TOL}")
            check(abs(z_std - 1.0) < ZSCORE_STD_TOL,
                  f"{var}: z-score std={z_std:.3f} (~1.0)",
                  f"{var}: z-score std={z_std:.3f} -- normalization suspect",
                  f"Expected |std - 1.0| < {ZSCORE_STD_TOL}")

    # E7: Time coverage
    times     = pd.DatetimeIndex(ds.time.values)
    exp_first = pd.Timestamp(f"{expected_years[0]}-01-01")
    exp_last  = pd.Timestamp(f"{expected_years[-1]}-12-31")
    check(times[0] == exp_first,
          f"First day: {times[0].date()} correct",
          f"First day: got {times[0].date()}, expected {exp_first.date()}")
    check(times[-1] == exp_last,
          f"Last day: {times[-1].date()} correct",
          f"Last day: got {times[-1].date()}, expected {exp_last.date()}")

    # E8: Chronological continuity -- no gaps, no duplicates
    delta      = np.diff(times.asi8)
    one_day_ns = 86_400 * 1_000_000_000
    gaps       = int((delta != one_day_ns).sum())
    check(gaps == 0,
          "Time axis: no gaps or duplicates across year boundaries",
          f"Time axis: {gaps} gap/duplicate transitions found",
          "Year-boundary append may have introduced a discontinuity")

    return ds


# ── F: Temporal leakage guard ─────────────────────────────────────────────────

def check_temporal_leakage(train_ds, val_ds, test_ds):
    section("F. Temporal Leakage Guard")

    train_times = set(pd.DatetimeIndex(train_ds.time.values).normalize())
    val_times   = set(pd.DatetimeIndex(val_ds.time.values).normalize())
    test_times  = set(pd.DatetimeIndex(test_ds.time.values).normalize())

    tv = train_times & val_times
    tt = train_times & test_times
    vt = val_times   & test_times

    check(len(tv) == 0,
          "No train/val time overlap",
          f"Train/val overlap: {len(tv)} days", str(sorted(tv)[:5]))
    check(len(tt) == 0,
          "No train/test time overlap",
          f"Train/test overlap: {len(tt)} days", str(sorted(tt)[:5]))
    check(len(vt) == 0,
          "No val/test time overlap",
          f"Val/test overlap: {len(vt)} days", str(sorted(vt)[:5]))


# ── G: Cross-split NaN geography ─────────────────────────────────────────────

def check_mask_consistency(train_ds, val_ds, test_ds):
    section("G. Cross-Split NaN Geography Consistency")

    # Input vars are 0-filled: verify no NaN in any split
    for split_name, split_ds in [("train", train_ds),
                                  ("val",   val_ds),
                                  ("test",  test_ds)]:
        n_nan = int(np.isnan(split_ds["sst"].isel(time=0).values).sum())
        check(n_nan == 0,
              f"sst NaN=0 in {split_name} day-0 (zero-fill confirmed)",
              f"sst has {n_nan} NaN in {split_name} day-0")

    # Target vars: NaN geography must be spatially identical across all splits
    # (same canonical land mask applied to all years)
    train_nan = np.isnan(train_ds["thetao_0m"].isel(time=0).values)
    val_nan   = np.isnan(val_ds["thetao_0m"].isel(time=0).values)
    test_nan  = np.isnan(test_ds["thetao_0m"].isel(time=0).values)

    check(np.array_equal(train_nan, val_nan),
          "thetao_0m NaN geography: train == val",
          "thetao_0m NaN geography DIFFERS between train and val",
          "Land mask inconsistency")
    check(np.array_equal(train_nan, test_nan),
          "thetao_0m NaN geography: train == test",
          "thetao_0m NaN geography DIFFERS between train and test",
          "Land mask inconsistency")


# ── H: NetCDF encoding ────────────────────────────────────────────────────────

def check_netcdf_encoding():
    section("H. NetCDF Encoding (compression, dtype, unlimited dim)")

    for fname in ["train.nc", "val.nc", "test.nc"]:
        path = OUT_DIR / fname
        if not path.exists():
            fail(f"{fname}: missing -- skipping encoding check")
            continue
        with nc4.Dataset(path, "r") as f:
            check(f.dimensions["time"].isunlimited(),
                  f"{fname}: time is unlimited dim",
                  f"{fname}: time is NOT unlimited")
            filters = f.variables["sst"].filters()
            check(filters.get("zlib") is True,
                  f"{fname}: sst zlib=True",
                  f"{fname}: sst zlib not set", str(filters))
            check(filters.get("complevel") == 4,
                  f"{fname}: sst complevel=4",
                  f"{fname}: sst complevel wrong", str(filters))
            nc_dtype = str(f.variables["sst"].dtype)
            check("float32" in nc_dtype or "f4" in nc_dtype,
                  f"{fname}: sst stored as float32",
                  f"{fname}: sst stored as {nc_dtype}")


# ── I: Deep Value Inspection ──────────────────────────────────────────────────

def check_deep_values(name: str, path: Path):
    section(f"I. Deep Value Inspection -- {name.upper()}")

    if not path.exists():
        fail(f"{name}.nc MISSING -- skipping deep check")
        return

    ds = xr.open_dataset(path)

    # Evaluate one variable at a time to keep RAM usage low 
    # (Train split is ~2GB total, so ~90MB per variable)
    for var in ALL_VARS:
        if var not in ds.data_vars:
            continue
            
        # load values to numpy for fast checking
        vals = ds[var].values
        
        n_nan = int(np.isnan(vals).sum())
        n_inf = int(np.isinf(vals).sum())
        
        if var in INPUT_VARS:
            check(n_nan == 0, 
                  f"{var}: 0 NaNs", 
                  f"{var}: {n_nan} NaNs found in inputs!")
        else: # TARGET_VARS
            check(n_nan > 0,
                  f"{var}: NaNs exist (land masked)",
                  f"{var}: 0 NaNs found (missing land mask!)")
                  
        check(n_inf == 0,
              f"{var}: 0 Infs",
              f"{var}: {n_inf} Infs found!")
              
        # Absurd values check: since data is normalized (Z-score), 
        # absolute values > 25 are extremely suspicious (25 std devs away).
        valid_vals = vals[np.isfinite(vals)]
        if len(valid_vals) > 0:
            v_min = float(valid_vals.min())
            v_max = float(valid_vals.max())
            
            check(-25.0 <= v_min and v_max <= 25.0,
                  f"{var}: values in reasonable range [{v_min:.2f}, {v_max:.2f}]",
                  f"{var}: absurd values detected: min={v_min:.2f}, max={v_max:.2f}")

    ds.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  OceanEmbed Output Verification")
    print("=" * 62)

    check_file_inventory()
    check_land_mask()
    check_year_checkpoints()
    stats = check_norm_stats() or {}

    section("E. Final Split Checks")
    if not stats:
        stats_path = OUT_DIR / "normalization_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

    train_path = OUT_DIR / "train.nc"
    val_path   = OUT_DIR / "val.nc"
    test_path  = OUT_DIR / "test.nc"

    train_ds = check_split("train", train_path, TRAIN_YEARS, EXPECTED_TRAIN_DAYS, stats)
    val_ds   = check_split("val",   val_path,   VAL_YEARS,   EXPECTED_VAL_DAYS,   {})
    test_ds  = check_split("test",  test_path,  TEST_YEARS,  EXPECTED_TEST_DAYS,  {})

    if all(ds is not None for ds in [train_ds, val_ds, test_ds]):
        check_temporal_leakage(train_ds, val_ds, test_ds)
        check_mask_consistency(train_ds, val_ds, test_ds)

    for ds in [train_ds, val_ds, test_ds]:
        if ds is not None:
            ds.close()

    check_netcdf_encoding()

    for name, p in [("train", train_path), ("val", val_path), ("test", test_path)]:
        check_deep_values(name, p)

    # Summary
    print("\n" + "=" * 62)
    total = PASS_COUNT + FAIL_COUNT
    if FAIL_COUNT == 0:
        print(f"  RESULT: {PASS_COUNT}/{total} checks passed  -- ALL PASS")
    else:
        print(f"  RESULT: {PASS_COUNT}/{total} passed  -- {FAIL_COUNT} FAILED")
        print("\n  Failed checks:")
        for i, msg in enumerate(FAILURES, 1):
            print(f"    {i}. {msg}")
    print("=" * 62 + "\n")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
