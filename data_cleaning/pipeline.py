#!/usr/bin/env python3
"""
OceanEmbed — Year-by-Year Preprocessing Pipeline
==================================================

Implements the Run 18 blueprint with a streaming, year-by-year execution
model that keeps peak RAM well under 16 GB.

Architecture
------------
Phase 1  — Mask (once):
    Build the canonical 81×201 land mask from GLORYS thetao 0m.
    Saved to: data/processed_v2/land_mask.nc

Phase 2  — Per-year regrid & save (resume-safe):
    For each year 1993–2024:
      1. Load that year's raw data (all variables)
      2. Regrid to 0.25° target grid
      3. Align daily timestamps
      4. Apply canonical mask
      5. Compute into memory and write to data/processed_v2/years/<year>.nc
      → If that file already exists, SKIP.

    There is no wind resampling step any more. ERA5 u10/v10 were hourly and
    needed a midnight-aligned daily resample; they are gone (atmospheric
    reanalysis, not observations), along with GLORYS uo/vo. Every input is now
    daily to begin with. See config.py for the full reasoning.

Phase 3  — Streaming normalization (train years only):
    Stream each training year from the per-year checkpoint files through
    Welford's online algorithm. Produces normalization_stats.json without
    ever loading more than one year at a time.
    → Skipped if normalization_stats.json already exists.

Phase 4  — Assemble ONE record:
    Stream every year from the per-year checkpoints, apply Z-score in-place,
    zero-fill inputs, and write a single continuous file.
    Written as: data/processed_v2/record.nc  (1993-01-01 .. 2024-12-15)

    This used to emit three files - train.nc / val.nc / test.nc - which baked
    the split boundaries into the data itself. The split is now applied at
    read time from training/splits.py, so changing it does not mean rebuilding
    8 GB of NetCDF. Current boundaries: train 1993-2020, val 2021-2022,
    test 2023-01-01 .. 2024-12-15.

Resume
------
Run with --resume to skip already-completed phases:
    python data_cleaning/pipeline.py --resume

Force restart of a specific phase:
    python data_cleaning/pipeline.py --phase 2   # re-run phase 2 onwards
    python data_cleaning/pipeline.py --phase 3   # re-run phases 3 & 4

Usage:
    cd <project_root>
    source .venv/bin/activate
    python data_cleaning/pipeline.py           # full run
    python data_cleaning/pipeline.py --resume  # skip completed steps
    python data_cleaning/pipeline.py --phase 3 # restart from phase 3
"""
import argparse
import sys
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import netCDF4 as nc4
import xarray as xr
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grid as G
from config import (
    OUT_DIR, YEARS_DIR, YEARS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    DEPTH_LABELS,
)
from loaders import (
    load_oisst_year,
    load_sla_year, load_sss_year, load_thetao_year,
)
from regridding import (
    regrid_oisst, regrid_sla, regrid_sss,
    regrid_thetao,
)
from temporal import align_daily_times
from masking import generate_land_mask, apply_mask, save_mask
from normalize import (
    init_accumulators, update_accumulators, accumulators_to_stats,
    save_norm_stats, load_norm_stats,
    apply_zscore_inplace, zero_fill_inputs_inplace,
)

# Satellite-only. ERA5 u10/v10 (atmospheric reanalysis) and GLORYS uo/vo
# (ocean model) are gone; ugos/vgos are DUACS geostrophic velocity derived from
# the altimeter's own SSH field and arrive in the same file as sla.
INPUT_VARS  = ["sst", "sla", "sss", "ugos", "vgos"]
TARGET_VARS = [f"thetao_{lbl}" for lbl in DEPTH_LABELS]
ALL_VARS    = INPUT_VARS + TARGET_VARS

NC_ENCODING = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
               for v in ALL_VARS}


def banner(msg: str):
    print(f"\n{'='*62}\n  {msg}\n{'='*62}")


# ── Phase 1: Mask ─────────────────────────────────────────────────────────────

def phase1_mask(force: bool = False) -> xr.DataArray:
    mask_path = OUT_DIR / "land_mask.nc"

    if mask_path.exists() and not force:
        print(f"[Phase 1] Mask already exists — loading from {mask_path}")
        return xr.open_dataset(mask_path)["land_mask"]

    banner("Phase 1: Generating Canonical Land Mask")
    # Surface thetao for ONE year, at native resolution - tiny memory footprint.
    # Uses YEARS[0] rather than a hardcoded 2015: that literal referred to a
    # combined multi-year file from the old domain which no longer exists, and
    # the mask must be built from data on the CURRENT box or it encodes the
    # wrong coastline.
    mask_year = YEARS[0]
    thetao_native_year = load_thetao_year(mask_year)
    thetao_0m = thetao_native_year["0m"].isel(time=0).compute()
    del thetao_native_year

    # Expand back to a single-timestep DataArray with time dim for the helper
    thetao_dummy = thetao_0m.expand_dims("time")
    land_mask = generate_land_mask(thetao_dummy)

    _report_ocean_fraction(land_mask, mask_year)

    save_mask(land_mask)
    return land_mask


def _report_ocean_fraction(land_mask: xr.DataArray, year: int) -> None:
    """Ocean fraction PER LONGITUDE QUADRANT, not just the total.

    The domain now spans two basins. If the Arabian Sea half came back empty -
    a download that silently returned the old box, or a bilinear step whose
    source grid did not cover 50 E - the overall percentage would still look
    plausible (the old Bay-of-Bengal domain was 55.6% ocean, and a half-empty
    wide domain lands near 45%). Only the per-quadrant breakdown makes it
    obvious, so it is printed every time rather than left to a separate check.
    """
    m = land_mask.values
    lon = land_mask.lon.values
    total = float((m == 1).mean())
    print("")
    print(f"  Land mask from {year}: {total * 100:.1f}% ocean overall")
    edges = np.linspace(lon[0], lon[-1], 5)
    for k in range(4):
        sel = (lon >= edges[k]) & (lon <= edges[k + 1])
        frac = float((m[:, sel] == 1).mean())
        print(f"    {edges[k]:6.1f}-{edges[k+1]:6.1f} E : {frac * 100:5.1f}% ocean")
    west = float((m[:, lon < 75.0] == 1).mean())
    east = float((m[:, lon >= 75.0] == 1).mean())
    print(f"    Arabian Sea half (<75 E): {west * 100:.1f}%   "
          f"Bay of Bengal half (>=75 E): {east * 100:.1f}%")
    if west < 0.10 or east < 0.10:
        raise RuntimeError(
            f"A half of the domain is essentially all land "
            f"(west {west:.1%}, east {east:.1%}). The source data almost "
            f"certainly does not cover the full 50-100 E box.")


# ── Phase 2: Per-year regrid & checkpoint ────────────────────────────────────

def process_one_year(year: int, land_mask: xr.DataArray) -> None:
    """Full pipeline for one year: load → regrid → mask → save."""
    t_year = time.time()
    print(f"\n  ── Year {year} ──")

    # 1–3: Load & coord harmonize
    sst_raw                      = load_oisst_year(year)
    sla_raw, ugos_raw, vgos_raw  = load_sla_year(year)
    sss_raw                      = load_sss_year(year)
    thetao_raw                   = load_thetao_year(year)

    # 4: Spatial regrid (one depth at a time for thetao to bound RAM)
    sst_grid = regrid_oisst(sst_raw)
    del sst_raw

    sla_grid, ugos_grid, vgos_grid = regrid_sla(sla_raw, ugos_raw, vgos_raw)
    del sla_raw, ugos_raw, vgos_raw

    sss_grid = regrid_sss(sss_raw)
    del sss_raw

    # Regrid thetao one depth at a time to bound peak RAM
    thetao_grid = {}
    for label in DEPTH_LABELS:
        da = thetao_raw[label]
        thetao_grid[label] = regrid_thetao({label: da})[label]
        del da
    del thetao_raw

    # 5 (others): Align all daily timestamps
    # Order here must match INPUT_VARS.
    all_arrays = [sst_grid, sla_grid, sss_grid, ugos_grid, vgos_grid]
    all_arrays += [thetao_grid[lbl] for lbl in DEPTH_LABELS]

    # How long this year SHOULD be, from each product's documented coverage.
    # The binding one is normally SSS, which stops on 2024-12-15 - so 2024 is
    # legitimately 350 days, not 366, and hardcoding "a year is 365/366 days"
    # would either reject a good 2024 or be loosened until it accepts a short
    # 1993.
    expect = min(G.expected_days(year, prod)
                 for prod in ("oisst", "sla", "sss", "glorys"))
    aligned = align_daily_times(*all_arrays, expect=expect)
    del all_arrays

    n_in = len(INPUT_VARS)
    sst_a, sla_a, sss_a, ugos_a, vgos_a = aligned[:n_in]
    thetao_a = {lbl: aligned[n_in + i] for i, lbl in enumerate(DEPTH_LABELS)}
    del aligned

    # 6: Apply mask
    data_vars = {
        "sst":  apply_mask(sst_a,  land_mask),
        "sla":  apply_mask(sla_a,  land_mask),
        "sss":  apply_mask(sss_a,  land_mask),
        "ugos": apply_mask(ugos_a, land_mask),
        "vgos": apply_mask(vgos_a, land_mask),
    }
    for lbl, da in thetao_a.items():
        data_vars[f"thetao_{lbl}"] = apply_mask(da, land_mask)

    del sst_a, sla_a, sss_a, ugos_a, vgos_a, thetao_a

    # Compute into RAM as float32, then write to disk
    year_ds = xr.Dataset(data_vars).astype(np.float32).compute()
    del data_vars

    out_path = YEARS_DIR / f"{year}.nc"
    year_ds.to_netcdf(out_path, encoding=NC_ENCODING)
    del year_ds
    gc.collect()

    elapsed = time.time() - t_year
    print(f"     → {out_path} written in {elapsed:.0f}s")


def year_ready(year: int) -> tuple[bool, str]:
    """Is every source this year needs present AND of the right vintage?

    Downloads land over hours, so Phase 2 runs repeatedly against a partially
    complete raw tree. Re-running must process what has arrived and leave the
    rest alone, which means readiness is checked per source rather than assumed.

    The sla check opens the file rather than trusting its name: sla_YYYY.nc
    existed BEFORE the satellite-only revision too, holding only `sla`. A
    filename test would happily feed a pre-revision file into a pipeline that
    now requires ugos/vgos, and the KeyError would surface 40 seconds into
    regridding instead of here.
    """
    from config import (oisst_files_for_year, sla_files_for_year,
                        sss_files_for_year, thetao_files_for_year)
    if not oisst_files_for_year(year):
        return False, "no OISST"
    if not sss_files_for_year(year):
        return False, "no SSS"
    missing = [lbl for lbl in DEPTH_LABELS
               if not thetao_files_for_year(lbl, year)]
    if missing:
        return False, f"thetao missing {len(missing)} depth(s)"
    f = sla_files_for_year(year)
    if not f:
        return False, "no SLA"
    try:
        with xr.open_dataset(f[0]) as ds:
            absent = [v for v in ("sla", "ugos", "vgos") if v not in ds]
    except Exception as e:                                       # noqa: BLE001
        return False, f"SLA unreadable ({type(e).__name__})"
    if absent:
        return False, f"SLA lacks {absent} - pre-revision file, re-fetch"
    return True, "ready"


def phase2_regrid(start_year: int, land_mask: xr.DataArray,
                  resume: bool = False) -> None:
    banner(f"Phase 2: Per-Year Regrid (start={start_year})")
    YEARS_DIR.mkdir(parents=True, exist_ok=True)

    done, skipped, waiting = 0, 0, []
    for year in YEARS:
        if year < start_year:
            continue
        out_path = YEARS_DIR / f"{year}.nc"
        if resume and out_path.exists():
            print(f"  [SKIP] {year} — checkpoint exists")
            skipped += 1
            continue
        ok, why = year_ready(year)
        if not ok:
            print(f"  [WAIT] {year} — {why}")
            waiting.append(year)
            continue
        process_one_year(year, land_mask)
        done += 1

    print("")
    print(f"  Phase 2: {done} processed, {skipped} already done, "
          f"{len(waiting)} waiting on downloads")
    if waiting:
        print(f"  waiting: {waiting}")
        print("  Re-run this once the downloads finish; completed years are "
              "skipped.")


# ── Phase 3: Streaming normalization stats ────────────────────────────────────

def phase3_stats(force: bool = False) -> dict:
    stats_path = OUT_DIR / "normalization_stats.json"

    if stats_path.exists() and not force:
        print(f"[Phase 3] Stats already exist — loading from {stats_path}")
        return load_norm_stats()

    banner("Phase 3: Streaming Normalization Stats (train years only)")
    # Load land_mask for ocean-cell filtering
    land_mask = xr.open_dataset(OUT_DIR / "land_mask.nc")["land_mask"]

    accumulators = init_accumulators(ALL_VARS)

    for year in TRAIN_YEARS:
        year_path = YEARS_DIR / f"{year}.nc"
        print(f"  Streaming year {year} …")
        year_ds = xr.open_dataset(year_path, chunks={}).astype(np.float32).compute()
        update_accumulators(accumulators, year_ds, land_mask)
        del year_ds
        gc.collect()

    print("\n  Final statistics:")
    stats = accumulators_to_stats(accumulators)
    save_norm_stats(stats)
    return stats


# ── Phase 4: Assemble final splits ────────────────────────────────────────────

def _parse_cf_epoch(units_str: str):
    """Parse a CF time units string like 'days since YYYY-MM-DD'.
    Returns (unit_type: str, epoch: pd.Timestamp).
    """
    parts = units_str.split()
    unit  = parts[0]  # 'days', 'hours', 'seconds'
    epoch = pd.Timestamp(" ".join(parts[2:]))
    return unit, epoch


def _times_to_cf_numeric(times_datetime64: np.ndarray, units_str: str) -> np.ndarray:
    """Convert numpy datetime64 array to CF-numeric integers/floats,
    using the exact units already written in the target NetCDF file.
    """
    unit, epoch = _parse_cf_epoch(units_str)
    ts    = pd.DatetimeIndex(times_datetime64)
    delta = ts - epoch
    if unit == "days":
        return delta.days.values.astype(np.int32)
    elif unit == "hours":
        return (delta.total_seconds() / 3600).values.astype(np.float64)
    elif unit == "seconds":
        return delta.total_seconds().values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported CF time unit: {unit!r}")


def _append_ds_nc4(ds: xr.Dataset, path: Path) -> None:
    """Append a fully computed xr.Dataset along the time axis.

    Uses the low-level netCDF4 library to extend an unlimited time
    dimension, matching the CF time units already in the file.
    This avoids loading any previously written data into RAM.

    IMPORTANT: `ds` must already be computed (not lazy) before calling.
    """
    with warnings.catch_warnings():
        # Suppress the NumPy 2.5 DeprecationWarning emitted by netCDF4 1.7.x
        # when it sets array shape internally. This is a netCDF4 bug,
        # not our code; correctness is unaffected.
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                module="netCDF4")
        with nc4.Dataset(path, "a") as f:
            t0        = f.dimensions["time"].size
            units_str = f.variables["time"].units
            n         = ds.sizes["time"]

            f.variables["time"][t0:t0 + n] = _times_to_cf_numeric(
                ds.time.values, units_str)

            for v in ds.data_vars:
                f.variables[v][t0:t0 + n] = ds[v].values


def _write_split(split_years: list, out_path: Path, stats: dict) -> None:
    """Stream per-year checkpoints into a single output file incrementally.

    Memory profile: approximately one year's dataset in RAM at a time,
    plus operating-system and Python overhead. No xr.concat() is used.

    If `out_path` already exists, it is removed first to ensure we never
    append to a file from a previous partial or failed run.
    """
    # Remove any partial file from a previous attempt
    if out_path.exists():
        out_path.unlink()
        print(f"     Removed existing partial file: {out_path.name}")

    for i, year in enumerate(split_years):
        year_path = YEARS_DIR / f"{year}.nc"
        print(f"     Processing year {year} …", end=" ", flush=True)
        t_yr = time.time()

        # Load ~one year into RAM
        ds = xr.open_dataset(year_path).astype(np.float32).compute()

        # Normalize and zero-fill in-place
        apply_zscore_inplace(ds, stats)
        zero_fill_inputs_inplace(ds, INPUT_VARS)

        if i == 0:
            # First year: write file with unlimited time dimension
            # and establish compression encoding for all variables.
            ds.to_netcdf(out_path, mode="w",
                         unlimited_dims=["time"],
                         encoding=NC_ENCODING,
                         engine="netcdf4")
        else:
            # Subsequent years: append via low-level netCDF4.
            # Encoding is NOT re-specified — it was locked in on first write.
            _append_ds_nc4(ds, out_path)

        elapsed = time.time() - t_yr
        print(f"{elapsed:.0f}s")

        ds.close()
        del ds
        gc.collect()


def phase4_assemble(stats: dict, force: bool = False) -> None:
    """Assemble ONE record.nc covering the whole 1993-2024 span.

    This used to emit three files - train.nc / val.nc / test.nc - which baked
    the split into the data itself. That is now wrong for two reasons:

      * The split is configurable (training/splits.py). Writing it into the
        NetCDF means changing SPLIT_MODE silently invalidates the files, with
        nothing to detect it.
      * A non-chronological split is not three contiguous blocks, so there is
        no correct way to chunk it into three files at all.

    build_arrays.py always concatenated the three back into one contiguous
    array anyway, so the split files were never more than chunking. One record
    plus splits.py deriving indices from dates.npy is strictly simpler and has
    one fewer thing to get out of step.
    """
    banner("Phase 4: Assembling the full record")

    out_path = OUT_DIR / "record.nc"
    if out_path.exists() and not force:
        print(f"  [SKIP] record.nc — already exists (use --force to rebuild)")
        return

    print(f"  Building record.nc from {YEARS[0]}–{YEARS[-1]} "
          f"({len(YEARS)} years) …")
    _write_split(YEARS, out_path, stats)
    print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(resume: bool = False, start_phase: int = 1) -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    YEARS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1
    force_p1 = (start_phase == 1 and not resume)
    land_mask = phase1_mask(force=force_p1)

    # Phase 2
    if start_phase <= 2:
        # When resuming, skip already-written year files
        # When restarting phase 2 exactly, determine which year to start from
        if resume:
            start_year = YEARS[0]
        else:
            start_year = YEARS[0]
        phase2_regrid(start_year=start_year, land_mask=land_mask, resume=resume)

    # Phase 3
    force_p3 = (start_phase >= 3 and not resume)
    stats = phase3_stats(force=force_p3)

    # Phase 4
    force_p4 = (start_phase >= 4 and not resume)
    phase4_assemble(stats, force=force_p4)

    elapsed = time.time() - t0
    print(f"\n✓  Pipeline complete in {elapsed/60:.1f} min")
    print(f"   Outputs: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OceanEmbed preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_cleaning/pipeline.py                 # fresh full run
  python data_cleaning/pipeline.py --resume        # skip finished steps
  python data_cleaning/pipeline.py --phase 3       # redo phases 3 & 4
  python data_cleaning/pipeline.py --phase 2 --resume  # resume phase 2 year-by-year
        """,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip phases and per-year files that are already on disk.",
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3, 4],
        help="Start from this phase (1=mask, 2=regrid, 3=stats, 4=splits). "
             "Phases earlier than this are loaded from disk.",
    )
    args = parser.parse_args()
    run_pipeline(resume=args.resume, start_phase=args.phase)
