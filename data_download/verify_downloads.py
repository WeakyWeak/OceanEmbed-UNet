#!/usr/bin/env python3
"""
Independently verify every downloaded file. Run this before preprocessing.

    .venv/bin/python data_download/verify_downloads.py
    .venv/bin/python data_download/verify_downloads.py --quick
    .venv/bin/python data_download/verify_downloads.py --sources cmems

THIS SCRIPT DELIBERATELY IGNORES THE .ok SIDECARS.

The downloader writes a sidecar after validating a file, and uses that sidecar
to decide what to skip. If this script also trusted the sidecars it would be
checking the downloader's bookkeeping rather than the data, and would agree
with it by construction - including when both are wrong. So everything here is
re-derived from the NetCDF files themselves.

It also checks three things the downloader does not, because they need the
whole file rather than its header:

  * TIME AXIS REGULARITY. A file can hold exactly 365 steps and still be
    broken - duplicated days, a gap, or an out-of-order axis. The step count
    alone would pass it.
  * ALL-NaN FIELDS. A subset request that lands entirely on land, or a
    server-side failure that returns a correctly shaped empty grid, produces a
    file that opens fine and has the right dimensions.
  * DTYPE. A field that arrives as int16 without scale/offset would silently
    quantise the temperatures.

Exits non-zero if anything fails, so it can gate the preprocessing step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_cleaning"))
import grid as G                                                # noqa: E402
from _common import FileSpec, check, coord_names, inspect       # noqa: E402


def all_specs(sources: list[str]) -> list[FileSpec]:
    """Specs for the sources the SATELLITE-ONLY model actually consumes.

    ERA5 is gone: it is atmospheric reanalysis, the fetch script was deleted,
    and verifying files nothing reads would only invent failures.

    GLORYS uo/vo ("currents") is opt-in rather than default. It is no longer a
    model input - it was ocean-model output, and it came from the same
    reanalysis run as the thetao target - but the files are kept so the
    "what does satellite-only cost?" ablation can be run against them. Verify
    it explicitly with --sources currents when you want that.
    """
    specs: list[FileSpec] = []
    if "cmems" in sources:
        import fetch_cmems
        specs += fetch_cmems.build_specs(["thetao", "sla", "sss"], G.YEARS)
    if "currents" in sources:
        import fetch_cmems
        specs += fetch_cmems.build_specs(["currents"], G.YEARS)
    if "oisst" in sources:
        import fetch_oisst
        specs += fetch_oisst.build_specs(G.YEARS)
    return specs


def deep_check(spec: FileSpec, quick: bool) -> list[str]:
    """Checks that need the file's contents, not just its header."""
    import xarray as xr
    bad: list[str] = []
    with xr.open_dataset(spec.path) as ds:
        _, _, tim = coord_names(ds)

        # ── time axis regularity ────────────────────────────────────────────
        if tim and ds.sizes.get(tim, 0) > 1:
            t = ds[tim].values
            try:
                d = np.diff(t.astype("datetime64[s]").astype("int64"))
                step = 3600 if spec.steps_per_day == 24 else 86400
                if (d <= 0).any():
                    bad.append("time axis not strictly increasing")
                elif not np.all(d == step):
                    uniq = np.unique(d)
                    bad.append(f"irregular time steps (s): {uniq[:5].tolist()}"
                               f"{' ...' if uniq.size > 5 else ''}")
            except Exception:                                   # noqa: BLE001
                bad.append("time axis could not be decoded")

        if quick:
            return bad

        # ── content sanity, first and last step only ────────────────────────
        for v in spec.variables:
            if v not in ds:
                continue
            da = ds[v]
            if not np.issubdtype(da.dtype, np.floating):
                bad.append(f"{v} dtype is {da.dtype}, expected float")
            sel = [0, -1] if tim and ds.sizes.get(tim, 0) > 1 else [0]
            for i in sel:
                arr = (da.isel({tim: i}) if tim else da).values
                finite = np.isfinite(arr).mean()
                if finite == 0.0:
                    bad.append(f"{v} step {i} is entirely NaN")
                elif finite < 0.05:
                    bad.append(f"{v} step {i} only {finite:.1%} finite")
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=["cmems", "oisst"],
                   choices=["cmems", "oisst", "currents"],
                   help="default verifies what the model consumes; add "
                        "'currents' for the ablation-only GLORYS uo/vo files")
    p.add_argument("--quick", action="store_true",
                   help="header and time axis only; skip the NaN/dtype scan")
    p.add_argument("--max-report", type=int, default=25)
    args = p.parse_args(argv)

    specs = all_specs(args.sources)
    print(G.describe())
    print(f"\nverifying {len(specs)} expected files"
          f"{' (quick)' if args.quick else ''} ...\n")

    missing, failed, ok = [], [], 0
    by_source: dict[str, list[int]] = {}

    for n, spec in enumerate(specs, 1):
        src = spec.path.parent.name
        by_source.setdefault(src, [0, 0, 0])          # ok, missing, failed
        if not spec.path.exists():
            missing.append(spec)
            by_source[src][1] += 1
        else:
            try:
                problems = check(inspect(spec.path), spec)
                problems += deep_check(spec, args.quick)
            except Exception as e:                              # noqa: BLE001
                problems = [f"unreadable: {type(e).__name__}: {str(e)[:120]}"]
            if problems:
                failed.append((spec, problems))
                by_source[src][2] += 1
            else:
                ok += 1
                by_source[src][0] += 1
        if n % 50 == 0 or n == len(specs):
            print(f"  {n}/{len(specs)} checked  "
                  f"ok={ok} missing={len(missing)} failed={len(failed)}",
                  flush=True)

    print(f"\n{'=' * 78}\n  RESULT\n{'=' * 78}")
    print(f"  {'directory':<26} {'ok':>6} {'missing':>8} {'failed':>7}")
    for src, (o, m, f) in sorted(by_source.items()):
        print(f"  {src:<26} {o:6d} {m:8d} {f:7d}")

    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for s in missing[:args.max_report]:
            print(f"    {s.path.relative_to(s.path.parents[2])}")
        if len(missing) > args.max_report:
            print(f"    ... and {len(missing) - args.max_report} more")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for s, problems in failed[:args.max_report]:
            print(f"    {s.path.name}: {'; '.join(problems)}")
        if len(failed) > args.max_report:
            print(f"    ... and {len(failed) - args.max_report} more")

    if missing or failed:
        print(f"\n  NOT READY. Re-run the relevant fetch script; it will skip\n"
              f"  everything already complete and retry only these.\n")
        return 1

    print(f"\n  All {ok} files present and verified. Ready for preprocessing.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
