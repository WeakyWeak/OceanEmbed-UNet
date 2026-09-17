#!/usr/bin/env python3
"""
Download NOAA OISST v2 high-resolution SST, and reuse what is already on disk.

    32 files, one per year, GLOBAL   ~15 GB total
    but 18 of those years are already downloaded and still valid

Usage
-----
    .venv/bin/python data_download/fetch_oisst.py --dry-run
    .venv/bin/python data_download/fetch_oisst.py            # links + fetches
    .venv/bin/python data_download/fetch_oisst.py --no-reuse # ignore old files

Why 18 of 32 years are free
---------------------------
These files are GLOBAL - the old download script never subsetted them, and the
crop to the study domain happens later in data_cleaning/loaders.py. So the
domain expansion does not invalidate them: a global file covers 50-100 E just
as well as it covered 75-100 E.

Every other product had to be re-fetched because it was subsetted server-side
to the old box. OISST is the one place where widening the domain costs nothing,
so the existing data/raw/NOAA_OISST/ files are hard-linked into raw2 (falling
back to a copy across filesystems) and then validated exactly as if they had
just been downloaded. Only 1993-2004 and 2023-2024 are actually fetched.

That is ~8.6 GB and several hours of bandwidth saved, and it is safe precisely
BECAUSE the files are validated after linking rather than trusted. If a linked
file were somehow wrong, it fails the same check a fresh download would.

Resume
------
These are static files over plain HTTP, so this is the one source with genuine
byte-range resume: an interrupted transfer continues from where it stopped
rather than restarting. The existing script's Range logic is preserved.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_cleaning"))
import grid as G                                                # noqa: E402
from _common import (RAW_ROOT, PROJECT_ROOT, FileSpec, run_specs,   # noqa: E402
                     inspect, check, mark_done, is_done)

OISST_DIR = RAW_ROOT / "NOAA_OISST"
LEGACY_DIR = PROJECT_ROOT / "data" / "raw" / "NOAA_OISST"
BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
CHUNK = 1 << 16
HTTP_RETRIES = 20


def _download(spec: FileSpec):
    """Byte-range resuming fetch. Writes to `dest`, which is the .part path."""
    import requests
    url = f"{BASE_URL}/sst.day.mean.{spec.year}.nc"

    def go(dest: Path) -> None:
        # The resume file hangs off the FINAL path, not the .part path.
        # _common._clear_partials globs "<stem>.part.nc*" before every attempt,
        # so a resume file named "<stem>.part.nc.resume" would be deleted on
        # each retry and across script re-runs - destroying exactly the
        # dropped-connection resume this source exists to provide. Naming it
        # off the final path keeps it out of that glob.
        # Safe to append to across runs: these are static files, so a partial
        # transfer of year N is always a valid prefix of year N.
        resume = spec.path.with_suffix(spec.path.suffix + ".resume")
        for attempt in range(1, HTTP_RETRIES + 1):
            have = resume.stat().st_size if resume.exists() else 0
            try:
                r = requests.get(url, headers={"Range": f"bytes={have}-"},
                                 timeout=60, stream=True)
                if r.status_code == 416:          # already complete
                    r.close()
                    break
                r.raise_for_status()
                with open(resume, "ab") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                r.close()
                break
            except Exception:                                   # noqa: BLE001
                if attempt == HTTP_RETRIES:
                    raise
                time.sleep(5)
        os.replace(resume, dest)
    return go


def build_specs(years: list[int]) -> list[FileSpec]:
    return [FileSpec(
        path=OISST_DIR / f"sst.day.mean.{y}.nc",
        product="oisst", year=y, label=f"oisst {y}", variables=["sst"],
        params={"source": "psl.noaa.gov highres", "global": True},
        # These are whole calendar years; the record window is applied later
        # by the time-axis intersection, same as ERA5.
        min_days=366 if (y % 4 == 0 and (y % 100 or y % 400 == 0)) else 365,
    ) for y in years]


def reuse_existing(specs: list[FileSpec]) -> int:
    """Hard-link already-downloaded global files into raw2 and validate them.

    Validated, not trusted: a linked file goes through exactly the same
    inspect/check as a fresh download before its sidecar is written.
    """
    linked = 0
    for spec in specs:
        if is_done(spec):
            continue
        src = LEGACY_DIR / spec.path.name
        if not src.exists():
            continue
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if spec.path.exists():
                spec.path.unlink()
            try:
                os.link(src, spec.path)
                how = "hardlink"
            except OSError:
                shutil.copy2(src, spec.path)
                how = "copy"
            found = inspect(spec.path)
            problems = check(found, spec)
            if problems:
                print(f"  reuse REJECTED {spec.path.name}: {'; '.join(problems)}")
                spec.path.unlink()
                continue
            mark_done(spec, found)
            linked += 1
            print(f"  reused ({how}) {spec.path.name}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  reuse failed {spec.path.name}: {type(e).__name__}: {e}")
    return linked


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, default=G.YEARS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-reuse", action="store_true",
                   help="re-download everything instead of linking data/raw/")
    args = p.parse_args(argv)

    specs = build_specs(sorted(args.years))

    if not args.no_reuse and not args.dry_run and LEGACY_DIR.exists():
        print(f"\nReusing global OISST files already in {LEGACY_DIR} ...")
        n = reuse_existing(specs)
        print(f"  {n} year(s) reused; the rest will be downloaded.")

    fails = run_specs(specs, _download, "NOAA OISST", args.dry_run)
    if fails:
        print(f"\n{fails} file(s) failed. Re-run - completed files are skipped.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
