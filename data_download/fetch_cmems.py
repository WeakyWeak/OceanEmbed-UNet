#!/usr/bin/env python3
"""
Download every Copernicus Marine product for the expanded domain and record.

    GLORYS thetao   15 depths x 32 years = 480 files   ~51 GB
    GLORYS uo/vo    surface, 32 files                   ~7 GB
    DUACS SLA       32 files                            ~3 GB
    Multi-Obs SSS   32 files                            ~3 GB

Usage
-----
    .venv/bin/python data_download/fetch_cmems.py --dry-run     # plan only
    .venv/bin/python data_download/fetch_cmems.py               # everything
    .venv/bin/python data_download/fetch_cmems.py --products thetao
    .venv/bin/python data_download/fetch_cmems.py --years 1993 1994

Safe to Ctrl-C and re-run at any point: it picks up at the first file that has
not been downloaded AND validated. See _common.py for why existence of a .nc
is never enough.

Why 480 separate thetao requests
--------------------------------
Deliberate. The alternative - one request per year with a depth RANGE - makes
the server return every native z-level between the surface and 1000 m, about
40 levels rather than the 15 we keep. That is roughly 140 GB downloaded to
store 51 GB, and it also destroys resume granularity: one request failing at
90% costs a whole year, whereas one of these failing costs one depth of one
year and retries in a couple of minutes.

Why year-outer, depth-inner
---------------------------
The old script looped depth-outer, so an interrupted run left ALL depths for
the early years and SOME depths for one year - which is useless, because
preprocessing needs all 15 depths of a year before it can regrid that year.
Year-outer means every completed year is immediately processable, so Phase 2
regridding can run against finished years while later years still download.
That overlap is what makes the schedule fit in a week instead of two.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_cleaning"))
import grid as G                                                # noqa: E402
from _common import RAW_ROOT, FileSpec, run_specs               # noqa: E402

# Explicit path: find_dotenv() searches upward from the CALLER, so it
# silently returns nothing when invoked from another directory - and CMEMS
# then blocks waiting for credentials on stdin with no error.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
USER = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
PASS = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")

MARINE_DIR = RAW_ROOT / "COPERNICUS_MARINE"
CURRENTS_DIR = RAW_ROOT / "COPERNICUS_CURRENTS"

DATASETS = {
    "thetao":   "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "currents": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "sla":      "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    "sss":      "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
}
# Which grid.PRODUCT_END entry governs each product's expected day count.
COVERAGE = {"thetao": "glorys", "currents": "glorys", "sla": "sla", "sss": "sss"}

# DUACS ships geostrophic velocity in the SAME L4 product as the sea level
# anomaly, computed from the altimetric SSH field by the geostrophic relation.
# We take them because the project is now satellite-only: GLORYS uo/vo are ocean
# MODEL output and had to go, and ugos/vgos are the observational equivalent of
# exactly that field rather than a loose substitute.
#
# Dropping GLORYS currents also removes a circularity - they came from the same
# reanalysis run that produces our thetao target, so surface currents and
# subsurface temperature were consistent through the model's own equations.
#
# Not taking `adt`: it is MDT + sla, and the time-invariant MDT part is already
# supplied to the network by the per-cell climatology and the CoordConv lat/lon
# channels, so its marginal value does not justify the extra download.
#
# NOTE: `variables` is part of FileSpec.spec_hash(), so simply changing this list
# invalidates every existing sla_*.nc.ok sidecar and the re-fetch is automatic.
SLA_VARS = ["sla", "ugos", "vgos"]


def _subset(spec: FileSpec, dataset_id: str, variables: list[str], extra: dict):
    """Return a fetch_fn(dest) closure for _common.fetch_one."""
    import copernicusmarine as cm
    lo, hi = G.year_bounds(spec.year, spec.product)

    def go(dest: Path) -> None:
        cm.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=G.DL_LON_MIN, maximum_longitude=G.DL_LON_MAX,
            minimum_latitude=G.DL_LAT_MIN, maximum_latitude=G.DL_LAT_MAX,
            # End at T00:00:00, NOT T23:59:59. coordinates_selection_method
            # ="nearest" is needed to snap the DEPTH onto a native z-level, but
            # it applies to every coordinate including time - so an end of
            # 23:59:59 snaps FORWARD to the next day and every GLORYS file
            # comes back with one extra day. Verified: a 1-day request returned
            # 2 steps, a 5-day request with this fix returns exactly 5.
            start_datetime=f"{lo.isoformat()}T00:00:00",
            end_datetime=f"{hi.isoformat()}T00:00:00",
            # Directory + bare filename is the documented 2.x form; passing a
            # full path works but makes the extension handling above harder to
            # reason about.
            output_directory=str(dest.parent),
            output_filename=dest.name,
            overwrite=True,                # writing to the .part path, not the final one
            username=USER, password=PASS,
            disable_progress_bar=True,
            **extra,
        )
    return go


def build_specs(products: list[str], years: list[int],
                depths: list[int] | None = None) -> list[FileSpec]:
    specs: list[FileSpec] = []

    # thetao first, year-outer / depth-inner, so (1993, 0 m) is the very first
    # file fetched. pipeline.py builds the land mask from that one file and
    # everything downstream consumes the mask, so it must exist early.
    if "thetao" in products:
        for year in years:
            for d in (depths if depths is not None else G.SIH_DEPTHS):
                req = G.SURFACE_DEPTH if d == 0 else float(d)
                specs.append(FileSpec(
                    path=MARINE_DIR / f"glorys_thetao_{d}m_{year}.nc",
                    product=COVERAGE["thetao"], year=year,
                    label=f"thetao {d}m {year}", variables=["thetao"],
                    params={"depth": req, "dataset": DATASETS["thetao"]}))

    if "currents" in products:
        for year in years:
            specs.append(FileSpec(
                path=CURRENTS_DIR / f"glorys_surface_currents_{year}.nc",
                product=COVERAGE["currents"], year=year,
                label=f"currents {year}", variables=["uo", "vo"],
                params={"depth": G.SURFACE_DEPTH, "dataset": DATASETS["currents"]}))

    if "sla" in products:
        for year in years:
            specs.append(FileSpec(
                path=MARINE_DIR / f"sla_{year}.nc",
                product=COVERAGE["sla"], year=year,
                label=f"sla {year}", variables=SLA_VARS,
                params={"dataset": DATASETS["sla"]}))

    if "sss" in products:
        for year in years:
            specs.append(FileSpec(
                path=MARINE_DIR / f"sss_{year}.nc",
                product=COVERAGE["sss"], year=year,
                label=f"sss {year}", variables=["sos"],
                params={"dataset": DATASETS["sss"]}))

    # A product that does not cover a year at all yields expected_days == 0;
    # drop those rather than requesting an empty range.
    return [s for s in specs if s.expected_days > 0]


def fetch_fn_for(spec: FileSpec):
    name = spec.label.split()[0]
    if name == "thetao":
        return _subset(spec, DATASETS["thetao"], ["thetao"],
                       {"minimum_depth": spec.params["depth"],
                        "maximum_depth": spec.params["depth"],
                        "coordinates_selection_method": "nearest"})
    if name == "currents":
        return _subset(spec, DATASETS["currents"], ["uo", "vo"],
                       {"minimum_depth": G.SURFACE_DEPTH,
                        "maximum_depth": G.SURFACE_DEPTH + 0.001,
                        "coordinates_selection_method": "nearest"})
    if name == "sla":
        return _subset(spec, DATASETS["sla"], SLA_VARS, {})
    if name == "sss":
        return _subset(spec, DATASETS["sss"], ["sos"], {})
    raise KeyError(spec.label)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--products", nargs="+",
                   default=["thetao", "currents", "sla", "sss"],
                   choices=["thetao", "currents", "sla", "sss"])
    p.add_argument("--years", nargs="+", type=int, default=G.YEARS)
    p.add_argument("--depths", nargs="+", type=int, default=None,
                   choices=G.SIH_DEPTHS,
                   help="thetao depths only; default all 15. Useful for "
                        "re-fetching one level without re-scanning the rest.")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be fetched; touches no network")
    args = p.parse_args(argv)

    if not args.dry_run and not (USER and PASS):
        print("COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD not set (.env)",
              file=sys.stderr)
        return 2

    print(G.describe())
    specs = build_specs(args.products, sorted(args.years), args.depths)
    fails = run_specs(specs, fetch_fn_for, "Copernicus Marine", args.dry_run)

    if fails:
        print(f"\n{fails} file(s) failed. Re-run this command - completed files "
              f"are skipped.\n", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("\nAll Copernicus Marine files present and validated.")
        print("Next: .venv/bin/python data_download/verify_downloads.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
