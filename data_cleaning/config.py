"""
OceanEmbed Data Cleaning Pipeline - Configuration
All paths, constants, grid definitions, and dataset metadata.

SATELLITE-ONLY, WIDE DOMAIN (revision of 2026-09-10)
---------------------------------------------------
Two changes drive this file, and they are related.

1. Inputs are now satellite-only. ERA5 10 m winds are ATMOSPHERIC REANALYSIS and
   GLORYS uo/vo are OCEAN MODEL output; neither is an observation, so both are
   gone. Their loaders, regridders and VARS entries were deleted rather than
   left unused, so any code still reaching for them fails loudly at import
   instead of silently processing a variable the model will never see.

   GLORYS uo/vo also came from the same reanalysis run that produces the thetao
   TARGET, so surface currents and subsurface temperature were consistent
   through the ocean model's own equations. Removing them deletes a circularity
   as well as a compliance problem.

   They are replaced by `ugos`/`vgos` - geostrophic velocity that DUACS derives
   from the altimeter's own sea-surface-height field, shipped inside the SAME
   L4 product as `sla`. So "currents" is no longer a separate source: it is two
   extra variables in the file we already read.

2. The domain widened to 50-100 E (adding the Arabian Sea) and the record now
   runs 1993-2024. Grid and calendar are therefore imported from grid.py rather
   than declared here - they were previously declared in TWO places, and the
   copy in training/train_config.py silently disagreeing is a failure mode with
   no error attached to it.

Outputs are versioned (`processed_v2`) because every stage of this pipeline
resumes on FILENAME EXISTENCE while this revision changes every file's SHAPE
and CHANNEL SET. Writing into the old directory would let a stale artefact be
picked up as complete - and `apply_mask` in masking.py is `da.where(mask == 1)`,
which xarray resolves with an INNER JOIN, so an old 81x101 mask against 81x201
data does not raise and does not produce NaN. It silently truncates the result
back to the Bay of Bengal.
"""
import numpy as np
from pathlib import Path

import grid as G

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw2"          # widened-domain downloads
OUT_DIR      = PROJECT_ROOT / "data" / "processed_v2"  # versioned: see docstring
YEARS_DIR    = OUT_DIR / "years"                       # per-year checkpoint files

OISST_DIR    = RAW_DIR / "NOAA_OISST"
MARINE_DIR   = RAW_DIR / "COPERNICUS_MARINE"

# ── Target Grid ──────────────────────────────────────────────────────────────
# Single source of truth. Do not redeclare these.
TARGET_LAT = G.TARGET_LAT      # 81 points, 5.0 .. 25.0 N
TARGET_LON = G.TARGET_LON      # 201 points, 50.0 .. 100.0 E

# ── Temporal ─────────────────────────────────────────────────────────────────
YEARS = G.YEARS                               # 1993 .. 2024

# Chronological split: held-out years sit at the END of the record, which is
# both the harder test and the one a reviewer expects. training/splits.py is the
# authority for per-day membership; these lists exist so the pipeline knows
# which years may contribute to the climatology and the normalisation stats.
#
# Those artefacts are computed on TRAINING DAYS ONLY, so changing the split
# invalidates them. Anything derived from them must be scoped by split name.
TRAIN_YEARS = list(range(1993, 2021))         # 1993-2020, 28 years
VAL_YEARS   = [2021, 2022]
TEST_YEARS  = [2023, 2024]                    # 2024 is partial: SSS ends 12-15

TRAIN_START, TRAIN_END = "1993-01-01", "2020-12-31"
VAL_START,   VAL_END   = "2021-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", str(G.RECORD_END)

# ── Target depths (metres) ──────────────────────────────────────────────────
TARGET_DEPTHS = G.SIH_DEPTHS
DEPTH_LABELS  = [f"{d}m" for d in TARGET_DEPTHS]

# ── Dataset file patterns ────────────────────────────────────────────────────
# No "combined file" fallbacks. The old versions of these returned
# `single if single else combined`, so a year whose own file was missing
# silently fell back to a multi-year file from the PREVIOUS domain - a
# wrong-domain, wrong-period answer with no error. A missing year must be a
# missing year; the caller decides what to do about it.

def oisst_files_for_year(year: int):
    return sorted(OISST_DIR.glob(f"sst.day.mean.{year}.nc"))

def sla_files_for_year(year: int):
    """Also carries ugos/vgos - see VARS["sla"]."""
    return sorted(MARINE_DIR.glob(f"sla_{year}.nc"))

def sss_files_for_year(year: int):
    return sorted(MARINE_DIR.glob(f"sss_{year}.nc"))

def thetao_files_for_year(depth_label: str, year: int):
    return sorted(MARINE_DIR.glob(f"glorys_thetao_{depth_label}_{year}.nc"))


# ── Variable names per dataset ───────────────────────────────────────────────
# `era5` and `currents` entries deliberately absent - see module docstring.
VARS = {
    "oisst":    {"sst": "sst", "time": "time", "lat": "lat", "lon": "lon"},
    # ugos/vgos ride along in the DUACS L4 file alongside sla.
    "sla":      {"sla": "sla", "ugos": "ugos", "vgos": "vgos", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
    "sss":      {"sos": "sos", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
    "thetao":   {"thetao": "thetao", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
}

# The surface inputs the network receives, in canonical channel order.
# training/train_config.py must agree with this list.
SURFACE_VARS = ["sst", "sla", "sss", "ugos", "vgos"]
