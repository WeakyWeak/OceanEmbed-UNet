"""
OceanEmbed Data Cleaning Pipeline — Configuration
All paths, constants, grid definitions, and dataset metadata.
"""
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

ERA5_DIR     = RAW_DIR / "Copernicus_ERA5"
OISST_DIR    = RAW_DIR / "NOAA_OISST"
CURRENTS_DIR = RAW_DIR / "COPERNICUS_CURRENTS"
MARINE_DIR   = RAW_DIR / "COPERNICUS_MARINE"

# ── Target Grid ──────────────────────────────────────────────────────────────
TARGET_LAT = np.arange(5.0, 25.25, 0.25)   # 81 points
TARGET_LON = np.arange(75.0, 100.25, 0.25)  # 101 points

# ── Temporal ─────────────────────────────────────────────────────────────────
YEARS       = list(range(2005, 2023))         # 2005–2022 inclusive
TRAIN_YEARS = list(range(2005, 2021))         # 2005–2020
VAL_YEARS   = [2021]
TEST_YEARS  = [2022]

TRAIN_START, TRAIN_END = "2005-01-01", "2020-12-31"
VAL_START,   VAL_END   = "2021-01-01", "2021-12-31"
TEST_START,  TEST_END  = "2022-01-01", "2022-12-31"

# ── Target depths (metres) ──────────────────────────────────────────────────
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS  = [f"{d}m" for d in TARGET_DEPTHS]

# ── Dataset file patterns ────────────────────────────────────────────────────
# ERA5: per-year files, hourly, coords = (valid_time, latitude, longitude)
def era5_files():
    return sorted(ERA5_DIR.glob("era5_winds_*.nc"))

# OISST: per-year files, daily, coords = (time, lat, lon)
def oisst_files():
    return sorted(OISST_DIR.glob("sst.day.mean.*.nc"))

# GLORYS currents: mixed — one multi-year + per-year files
def currents_files():
    return sorted(CURRENTS_DIR.glob("glorys_surface_currents_*.nc"))

# SLA: mixed — one multi-year + per-year files
def sla_files():
    return sorted(MARINE_DIR.glob("sla_*.nc"))

# SSS: mixed — one multi-year + per-year files
def sss_files():
    return sorted(MARINE_DIR.glob("sss_*.nc"))

# GLORYS thetao: per-depth, each depth has multi-year + per-year files
def thetao_files(depth_label: str):
    """Return all files for one depth level, e.g. '0m', '100m'."""
    return sorted(MARINE_DIR.glob(f"glorys_thetao_{depth_label}_*.nc"))


# ── Variable names per dataset ───────────────────────────────────────────────
VARS = {
    "era5":     {"u10": "u10", "v10": "v10", "time": "valid_time",
                 "lat": "latitude", "lon": "longitude"},
    "oisst":    {"sst": "sst", "time": "time", "lat": "lat", "lon": "lon"},
    "currents": {"uo": "uo", "vo": "vo", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
    "sla":      {"sla": "sla", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
    "sss":      {"sos": "sos", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
    "thetao":   {"thetao": "thetao", "time": "time",
                 "lat": "latitude", "lon": "longitude"},
}
