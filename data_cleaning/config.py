"""
OceanEmbed Data Cleaning Pipeline — Configuration
All paths, constants, grid definitions, and dataset metadata.
"""
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"    # final outputs
YEARS_DIR    = OUT_DIR / "years"                      # per-year checkpoint files

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
def era5_files_for_year(year: int):
    return sorted(ERA5_DIR.glob(f"era5_winds_{year}.nc"))

def oisst_files_for_year(year: int):
    return sorted(OISST_DIR.glob(f"sst.day.mean.{year}.nc"))

def currents_files_for_year(year: int):
    # GLORYS currents: 2005-2014 have individual files; 2015-2022 are in the combined file
    single = sorted(CURRENTS_DIR.glob(f"glorys_surface_currents_{year}.nc"))
    combined = sorted(CURRENTS_DIR.glob("glorys_surface_currents_2015_2022.nc"))
    return single if single else combined

def sla_files_for_year(year: int):
    single = sorted(MARINE_DIR.glob(f"sla_{year}.nc"))
    combined = sorted(MARINE_DIR.glob("sla_2015_2022.nc"))
    return single if single else combined

def sss_files_for_year(year: int):
    single = sorted(MARINE_DIR.glob(f"sss_{year}.nc"))
    combined = sorted(MARINE_DIR.glob("sss_2015_2022.nc"))
    return single if single else combined

def thetao_files_for_year(depth_label: str, year: int):
    single = sorted(MARINE_DIR.glob(f"glorys_thetao_{depth_label}_{year}.nc"))
    combined = sorted(MARINE_DIR.glob(f"glorys_thetao_{depth_label}_2015_2022.nc"))
    return single if single else combined


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
