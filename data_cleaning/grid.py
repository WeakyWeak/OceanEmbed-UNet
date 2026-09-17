from __future__ import annotations

import datetime as _dt

import numpy as np

# ── target grid ──────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 5.0, 25.0
LON_MIN, LON_MAX = 50.0, 100.0
RESOLUTION = 0.25

TARGET_LAT = np.arange(LAT_MIN, LAT_MAX + RESOLUTION / 2, RESOLUTION)   # 81
TARGET_LON = np.arange(LON_MIN, LON_MAX + RESOLUTION / 2, RESOLUTION)   # 201
N_LAT, N_LON = len(TARGET_LAT), len(TARGET_LON)

# ── download box ─────────────────────────────────────────────────────────────
DOWNLOAD_PAD = 0.2
DL_LAT_MIN, DL_LAT_MAX = LAT_MIN - DOWNLOAD_PAD, LAT_MAX + DOWNLOAD_PAD
DL_LON_MIN, DL_LON_MAX = LON_MIN - DOWNLOAD_PAD, LON_MAX + DOWNLOAD_PAD

# ── record period ────────────────────────────────────────────────────────────
RECORD_START = _dt.date(1993, 1, 1)
RECORD_END = _dt.date(2024, 12, 15)          # Multi-Obs SSS ends here
YEARS = list(range(RECORD_START.year, RECORD_END.year + 1))   # 1993..2024

# Per-product coverage, verified against copernicusmarine.describe on
# 2026-09-09. Used to compute how many days each file should contain, so a
# short download is caught instead of silently shrinking the record.
PRODUCT_END = {
    "glorys": _dt.date(2026, 6, 23),
    "sla":    _dt.date(2026, 1, 16),
    "sss":    _dt.date(2024, 12, 15),
    "era5":   _dt.date(2026, 6, 30),
    "oisst":  _dt.date(2026, 6, 30),
}
PRODUCT_START = {k: _dt.date(1993, 1, 1) for k in PRODUCT_END}
PRODUCT_START["oisst"] = _dt.date(1981, 9, 1)
PRODUCT_START["era5"] = _dt.date(1940, 1, 1)

# ── depth levels (nominal labels; GLORYS snaps to its own native z) ──────────
SIH_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
SURFACE_DEPTH = 0.494025          # the GLORYS level that "0 m" resolves to


def year_bounds(year: int, product: str) -> tuple[_dt.date, _dt.date]:
    """First and last day of `year` that `product` actually covers."""
    lo = max(_dt.date(year, 1, 1), PRODUCT_START[product], RECORD_START)
    hi = min(_dt.date(year, 12, 31), PRODUCT_END[product], RECORD_END)
    return lo, hi


def expected_days(year: int, product: str) -> int:
    """How many daily steps a file for this year/product should contain.

    Returns 0 when the product does not cover the year at all. 2024 is the
    case that matters: SSS stops on 12-15, so its 2024 file holds 350 days,
    not 366 - and a checker that assumes a full year would either reject a
    good file or, worse, be loosened until it accepts a bad one.
    """
    lo, hi = year_bounds(year, product)
    return 0 if hi < lo else (hi - lo).days + 1


def describe() -> str:
    total = sum(expected_days(y, "sss") for y in YEARS)
    return (f"grid {N_LAT} x {N_LON} at {RESOLUTION} deg  "
            f"[{LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E]\n"
            f"download box padded by {DOWNLOAD_PAD} deg: "
            f"[{DL_LAT_MIN}-{DL_LAT_MAX} N, {DL_LON_MIN}-{DL_LON_MAX} E]\n"
            f"record {RECORD_START} .. {RECORD_END}  "
            f"({len(YEARS)} years, {total} days on the SSS-limited intersection)")


if __name__ == "__main__":
    print(describe())
