"""
Two loading modes:
  - load_year_*()  →  load ONE year's data (used by year-by-year pipeline)
  - The helpers below can also be used to load mask-generation data.

Each loader returns a lazy xarray DataArray/Dataset with:
  - coords renamed to (time, lat, lon)
  - lat sorted ascending
  - depth squeezed where applicable
"""
import xarray as xr
import numpy as np
from config import (
    VARS, TARGET_DEPTHS, DEPTH_LABELS,
    era5_files_for_year, oisst_files_for_year, currents_files_for_year,
    sla_files_for_year, sss_files_for_year, thetao_files_for_year,
)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _rename_coords(ds, src_lat, src_lon, src_time=None):
    renames = {}
    if src_lat != "lat":
        renames[src_lat] = "lat"
    if src_lon != "lon":
        renames[src_lon] = "lon"
    if src_time and src_time != "time":
        renames[src_time] = "time"
    if renames:
        ds = ds.rename(renames)
    return ds


def _sort_lat(ds):
    if float(ds.lat.values[0]) > float(ds.lat.values[-1]):
        ds = ds.sortby("lat")
    return ds


def _squeeze_depth(da: xr.DataArray) -> xr.DataArray:
    if "depth" in da.dims:
        da = da.isel(depth=0, drop=True)
    return da


def _sel_year(da: xr.DataArray, year: int) -> xr.DataArray:
    """Select only days that belong to `year`."""
    return da.sel(time=str(year))


# ── Per-year loaders ──────────────────────────────────────────────────────────

def load_era5_year(year: int):
    """ERA5 u10/v10, one calendar year, hourly, lat ascending."""
    v = VARS["era5"]
    files = era5_files_for_year(year)
    if not files:
        raise FileNotFoundError(f"No ERA5 file for {year}")
    ds = xr.open_mfdataset(files, chunks={"valid_time": 24},
                           combine="by_coords")
    ds = _rename_coords(ds, v["lat"], v["lon"], v["time"])
    ds = _sort_lat(ds)
    return ds[v["u10"]], ds[v["v10"]]


def load_oisst_year(year: int):
    """OISST sst, one calendar year, daily, BoB-padded subset."""
    v = VARS["oisst"]
    files = oisst_files_for_year(year)
    if not files:
        raise FileNotFoundError(f"No OISST file for {year}")
    ds = xr.open_mfdataset(files, chunks={"time": 365}, combine="by_coords")
    ds = _rename_coords(ds, v["lat"], v["lon"], v["time"])
    ds = ds.sel(lat=slice(4.5, 25.5), lon=slice(74.5, 100.5))
    sst = ds[v["sst"]]
    if "zlev" in sst.dims:
        sst = sst.isel(zlev=0, drop=True)
    return sst


def load_currents_year(year: int):
    """GLORYS uo/vo, one calendar year, daily, surface-only."""
    v = VARS["currents"]
    files = currents_files_for_year(year)
    if not files:
        raise FileNotFoundError(f"No GLORYS currents file for {year}")
    ds = xr.open_mfdataset(files, chunks={"time": 365}, combine="by_coords")
    ds = _rename_coords(ds, v["lat"], v["lon"], v["time"])
    uo = _squeeze_depth(ds[v["uo"]])
    vo = _squeeze_depth(ds[v["vo"]])
    # When using the combined 2015-2022 file, select only the target year
    uo = _sel_year(uo, year)
    vo = _sel_year(vo, year)
    return uo, vo


def load_sla_year(year: int):
    """Copernicus SLA, one calendar year, daily."""
    v = VARS["sla"]
    files = sla_files_for_year(year)
    if not files:
        raise FileNotFoundError(f"No SLA file for {year}")
    ds = xr.open_mfdataset(files, chunks={"time": 365}, combine="by_coords")
    ds = _rename_coords(ds, v["lat"], v["lon"], v["time"])
    sla = ds[v["sla"]]
    return _sel_year(sla, year)


def load_sss_year(year: int):
    """Copernicus SSS, one calendar year, daily."""
    v = VARS["sss"]
    files = sss_files_for_year(year)
    if not files:
        raise FileNotFoundError(f"No SSS file for {year}")
    ds = xr.open_mfdataset(files, chunks={"time": 365}, combine="by_coords")
    ds = _rename_coords(ds, v["lat"], v["lon"], v["time"])
    sos = _squeeze_depth(ds[v["sos"]])
    return _sel_year(sos, year)


def load_thetao_year(year: int):
    """GLORYS thetao at all 15 depths, one calendar year.
    Returns dict {depth_label: DataArray} at native 1/12° resolution.
    """
    v = VARS["thetao"]
    result = {}
    for label in DEPTH_LABELS:
        files = thetao_files_for_year(label, year)
        if not files:
            raise FileNotFoundError(f"No thetao {label} file for {year}")
        ds = xr.open_mfdataset(files, chunks={"time": 365}, combine="by_coords")
        ds = _rename_coords(ds, v["lat"], v["lon"], v["time"])
        da = _squeeze_depth(ds[v["thetao"]])
        result[label] = _sel_year(da, year)
    return result
