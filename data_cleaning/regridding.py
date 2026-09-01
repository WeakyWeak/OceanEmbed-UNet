"""
Methods chosen :
  - ERA5:   already on target grid (after coord harmonization).  No-op.
  - OISST:  bilinear interpolation (0.125° offset correction).
  - GLORYS: conservative block-average (1/12° → 0.25°, 3× coarsening).
  - SLA:    bilinear interpolation with nearest-neighbour edge fill.
  - SSS:    bilinear interpolation with nearest-neighbour edge fill.

SLA/SSS Edge Problem
---------------------
SLA and SSS native grids run from 5.0625–24.9375 in latitude and
75.0625–99.9375 in longitude (cell centres offset by 0.0625°).
Our target grid runs 5.0–25.0 and 75.0–100.0, so the four domain
edges fall *outside* the source data range by 0.0625°.

Bilinear interpolation (scipy RegularGridInterpolator) correctly
returns NaN for target points outside the source convex hull.
This would produce 2 full NaN rows + 2 full NaN columns (360 cells).
"""
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from config import TARGET_LAT, TARGET_LON


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bilinear_with_nn_edge_fill(da: xr.DataArray) -> xr.DataArray:
    """Regrid a 2-D field via bilinear interpolation, then fill domain-edge
    NaN with nearest-neighbour extrapolation.

    Works per-timestep to keep memory bounded.  Expects da with dims
    (time, lat, lon) and finite values over ocean.
    """
    src_lat = da.lat.values.astype(np.float64)
    src_lon = da.lon.values.astype(np.float64)
    tgt_mesh = np.meshgrid(TARGET_LAT, TARGET_LON, indexing="ij")
    tgt_pts  = np.stack([tgt_mesh[0].ravel(), tgt_mesh[1].ravel()], axis=-1)

    nt = da.sizes["time"]
    out = np.empty((nt, len(TARGET_LAT), len(TARGET_LON)), dtype=np.float32)

    for t in range(nt):
        field = da.isel(time=t).values.astype(np.float64)

        # ── bilinear ─────────────────────────────────────────────────────
        interp_bl = RegularGridInterpolator(
            (src_lat, src_lon), field,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        result = interp_bl(tgt_pts).reshape(len(TARGET_LAT), len(TARGET_LON))

        # ── nearest-neighbour fill for edge NaN only ─────────────────────
        edge_nan = np.isnan(result)
        if edge_nan.any():
            interp_nn = RegularGridInterpolator(
                (src_lat, src_lon), field,
                method="nearest", bounds_error=False, fill_value=np.nan,
            )
            nn_vals = interp_nn(tgt_pts).reshape(len(TARGET_LAT), len(TARGET_LON))
            result[edge_nan] = nn_vals[edge_nan]

        out[t] = result.astype(np.float32)

    return xr.DataArray(
        out,
        dims=["time", "lat", "lon"],
        coords={"time": da.time, "lat": TARGET_LAT, "lon": TARGET_LON},
        name=da.name,
    )


def _conservative_coarsen_glorys(da: xr.DataArray) -> xr.DataArray:
    """Conservative block-average a 1/12° GLORYS field to 0.25°.

    For each 0.25° target cell, average all ~3×3 native cells whose
    centres fall within ±0.125° of the target centre.
    NaN (land) cells are excluded from the average — if ALL native cells
    are NaN, the target cell is NaN.
    """
    src_lat = da.lat.values.astype(np.float64)
    src_lon = da.lon.values.astype(np.float64)

    nt = da.sizes["time"]
    out = np.empty((nt, len(TARGET_LAT), len(TARGET_LON)), dtype=np.float32)

    # Pre-compute index ranges for each target cell
    lat_slices = []
    for tlat in TARGET_LAT:
        mask = (src_lat >= tlat - 0.125) & (src_lat < tlat + 0.125)
        idx = np.where(mask)[0]
        lat_slices.append(idx)

    lon_slices = []
    for tlon in TARGET_LON:
        mask = (src_lon >= tlon - 0.125) & (src_lon < tlon + 0.125)
        idx = np.where(mask)[0]
        lon_slices.append(idx)

    for t in range(nt):
        field = da.isel(time=t).values  # (nlat_src, nlon_src)
        for i, li in enumerate(lat_slices):
            for j, lj in enumerate(lon_slices):
                if len(li) == 0 or len(lj) == 0:
                    out[t, i, j] = np.nan
                    continue
                patch = field[li[0]:li[-1]+1, lj[0]:lj[-1]+1]
                valid = patch[np.isfinite(patch)]
                out[t, i, j] = valid.mean() if len(valid) > 0 else np.nan

    return xr.DataArray(
        out,
        dims=["time", "lat", "lon"],
        coords={"time": da.time, "lat": TARGET_LAT, "lon": TARGET_LON},
        name=da.name,
    )


# ── Public per-dataset regridders ────────────────────────────────────────────

def regrid_era5(u10: xr.DataArray, v10: xr.DataArray):
    """ERA5 is already on the target grid (after coord harmonization).
    Just assign the canonical coordinate arrays so merges align exactly.
    """
    print("[4] Regridding ERA5 → no-op (already 0.25°)")
    u10 = u10.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)
    v10 = v10.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)
    return u10, v10


def regrid_oisst(sst: xr.DataArray) -> xr.DataArray:
    """OISST: bilinear interpolation from 0.125°-offset grid.

    OISST is already 0.25° but with centres at 5.125, 5.375, …
    The shift is 0.125° — bilinear with NN edge fill handles it.
    """
    print("[4] Regridding OISST (bilinear + NN edge fill) …")
    sst_out = _bilinear_with_nn_edge_fill(sst.compute())
    sst_out.name = "sst"
    return sst_out


def regrid_currents(uo: xr.DataArray, vo: xr.DataArray):
    """GLORYS surface currents: conservative 1/12° → 0.25°."""
    print("[4] Regridding GLORYS currents (conservative) …")
    uo_out = _conservative_coarsen_glorys(uo.compute())
    uo_out.name = "uo"
    vo_out = _conservative_coarsen_glorys(vo.compute())
    vo_out.name = "vo"
    return uo_out, vo_out


def regrid_sla(sla: xr.DataArray) -> xr.DataArray:
    """SLA: bilinear + NN edge fill from 0.125° offset grid."""
    print("[4] Regridding SLA (bilinear + NN edge fill) …")
    sla_out = _bilinear_with_nn_edge_fill(sla.compute())
    sla_out.name = "sla"
    return sla_out


def regrid_sss(sss: xr.DataArray) -> xr.DataArray:
    """SSS: bilinear + NN edge fill from 0.125° offset grid."""
    print("[4] Regridding SSS (bilinear + NN edge fill) …")
    sss_out = _bilinear_with_nn_edge_fill(sss.compute())
    sss_out.name = "sss"
    return sss_out


def regrid_thetao(thetao_dict: dict) -> dict:
    """GLORYS thetao (all 15 depths): conservative 1/12° → 0.25°."""
    print("[4] Regridding GLORYS thetao (conservative, 15 depths) …")
    result = {}
    for label, da in thetao_dict.items():
        print(f"     → {label}")
        da_out = _conservative_coarsen_glorys(da.compute())
        da_out.name = f"thetao_{label}"
        result[label] = da_out
    return result
