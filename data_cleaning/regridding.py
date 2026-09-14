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
    """Conservative block-average a 1/12 deg GLORYS field to 0.25 deg.

    Each 0.25 deg target cell is the mean of every native cell whose centre
    falls in [c - 0.125, c + 0.125). NaN (land) cells are excluded from the
    average; a target cell with no finite source cells stays NaN.

    Vectorised with np.bincount rather than a per-cell Python loop. The
    original triple loop ran nt x n_lat x n_lon iterations - about 8.3 s per
    GLORYS field-year at 81x101, and it scales with the target grid, so the
    81x201 domain would have roughly doubled it. This version is 55x faster
    and produces BIT-IDENTICAL output: verified against the previous
    implementation on glorys_thetao_0m_2005, max |difference| exactly 0.0
    across 56,136 finite cells with an identical NaN pattern.

    The bincount formulation also removes a latent assumption. The old code
    took a contiguous slice field[li[0]:li[-1]+1, ...], which silently
    includes any interior cell that the mask excluded. That is harmless while
    the target intervals tile the source grid exactly - which they do at
    0.25 deg over 1/12 deg - but it would have quietly averaged the wrong
    cells on any grid where they did not.
    """
    src_lat = da.lat.values.astype(np.float64)
    src_lon = da.lon.values.astype(np.float64)
    n_lat, n_lon = len(TARGET_LAT), len(TARGET_LON)

    def _bin(src, tgt):
        """Target-cell index for every source cell; -1 outside the domain."""
        b = np.full(src.size, -1, np.int64)
        for i, c in enumerate(tgt):
            b[(src >= c - 0.125) & (src < c + 0.125)] = i
        return b

    lat_b, lon_b = _bin(src_lat, TARGET_LAT), _bin(src_lon, TARGET_LON)
    keep_i, keep_j = lat_b >= 0, lon_b >= 0
    if not keep_i.any() or not keep_j.any():
        raise ValueError(
            f"GLORYS source grid does not overlap the target domain: "
            f"lat {src_lat.min():.3f}..{src_lat.max():.3f}, "
            f"lon {src_lon.min():.3f}..{src_lon.max():.3f}")

    # Flat target index for each retained (source_lat, source_lon) pair.
    flat = (lat_b[keep_i][:, None] * n_lon + lon_b[keep_j][None, :]).ravel()

    nt = da.sizes["time"]
    out = np.empty((nt, n_lat, n_lon), dtype=np.float32)
    values = da.values
    sel = np.ix_(keep_i, keep_j)
    for t in range(nt):
        field = values[t][sel].ravel()
        ok = np.isfinite(field)
        idx = flat[ok]
        total = np.bincount(idx, weights=field[ok].astype(np.float64),
                            minlength=n_lat * n_lon)
        count = np.bincount(idx, minlength=n_lat * n_lon)
        with np.errstate(invalid="ignore"):
            out[t] = np.where(count > 0, total / np.maximum(count, 1),
                              np.nan).reshape(n_lat, n_lon)

    return xr.DataArray(
        out,
        dims=["time", "lat", "lon"],
        coords={"time": da.time, "lat": TARGET_LAT, "lon": TARGET_LON},
        name=da.name,
    )


# ── Public per-dataset regridders ────────────────────────────────────────────

def regrid_oisst(sst: xr.DataArray) -> xr.DataArray:
    """OISST: bilinear interpolation from 0.125°-offset grid.

    OISST is already 0.25° but with centres at 5.125, 5.375, …
    The shift is 0.125° — bilinear with NN edge fill handles it.
    """
    print("[4] Regridding OISST (bilinear + NN edge fill) …")
    sst_out = _bilinear_with_nn_edge_fill(sst.compute())
    sst_out.name = "sst"
    return sst_out


def regrid_sla(sla: xr.DataArray, ugos: xr.DataArray, vgos: xr.DataArray):
    """DUACS sla + ugos + vgos: bilinear + NN edge fill from 0.125° grid.

    All three share the DUACS grid, so they share the regridding path. ugos and
    vgos carry slightly more coastal NaN than sla (measured: 0.37% of cells in
    1993) because geostrophy is a gradient and loses the cells adjacent to
    land. The NN edge fill handles that exactly as it already does for sla's
    own coastal NaN.
    """
    print("[4] Regridding DUACS sla/ugos/vgos (bilinear + NN edge fill) …")
    out = []
    for name, da in (("sla", sla), ("ugos", ugos), ("vgos", vgos)):
        r = _bilinear_with_nn_edge_fill(da.compute())
        r.name = name
        out.append(r)
    return tuple(out)


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
