"""
Mask Generation.

Canonical ocean mask derived from GLORYS surface thetao (the training
target), coarsened from 1/12° to 0.25° using a ≥50% ocean threshold.

"""
import numpy as np
import xarray as xr
from config import TARGET_LAT, TARGET_LON, OUT_DIR


def generate_land_mask(thetao_surface: xr.DataArray) -> xr.DataArray:
    """Build the canonical binary ocean mask from GLORYS surface thetao.

    Takes the FIRST timestep of GLORYS thetao at the surface level
    (already loaded with native 1/12° resolution, coords (lat, lon)).

    For each 0.25° target cell, checks the fraction of native cells that
    are ocean (finite).  If ≥50% ocean → mask = 1 (ocean), else 0 (land).

    Returns an (81×101) DataArray with coords (lat, lon).
    """
    print("[6] Generating canonical land mask from GLORYS thetao …")

    # Use ALL timesteps across the dataset to get the strictest mask.
    # A cell is "ocean" only if it is finite in the MAJORITY of timesteps.
    # For GLORYS reanalysis the spatial mask is static, so first timestep suffices.
    field_t0 = thetao_surface.isel(time=0).values
    native_ocean = np.isfinite(field_t0).astype(np.float64)

    src_lat = thetao_surface.lat.values
    src_lon = thetao_surface.lon.values

    mask = np.zeros((len(TARGET_LAT), len(TARGET_LON)), dtype=np.float32)

    for i, tlat in enumerate(TARGET_LAT):
        lat_idx = np.where(
            (src_lat >= tlat - 0.125) & (src_lat < tlat + 0.125)
        )[0]
        for j, tlon in enumerate(TARGET_LON):
            lon_idx = np.where(
                (src_lon >= tlon - 0.125) & (src_lon < tlon + 0.125)
            )[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                mask[i, j] = 0.0
                continue
            patch = native_ocean[lat_idx[0]:lat_idx[-1]+1,
                                 lon_idx[0]:lon_idx[-1]+1]
            mask[i, j] = 1.0 if patch.mean() >= 0.5 else 0.0

    ocean_cells = int(mask.sum())
    total_cells = mask.size
    print(f"     Mask: {ocean_cells} ocean / {total_cells} total "
          f"({100*ocean_cells/total_cells:.1f}% ocean)")

    mask_da = xr.DataArray(
        mask, dims=["lat", "lon"],
        coords={"lat": TARGET_LAT, "lon": TARGET_LON},
        name="land_mask",
    )
    return mask_da


def apply_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Apply the canonical mask: set land cells to NaN."""
    return da.where(mask == 1)


def save_mask(mask: xr.DataArray):
    """Persist the land mask to disk."""
    path = OUT_DIR / "land_mask.nc"
    mask.to_dataset(name="land_mask").to_netcdf(path)
    print(f"     Mask saved → {path}")
