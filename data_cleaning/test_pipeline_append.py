import sys
import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# Add data_cleaning to sys path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from pipeline import _write_split, INPUT_VARS, ALL_VARS, NC_ENCODING
import pipeline

print("Creating mock datasets...")
# Setup a temp years dir
temp_years_dir = Path("temp_years")
temp_years_dir.mkdir(exist_ok=True)
out_path = Path("temp_train.nc")

# Mock YEARS_DIR in config/pipeline
pipeline.YEARS_DIR = temp_years_dir

lat = np.arange(5.0, 25.25, 0.25)
lon = np.arange(75.0, 100.25, 0.25)
rng = np.random.default_rng(42)

def make_mock_year(year):
    times = pd.date_range(f"{year}-01-01", f"{year}-01-05", freq="D")
    data_vars = {}
    for var in ALL_VARS:
        # Create random float32 data with some NaNs
        arr = rng.random((len(times), len(lat), len(lon)), dtype=np.float32)
        # Put NaNs in a specific location
        arr[:, 10:20, 10:20] = np.nan
        data_vars[var] = (["time", "lat", "lon"], arr)
        
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "lat": lat, "lon": lon}
    )
    return ds

ds_2005 = make_mock_year(2005)
ds_2006 = make_mock_year(2006)

ds_2005.to_netcdf(temp_years_dir / "2005.nc")
ds_2006.to_netcdf(temp_years_dir / "2006.nc")

# Mock stats
stats = {}
for var in ALL_VARS:
    stats[var] = {"mean": 0.5, "std": 0.2}

print("\nRunning _write_split...")
# Run _write_split
_write_split([2005, 2006], out_path, stats)

print("\nVerifying outputs...")
# Verify
res = xr.open_dataset(out_path)

print("Dimensions:", res.sizes)
assert res.sizes["time"] == 10
assert res.sizes["lat"] == 81
assert res.sizes["lon"] == 101

print("Variables:", list(res.data_vars.keys())[:3], "...", list(res.data_vars.keys())[-3:])
for var in ALL_VARS:
    assert var in res.data_vars

print("Dtype:", res["sst"].dtype)
assert res["sst"].dtype == np.float32

# Verify Zlib encoding
import netCDF4 as nc
with nc.Dataset(out_path) as ds_nc:
    sst_var = ds_nc.variables["sst"]
    filters = sst_var.filters()
    print("Filters (sst):", filters)
    assert filters is not None and filters.get("zlib") == True
    
    # Check unlimited dim
    assert ds_nc.dimensions["time"].isunlimited()
    print("Time dim is unlimited:", ds_nc.dimensions["time"].isunlimited())

# Verify NaNs and zero-fill
# INPUT_VARS should have NaNs replaced by 0.0
# TARGET_VARS should still have NaNs
input_nans = np.isnan(res["sst"].values).sum()
print("Input var (sst) NaN count:", input_nans)
assert input_nans == 0

target_nans = np.isnan(res["thetao_0m"].values).sum()
print("Target var (thetao_0m) NaN count:", target_nans)
assert target_nans > 0

# Check z-score logic on target (target wasn't zero-filled, but should be z-scored)
# original data was ~U(0,1), mean=0.5, std=0.2. So z-scored should be mean~0, std~1.4
# (we don't test precise value, just that they changed)

print("Time coordinates (first and last of each year):")
print(res.time.values[0])
print(res.time.values[4])
print(res.time.values[5])
print(res.time.values[9])
assert str(res.time.values[0])[:10] == "2005-01-01"
assert str(res.time.values[4])[:10] == "2005-01-05"
assert str(res.time.values[5])[:10] == "2006-01-01"
assert str(res.time.values[9])[:10] == "2006-01-05"

res.close()

print("\nAll verifications passed! Cleanup...")
shutil.rmtree(temp_years_dir)
out_path.unlink()
print("Done.")
