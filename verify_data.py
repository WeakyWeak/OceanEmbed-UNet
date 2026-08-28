import os
import glob
import xarray as xr
import warnings

# Ignore xarray serialization warnings for cleaner output
warnings.filterwarnings("ignore")

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/raw'))
all_files = glob.glob(os.path.join(RAW_DIR, '**', '*.nc'), recursive=True)

print(f"Found {len(all_files)} NetCDF files. Starting Health & Depth Check...\n")
print("="*60)

corrupted_files = []

for file_path in all_files:
    file_name = os.path.basename(file_path)
    print(f"Checking: {file_name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
            lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
            time_name = 'time'
            
            print(f"  - Status: HEALTHY \u2705")
            
            # --- THE NEW DEPTH CHECKER ---
            depth_name = None
            if 'depth' in ds.coords: depth_name = 'depth'
            elif 'z' in ds.coords: depth_name = 'z'
            
            if depth_name:
                depths = ds[depth_name].values
                # If it's a single depth, print it clearly. If multiple, print min/max.
                if len(depths) == 1:
                    print(f"  - Depth Layer: {depths[0]:.3f} meters")
                else:
                    print(f"  - Depth Range: {depths.min():.3f}m to {depths.max():.3f}m ({len(depths)} levels)")
            else:
                print(f"  - Depth Layer: Surface Only (2D)")
            # -----------------------------
            
            if time_name in ds.coords:
                times = ds[time_name].values
                print(f"  - Time Range: {str(times[0])[:10]} to {str(times[-1])[:10]} ({len(times)} steps)")
            
            if lat_name in ds.coords and lon_name in ds.coords:
                lats = ds[lat_name].values
                lons = ds[lon_name].values
                print(f"  - Lat Range: {lats.min():.2f} to {lats.max():.2f}")
                print(f"  - Lon Range: {lons.min():.2f} to {lons.max():.2f}")
            
            vars = [v for v in ds.data_vars]
            print(f"  - Variables: {vars}")
            
    except Exception as e:
        print(f"  - Status: CORRUPTED OR UNREADABLE \u274C")
        print(f"  - Error: {e}")
        corrupted_files.append(file_name)
        
    print("-" * 60)

print("\n=== VERIFICATION COMPLETE ===")
