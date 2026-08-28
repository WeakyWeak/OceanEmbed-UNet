import os
import copernicusmarine
from dotenv import load_dotenv

load_dotenv()
user = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
password = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/COPERNICUS_MARINE'))
os.makedirs(RAW_DIR, exist_ok=True)

sih_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

print("Downloading GLORYS Target Depths individually...")

for depth in sih_depths:
    filename = os.path.join(RAW_DIR, f"glorys_thetao_{depth}m_2015_2022.nc")
    
    if os.path.exists(filename):
        print(f"Skipping {depth}m, already downloaded.")
        continue
        
    print(f"\n--- Downloading Depth: {depth}m ---")
    
    if depth == 0:
        req_depth = 0.494025
    else:
        req_depth = float(depth)
        
    try:
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao"],
            minimum_longitude=75.0, maximum_longitude=100.0,
            minimum_latitude=5.0, maximum_latitude=25.0,
            start_datetime="2015-01-01T00:00:00",
            end_datetime="2022-12-31T23:59:59",
            # Request exactly one depth, let 'nearest' snap it to the true level
            minimum_depth=req_depth,
            maximum_depth=req_depth,
            coordinates_selection_method="nearest",
            output_filename=filename,
            overwrite=True,
            username=user,
            password=password
        )
    except Exception as e:
        print(f"FAILED on depth {depth}m. Error: {e}")

# 2. DOWNLOAD SLA (Surface only)
print("\n--- Downloading SLA ---")
try:
    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["sla"],
        minimum_longitude=75.0, maximum_longitude=100.0,
        minimum_latitude=5.0, maximum_latitude=25.0,
        start_datetime="2015-01-01T00:00:00",
        end_datetime="2022-12-31T23:59:59",
        output_filename=os.path.join(RAW_DIR, "sla_2015_2022.nc"),
        overwrite=True,
        username=user, password=password
    )
except Exception as e:
    print(f"FAILED SLA: {e}")

# 3. DOWNLOAD SSS (Surface only)
print("\n--- Downloading SSS ---")
try:
    copernicusmarine.subset(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        variables=["sos"],
        minimum_longitude=75.0, maximum_longitude=100.0,
        minimum_latitude=5.0, maximum_latitude=25.0,
        start_datetime="2015-01-01T00:00:00",
        end_datetime="2022-12-31T23:59:59",
        output_filename=os.path.join(RAW_DIR, "sss_2015_2022.nc"),
        overwrite=True,
        username=user, password=password
    )
except Exception as e:
    print(f"FAILED SSS: {e}")
