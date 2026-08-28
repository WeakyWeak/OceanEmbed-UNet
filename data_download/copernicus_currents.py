import os
import copernicusmarine
from dotenv import load_dotenv

load_dotenv()
user = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
password = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/COPERNICUS_CURRENTS'))
os.makedirs(RAW_DIR, exist_ok=True)

print("Resizing on the server side...")

try:
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"], 
        minimum_longitude=75.0, maximum_longitude=100.0,
        minimum_latitude=5.0, maximum_latitude=25.0,
        start_datetime="2015-01-01T00:00:00",
        end_datetime="2022-12-31T23:59:59",
        # STRICT BOUNDARY FIX:
        minimum_depth=0.494025, 
        maximum_depth=0.495, 
        coordinates_selection_method="nearest", 
        output_filename=os.path.join(RAW_DIR, "glorys_surface_currents_2015_2022.nc"),
        overwrite=True,
        username=user,
        password=password
    )
    print("SUCCESS! Surface Currents downloaded.")
except Exception as e:
    print(f"Failed to download currents: {e}")
