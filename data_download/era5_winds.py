import cdsapi
import os
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/Copernicus_ERA5'))
os.makedirs(RAW_DIR, exist_ok=True)

c = cdsapi.Client(
    url=os.getenv("CDS_URL"),
    key=os.getenv("CDS_KEY")
)

for year in range(2005, 2023):
    outfile = os.path.join(RAW_DIR, f'era5_winds_{year}.nc')
    if os.path.exists(outfile):
        print(f"Skipping {year}, already exists.")
        continue
        
    print(f"Downloading ERA5 Winds for {year}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                'year': str(year),
                'month': [f"{m:02d}" for m in range(1, 13)],
                'day': [f"{d:02d}" for d in range(1, 32)],
                # Download all 24 hours to average into a daily mean later
                'time': [f"{h:02d}:00" for h in range(0, 24)], 
                'area': [25.0, 75.0, 5.0, 100.0], # N, W, S, E
            },
            outfile
        )
    except Exception as e:
        print(f"Failed {year}: {e}")
