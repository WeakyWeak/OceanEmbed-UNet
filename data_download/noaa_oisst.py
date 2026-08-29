import requests
import os
import time

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/NOAA_OISST'))
os.makedirs(RAW_DIR, exist_ok=True)

for year in range(2005, 2023):
    url = f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    final_filename = os.path.join(RAW_DIR, f"sst.day.mean.{year}.nc")
    tmp_filename = os.path.join(RAW_DIR, f"sst.day.mean.{year}.nc.tmp")
    
    if os.path.exists(final_filename):
        print(f"Skipping {year}, file already exists.")
        continue
        
    print(f"\n--- Downloading OISST for {year} ---")
    
    max_retries = 20
    for attempt in range(max_retries):
        # 1. Check how much we already downloaded!
        downloaded_bytes = 0
        if os.path.exists(tmp_filename):
            downloaded_bytes = os.path.getsize(tmp_filename)
            if downloaded_bytes > 0:
                print(f"Wi-Fi drop detected. Resuming from {downloaded_bytes / (1024*1024):.1f} MB...")

        # 2. Tell the NOAA server to ONLY send the remaining bytes
        headers = {"Range": f"bytes={downloaded_bytes}-"}
        
        try:
            # Increased timeout to 60 seconds to survive lag spikes
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            
            # If the server says 416, it means we actually finished downloading it!
            if response.status_code == 416:
                os.rename(tmp_filename, final_filename)
                print(f"Finished {year}!")
                break
                
            response.raise_for_status() 
            
            # 3. Open in 'ab' (Append Binary) mode so we don't overwrite our progress
            with open(tmp_filename, 'ab') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 4. If we get here without crashing, the file is 100% done!
            os.rename(tmp_filename, final_filename)
            print(f"Finished {year}!")
            break # Break out of the retry loop and move to the next year
            
        except Exception as e:
            print(f"Connection dropped (Attempt {attempt+1}/{max_retries}). Retrying in 5 seconds...")
            time.sleep(5) 
    else:
        print(f"Failed to download {year} after {max_retries} attempts. We'll try again later.")
