import os
import requests
import pandas as pd
import xarray as xr
from bs4 import BeautifulSoup
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import time

# Google Drive output folder - FIXED PATH (use actual path, not URL)
CSV_ROOT = "https://drive.google.com/drive/folders/1GQe63N3loqQbsi3G6cvQwPpG5MqnT7oA?usp=drive_linka"  # Change this to your actual Google Drive path
os.makedirs(CSV_ROOT, exist_ok=True)

# THREDDS XML catalog template
CATALOG_TEMPLATE = "https://www.ncei.noaa.gov/thredds-ocean/catalog/argo/gadr/indian//{year}/{month:02d}/catalog.xml"

# Thread safety
print_lock = Lock()
error_lock = Lock()
failed_downloads = []

def safe_print(message):
    """Thread-safe printing"""
    with print_lock:
        print(message)

def list_nc_files(year, month):
    """Scrape THREDDS XML catalog and return .nc file names."""
    url = CATALOG_TEMPLATE.format(year=year, month=month)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.content, "xml")
        files = [ds["urlPath"].split("/")[-1] for ds in soup.find_all("dataset") 
                if ds.get("urlPath", "").endswith(".nc")]
        return files
    except Exception as e:
        safe_print(f"❌ Failed to list files for {year}-{month:02d}: {e}")
        return []

def process_nc_file(year, month, filename, max_retries=3):
    """Open remote NetCDF via OPeNDAP and save as CSV in Drive."""
    url = f"https://www.ncei.noaa.gov/thredds-ocean/dodsC/argo/gadr/indian/{year}/{month:02d}/{filename}"

    # Output folder structure: /CSV_Output/year/month/
    out_dir = os.path.join(CSV_ROOT, str(year), f"{month:02d}")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, filename.replace(".nc", ".csv"))

    if os.path.exists(csv_file):
        safe_print(f"⏩ Skipping {csv_file} (already exists)")
        return True

    for attempt in range(max_retries):
        try:
            # Open the remote dataset with timeout
            ds = xr.open_dataset(url, decode_times=False)
            
            # Select available variables
            available_vars = ["latitude", "longitude", "pres", "temp", "psal", "juld"]
            variables = [v for v in available_vars if v in ds.variables]
            
            if not variables:
                safe_print(f"⚠ No valid variables in {filename}")
                return False

            # Convert to DataFrame
            df = ds[variables].to_dataframe().reset_index()

            # Convert juld to time if present
            if "juld" in df.columns:
                if pd.api.types.is_numeric_dtype(df["juld"]):
                    df["time"] = pd.to_datetime(df["juld"], unit="D", origin=pd.Timestamp("1950-01-01"))
                else:
                    df["time"] = pd.to_datetime(df["juld"])
                df = df.drop(columns=["juld"])

            # Save to CSV
            df.to_csv(csv_file, index=False)
            safe_print(f"✅ Saved {csv_file} ({len(df)} rows)")
            return True

        except Exception as e:
            if attempt == max_retries - 1:
                error_msg = f"❌ Failed {filename} after {max_retries} attempts: {e}"
                safe_print(error_msg)
                with error_lock:
                    failed_downloads.append((year, month, filename, str(e)))
                return False
            time.sleep(2 ** attempt)  # Exponential backoff

def process_year_month(year_month):
    """Process all files for a specific year-month combination"""
    year, month = year_month
    nc_files = list_nc_files(year, month)
    
    if not nc_files:
        safe_print(f"⚠ No files for {year}-{month:02d}")
        return 0
    
    safe_print(f"\n📂 Processing {year}-{month:02d} ({len(nc_files)} files)")
    
    successful = 0
    # Process files in parallel for this month
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Create futures for all files in this month
        futures = {
            executor.submit(process_nc_file, year, month, f): f 
            for f in nc_files
        }
        
        # Monitor progress with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(nc_files), 
                          desc=f"{year}-{month:02d}"):
            if future.result():
                successful += 1
    
    return successful

def main():
    """Main function with parallel execution"""
    # Generate all year-month combinations
    years = range(1999, 2021)   # 1999–2020
    months = range(1, 13)
    year_months = [(year, month) for year in years for month in months]
    
    total_files_processed = 0
    total_files_successful = 0
    
    print(f"🚀 Starting parallel download of ARGO data (1999-2020)")
    print(f"📁 Output directory: {CSV_ROOT}")
    print(f"🧵 Using 8 parallel workers per month")
    
    start_time = time.time()
    
    # Process each year-month in sequence, but files within each month in parallel
    for year_month in year_months:
        year, month = year_month
        successful = process_year_month(year_month)
        total_files_successful += successful
        total_files_processed += len(list_nc_files(year, month)) or 0
    
    end_time = time.time()
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Total files processed: {total_files_processed}")
    print(f"Successfully downloaded: {total_files_successful}")
    print(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"\n❌ Failed downloads (saved for retry):")
        for year, month, filename, error in failed_downloads[:5]:  # Show first 5
            print(f"  {year}-{month:02d}/{filename}: {error}")
        if len(failed_downloads) > 5:
            print(f"  ... and {len(failed_downloads) - 5} more")
        
        # Save failed downloads to file for retry
        failed_df = pd.DataFrame(failed_downloads, 
                               columns=['year', 'month', 'filename', 'error'])
        failed_df.to_csv(os.path.join(CSV_ROOT, 'failed_downloads.csv'), index=False)
        print(f"💾 Failed downloads saved to: {os.path.join(CSV_ROOT, 'failed_downloads.csv')}")

if __name__ == "__main__":
    main()