import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Base URL for the month you want
BASE_URL = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/2013/10/"

# Local folder to save files
SAVE_DIR = "argo_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Get HTML page
response = requests.get(BASE_URL)
response.raise_for_status()

# Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# Find all .nc file links
nc_files = [urljoin(BASE_URL, a['href']) for a in soup.find_all('a', href=True) if a['href'].endswith(".nc")]

print(f"📂 Found {len(nc_files)} files.")

# Function to download a single file
def download_file(url):
    filename = os.path.join(SAVE_DIR, url.split("/")[-1])
    if not os.path.exists(filename):  # Skip if already downloaded
        try:
            print(f"⬇️ Downloading {filename} ...")
            file_data = requests.get(url, timeout=30)
            file_data.raise_for_status()
            with open(filename, "wb") as f:
                f.write(file_data.content)
            print(f"✅ Done: {filename}")
        except Exception as e:
            print(f"❌ Failed: {url} — {e}")

# Run downloads in parallel
MAX_WORKERS = 10  # Adjust based on your internet speed
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(download_file, nc_files)

print("🎉 All downloads completed.")
