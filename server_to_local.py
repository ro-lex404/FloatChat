import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ----------------- Folder for raw .nc files -----------------
raw_nc_folder = r"C:\Users\aim\Desktop\copy sih\sih divy/nc_Argo"
os.makedirs(raw_nc_folder, exist_ok=True)

# ----------------- Function to download -----------------
def download_file(url, folder):
    file_name = url.split('/')[-1]
    file_path = os.path.join(folder, file_name)

    if os.path.exists(file_path):
        print(f"Skipped (already exists): {file_name}")
        return

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {file_path}")
    except Exception as e:
        print(f"Failed to download {file_name}: {e}")

# ----------------- Years and Months -----------------
years = ["2020"]
months = [f"{i:02d}" for i in range(1, 13)]  # 01 to 12

# ----------------- Download .nc files -----------------
for year in years:
    for month in months:
        print(f"\nProcessing Year: {year}, Month: {month}")
        folder_url = f"https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/{year}/{month}/"

        try:
            r = requests.get(folder_url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Find all .nc files
            links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(".nc")]

            if not links:
                print(f"No files found for {year}/{month}")
                continue

            # Download each file
            for link in links:
                file_url = urljoin(folder_url, link)
                download_file(file_url, raw_nc_folder)

        except Exception as e:
            print(f"Failed for {year}/{month}: {e}")

print("All downloads complete. Raw .nc files saved in:", raw_nc_folder)
