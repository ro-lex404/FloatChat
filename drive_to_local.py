# pip install gdown
import gdown
import os
import zipfile

# Local folder to save the zip and extract
local_folder = r"C:\Users\aim\Desktop\copy sih\sih divy\nc_Argo"
os.makedirs(local_folder, exist_ok=True)

# Google Drive file ID of the zipped folder
zip_file_id = "YOUR_ZIP_FILE_ID"  # Replace with your Drive zip file ID
zip_path = os.path.join(local_folder, "argo_data.zip")

# Download zip
url = f"https://drive.google.com/uc?id={zip_file_id}"
gdown.download(url, zip_path, quiet=False)
print(f"Downloaded zip: {zip_path}")

# Extract zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(local_folder)
print(f"Extracted all files to: {local_folder}")
