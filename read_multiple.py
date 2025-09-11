import xarray as xr
import pandas as pd
from pathlib import Path

metadata_list = []
profile_list = []

data_dir = Path(r"C:\Users\alexe\Desktop\argo-floatchat\argo_data")

for nc_file in data_dir.glob("*.nc"):
    ds = xr.open_dataset(nc_file)
    print(f"Processing {nc_file.name} with shape {ds.dims}")

    # Handle case-insensitive variable names
    var_names = {name.lower(): name for name in ds.variables.keys()}
    
    # Get platform_number variable name
    platform_var = var_names.get("platform_number")
    cycle_var = var_names.get("cycle_number")
    lat_var = var_names.get("latitude")
    lon_var = var_names.get("longitude")
    juld_var = var_names.get("juld")

    # Metadata
    metadata_df = pd.DataFrame({
        "float_id": ds.variables[platform_var][:].astype(str),
        "cycle_number": ds.variables[cycle_var][:],
        "latitude": ds.variables[lat_var][:],
        "longitude": ds.variables[lon_var][:],
        "datetime": pd.to_datetime(ds[juld_var].values)  # Already datetime64
    })

        # Handle case-insensitive variable names
    var_names = {name.lower(): name for name in ds.variables.keys()}
    
    # Get platform_number variable name
    platform_var = var_names.get("platform_number")
    cycle_var = var_names.get("cycle_number")
    lat_var = var_names.get("latitude")
    lon_var = var_names.get("longitude")
    juld_var = var_names.get("juld")


    # -------------------------------
    # PROFILE DATA (case-insensitive)
    # -------------------------------
    pres_var = var_names.get("pres") or var_names.get("pressure")
    temp_var = var_names.get("temp") or var_names.get("temperature")
    psal_var = var_names.get("psal") or var_names.get("salinity")

    pres_qc_var = var_names.get("pres_qc")
    temp_qc_var = var_names.get("temp_qc")
    psal_qc_var = var_names.get("psal_qc")

    profile_df = pd.DataFrame({
        "pressure": ds[pres_var].values.flatten() if pres_var else None,
        "temperature": ds[temp_var].values.flatten() if temp_var else None,
        "salinity": ds[psal_var].values.flatten() if psal_var else None,
        "pres_qc": ds[pres_qc_var].values.flatten() if pres_qc_var else None,
        "temp_qc": ds[temp_qc_var].values.flatten() if temp_qc_var else None,
        "sal_qc": ds[psal_qc_var].values.flatten() if psal_qc_var else None,
        "float_id": metadata_df["float_id"].repeat(ds[pres_var].shape[1]).values if pres_var else None
    })
    
    # Append to lists
    metadata_list.append(metadata_df)
    profile_list.append(profile_df)

# Combine all files into single DataFrames
metadata_df = pd.concat(metadata_list, ignore_index=True)
profile_df = pd.concat(profile_list, ignore_index=True)

print("Metadata shape:", metadata_df.shape)
print("Profile shape:", profile_df.shape)

# Save to CSV
metadata_df.to_csv("argo_metadata.csv", index=False)
profile_df.to_csv("argo_profiles.csv", index=False)
print("✅ Data saved to argo_metadata.csv and argo_profiles.csv")