import xarray as xr

# Load NetCDF file
ds = xr.open_dataset(r'C:\Users\alexe\Desktop\argo-floatchat\argo_data\nodc_D1900270_309.nc')

print(ds)  # See variables, dimensions, attributes
print(ds.variables.keys())  # List all variables

import pandas as pd
from netCDF4 import Dataset, num2date

lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]

metadata_df = pd.DataFrame({
    "float_id": ds.variables['platform_number'][:].astype(str),
    "cycle_number": ds.variables['cycle_number'][:],
    "latitude": lat,
    "longitude": lon,
    "datetime": pd.to_datetime(ds["juld"].values)  # Already datetime64
})

print(metadata_df)