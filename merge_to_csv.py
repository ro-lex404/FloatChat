import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
import logging

# --- Configuration ---
NC_FILES_DIRECTORY = './argo_data_2013_09/'
OUTPUT_CSV_FILE = 'argo_metadata1.csv'

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_nc_file(file_path):
    """
    Reads a single ARGO NetCDF file and extracts relevant data into a pandas DataFrame.
    This version dynamically handles data structures and date formats.
    """
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds: # decode_times=False gives us raw values
            if 'n_prof' not in ds.sizes:
                logging.warning(f"Skipping file {os.path.basename(file_path)}: 'n_prof' dimension not found.")
                return None
            num_profiles = ds.sizes['n_prof']

            platform_data = ds['platform_number'].values[0]
            float_id = platform_data.decode('utf-8').strip() if isinstance(platform_data, bytes) else str(platform_data).strip()

            cycle_numbers = ds['cycle_number'].values
            latitudes = ds['latitude'].values
            longitudes = ds['longitude'].values

            ## --- FIX: Robust Datetime Conversion ---
            # Check the data type of the 'juld' variable to handle different formats.
            juld_values = ds['juld'].values
            if np.issubdtype(juld_values.dtype, np.number):
                # If it's a number, it's a Julian day offset. Use the origin.
                datetimes = pd.to_datetime(juld_values, origin='1950-01-01', unit='D')
            else:
                # If it's not a number, it's likely a datetime string or object.
                # Let pandas parse it directly without an origin.
                datetimes = pd.to_datetime(juld_values)
            ## --- END OF FIX ---

            pressures = ds['pres'].values if 'pres' in ds else None
            temperatures = ds['temp'].values if 'temp' in ds else None
            salinities = ds['psal'].values if 'psal' in ds else None
            
            pressure_qc = ds['pres_qc'].values if 'pres_qc' in ds else None
            temperature_qc = ds['temp_qc'].values if 'temp_qc' in ds else None
            salinity_qc = ds['psal_qc'].values if 'sal_qc' in ds else None

            records = []
            for i in range(num_profiles):
                if pressures is not None:
                    num_measurements = pressures.shape[1] if pressures.ndim == 2 else pressures.shape[0]
                else:
                    logging.warning(f"No pressure data in profile {i} for file {os.path.basename(file_path)}. Skipping profile.")
                    continue
                
                for j in range(num_measurements):
                    def get_value(array, prof_idx, meas_idx):
                        if array is None: return np.nan
                        if array.ndim == 2: return array[prof_idx, meas_idx]
                        if array.ndim == 1 and prof_idx == 0: return array[meas_idx]
                        return np.nan

                    def get_qc_value(array, prof_idx, meas_idx):
                        val = get_value(array, prof_idx, meas_idx)
                        try:
                            if isinstance(val, bytes): val = val.decode('utf-8')
                            return int(val) if pd.notna(val) else 0
                        except (ValueError, TypeError):
                            return 0

                    record = {
                        'float_id': float_id,
                        'cycle_number': int(cycle_numbers[i]),
                        'latitude': latitudes[i],
                        'longitude': longitudes[i],
                        'datetime': datetimes[i],
                        'pressure': get_value(pressures, i, j),
                        'temperature': get_value(temperatures, i, j),
                        'salinity': get_value(salinities, i, j),
                        'pressure_qc': get_qc_value(pressure_qc, i, j),
                        'temperature_qc': get_qc_value(temperature_qc, i, j),
                        'salinity_qc': get_qc_value(salinity_qc, i, j)
                    }
                    records.append(record)
            
            return pd.DataFrame(records)

    except Exception as e:
        logging.error(f"Failed to process file {os.path.basename(file_path)}: {e}")
        return None

if __name__ == "__main__":
    nc_files = glob.glob(os.path.join(NC_FILES_DIRECTORY, '*.nc'))
    
    if not nc_files:
        logging.warning(f"No .nc files found in directory: {NC_FILES_DIRECTORY}")
    else:
        logging.info(f"Found {len(nc_files)} NetCDF files to process.")
        
        all_dataframes = []
        for i, file_path in enumerate(nc_files):
            logging.info(f"Processing file {i+1}/{len(nc_files)}: {os.path.basename(file_path)}")
            df = process_nc_file(file_path)
            if df is not None and not df.empty:
                all_dataframes.append(df)
                logging.info(f"  ✓ Extracted {len(df)} records")
            else:
                logging.warning(f"  ✗ No data extracted from file")

        if all_dataframes:
            logging.info("Merging all data into a single CSV file...")
            final_df = pd.concat(all_dataframes, ignore_index=True)
            
            final_df.dropna(subset=['pressure', 'temperature', 'salinity'], how='all', inplace=True)
            
            final_df.to_csv(OUTPUT_CSV_FILE, index=False)
            logging.info(f"✅ Successfully created {OUTPUT_CSV_FILE}")
            logging.info(f"   Total records: {len(final_df)}")
            logging.info(f"   Unique floats: {final_df['float_id'].nunique()}")
            logging.info(f"   Date range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
        else:
            logging.warning("No data could be extracted from the provided files.")