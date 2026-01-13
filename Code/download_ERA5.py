import cdsapi
import os
#%% request for 2mtemp pressure wind @10m and @100m geopotential

year_range = range(2023,2024)
dataset = "reanalysis-era5-single-levels"

# loop over months
for year_i in year_range: 
    for month_i in range(1,13):
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "100m_u_component_of_wind",
                "100m_v_component_of_wind"
            ],
            "year": int(year_i),
            "month": f"{month_i:02d}",
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [54, 3, 50, 8]
        }

    
        client = cdsapi.Client()
        #output file name
        download_path = os.path.expanduser('./../Data/ERA5/downloads/')
        name = f"{download_path}dat_{year_i}_{month_i}.nc"
        client.retrieve(dataset, request).download(target=name)
# still need radiation!!!
#%% download radiation
year_range = range(2018,2024)
dataset = "reanalysis-era5-single-levels"

# loop over months
for year_i in year_range: 
    for month_i in range(1,13):
        request = {
            "product_type": ["reanalysis"],
            "variable": ["surface_solar_radiation_downwards"],
            "year": int(year_i),
            "month": f"{month_i:02d}",
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [54, 3, 50, 8]
        }

    
        client = cdsapi.Client()
        #output file name
        download_path = os.path.expanduser('./../Data/ERA5/downloads/')
        name = f"{download_path}dat_{year_i}_{month_i}_ssrd.nc"
        client.retrieve(dataset, request).download(target=name)

#%% download only 10m wind speeds

year_range = range(2013, 2024)
dataset = "reanalysis-era5-single-levels"

# Full list of days and times for a complete month/year
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)] # 01 to 31
ALL_TIMES = [f"{h:02d}:00" for h in range(0, 24)] # 00:00 to 23:00
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind"
]
AREA = [54, 3, 50, 8] # [N, W, S, E]

download_path = os.path.expanduser('./../Data/ERA5/downloads/')
os.makedirs(download_path, exist_ok=True)

client = cdsapi.Client()

# Loop over years
for year_i in year_range:
    # Loop over variables
    for variable_i in VARIABLES:
        request = {
            "product_type": ["reanalysis"],
            "variable": [variable_i], # Request one variable at a time
            "year": str(year_i), # CDS API often prefers strings for year/month
            "month": [f"{m:02d}" for m in range(1, 13)], # Request ALL months (01 to 12)
            "day": ALL_DAYS, # Request ALL days (01 to 31)
            "time": ALL_TIMES,
            "data_format": "netcdf",
            "area": AREA
        }

        # Output file name: include year and variable name
        # Replacing spaces and parentheses in variable name for a clean filename
        var_name_short = variable_i.replace('10m_', '').replace('_of_wind', '').replace('_component', '').replace('(', '').replace(')', '')
        name = f"{download_path}dat_{year_i}_{var_name_short}.nc"

        print(f"Requesting: {variable_i} for year {year_i}")
        try:
            client.retrieve(dataset, request).download(target=name)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            # Consider adding logic here to retry or log the failed request
#%%
year_range = range(2050, 2051)

dataset = "projections-cmip6"
model = "mri_esm2_0"
scenario = "ssp3_7_0"

# Full list of days and times for a complete month/year
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)] # 01 to 31
ALL_TIMES = [f"{h:02d}:00" for h in range(0, 24)] # 00:00 to 23:00
VARIABLES = [
    "100m_wind_speed",
    "2m_temperature",
    "surface_solar_radiation_downwards"
]
AREA = [54, 3, 50, 8] # [N, W, S, E]

download_path = os.path.expanduser('./../Data/ERA5/downloads/')
os.makedirs(download_path, exist_ok=True)

client = cdsapi.Client()

# Loop over years
for year_i in year_range:
    # Loop over variables
    for variable_i in VARIABLES:
        request = {
            "pecd_version": "pecd4_2",
            "temporal_period": ["future_projections"],
            "origin": model,
            "emission_scenario": scenario,
            "variable": [variable_i], # Request one variable at a time
            "year": str(year_i), # CDS API often prefers strings for year/month
            "month": [f"{m:02d}" for m in range(1, 13)], # Request ALL months (01 to 12)
            "day": ALL_DAYS, # Request ALL days (01 to 31)
            "time": ALL_TIMES,
            "data_format": "netcdf",
            "area": AREA
        }

        # Output file name: include year and variable name
        # Replacing spaces and parentheses in variable name for a clean filename
        var_name_short = variable_i.replace('10m_', '').replace('_of_wind', '').replace('_component', '').replace('(', '').replace(')', '')
        name = f"{download_path}dat_{year_i}_{var_name_short}.nc"

        print(f"Requesting: {variable_i} for year {year_i}")
        try:
            client.retrieve(dataset, request).download(target=name)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            # Consider adding logic here to retry or log the failed request
#%% Download soil moisture

year_range = range(2003, 2024)
dataset = "reanalysis-era5-single-levels"

# Loop over months
for year_i in year_range: 
    for month_i in range(1, 13):
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "volumetric_soil_water_layer_1",  # top soil moisture layer (0–7 cm)
            ],
            "year": int(year_i),
            "month": f"{month_i:02d}",
            "day": [
                "01", "02", "03", "04", "05", "06", "07",
                "08", "09", "10", "11", "12", "13", "14",
                "15", "16", "17", "18", "19", "20", "21",
                "22", "23", "24", "25", "26", "27", "28",
                "29", "30", "31"
            ],
            "time": [
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [54, 3, 50, 8],  # [N, W, S, E] — roughly Netherlands region
        }

        client = cdsapi.Client()
        # Output file name
        download_path = os.path.expanduser('./../Data/ERA5/downloads/')
        os.makedirs(download_path, exist_ok=True)
        name = f"{download_path}soil_moisture_{year_i}_{month_i:02d}.nc"
        client.retrieve(dataset, request).download(target=name)

#%% Download dewpoint temperature

year_range = range(1980, 2024)
dataset = "reanalysis-era5-single-levels"

# Loop over months
for year_i in year_range: 
    for month_i in range(1, 13):
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_dewpoint_temperature",  # top soil moisture layer (0–7 cm)
            ],
            "year": int(year_i),
            "month": f"{month_i:02d}",
            "day": [
                "01", "02", "03", "04", "05", "06", "07",
                "08", "09", "10", "11", "12", "13", "14",
                "15", "16", "17", "18", "19", "20", "21",
                "22", "23", "24", "25", "26", "27", "28",
                "29", "30", "31"
            ],
            "time": [
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [54, 3, 50, 8],  # [N, W, S, E] — roughly Netherlands region
        }

        client = cdsapi.Client()
        # Output file name
        download_path = os.path.expanduser('./../Data/ERA5/downloads/')
        os.makedirs(download_path, exist_ok=True)
        name = f"{download_path}2m_dwpt_t_{year_i}_{month_i:02d}.nc"
        client.retrieve(dataset, request).download(target=name)
