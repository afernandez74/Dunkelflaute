#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:47:18 2025

@author: afer
"""

import os 
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import rankdata
from scipy import stats
#%% load 2 datasets: wind and temp files AND solar rad files

dat_path = os.path.expanduser('./../Data/ERA5/')

files = os.listdir(dat_path)

name_file = 'ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8'

file_path = dat_path + name_file

ds = xr.open_mfdataset(file_path)
u_100 = ds.u100
v_100 = ds.v100
#%% maximum coordinates in dataset

lat_min = ds['latitude'].min().item()
lat_max = ds['latitude'].max().item()
lon_min = ds['longitude'].min().item()
lon_max = ds['longitude'].max().item()

plot_region = [lon_min-1, lon_max+1, lat_min-1, lat_max+1]

#%% Calculate wind power from wind speed @100m 
 
wind_100 = np.sqrt(u_100**2 + v_100**2)
# wind_100_daily = wind_100.resample(valid_time = '1D').mean()

# constants
P_r = 2000

def power_curve (ws):
    return(634.228 - 1248.5*ws + 999.57*(ws**2) - 426.224*(ws**3) +
            105.617*(ws**4) - 15.4587*(ws**5) + 1.3223*(ws**6) -
            0.0609186*(ws**7) + 0.00116265*(ws**8))

# Wind Turbine Power Curve froma V90-2.0MW Vestas Turbine (Brown et al., 2021)
WP = xr.where(wind_100 > 25, 0,
       xr.where(wind_100 > 13, P_r,
         xr.where(wind_100 < 3, 0,
           power_curve(wind_100)))) #kW

# Wind power 

# Capacity factor

CF_wind = WP / P_r


#%% save CF_wind

path_save = os.path.expanduser('./../Results/CF_wind/')
name = 'CF_wind_time_1980_2023_latlon_50_54_3_8.nc'
save_toggle = input("Save CF_wind results? (Y or N) \n ...")
if save_toggle.lower() == 'y':
    CF_wind.to_netcdf(path_save+name)


#%% Heavy calculations

#Time-averaged wind power production
WP_time_avg = WP.mean(dim='valid_time')

# daily wind power
WP_daily = WP.resample(valid_time = '1D').mean()

# CF_wind daily
CF_wind_dly = CF_wind.resample(valid_time='1D').mean()

#%% map average wind power over time period

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
    central_longitude=5, central_latitude=40
    ))

# Set the extent of the map to your region
ax.set_extent(plot_region, crs=ccrs.PlateCarree())

# Add basic map features
ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAKES, facecolor='lightblue')

# Plot your xarray data on the map
WP_time_avg.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Data Value'})

# Add a title
ax.set_title('Average hourly wind power @100m')

# Show the plot
plt.show()

