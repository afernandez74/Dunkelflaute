#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:04:03 2025

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
#%%

CF_wind_path = os.path.expanduser('./../Results/CF_wind/')
CF_solar_path = os.path.expanduser('./../Results/CF_solar/')

CF_wind_file = os.listdir(CF_wind_path)

CF_wind = xr.open_dataarray(CF_wind_path+CF_wind_file[0])

CF_solar_file = os.listdir(CF_solar_path)

CF_solar = xr.open_dataarray(CF_solar_path+CF_solar_file[0])

#%% daily timestep of CF values

# daily mean of CF wind
CF_wind_dly = CF_wind.resample(valid_time = "1D").mean()

# daily mean of CF solar daytime hours
CF_daylight = CF_solar.where(CF_solar > 0)
CF_solar_dly = CF_daylight.resample(valid_time = '1D').mean()

#%% Histogram of daily CF values 

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(CF_wind_dly.values.flatten(), bins=50,ax=ax, label = 'CF wind', fill = True, color = 'royalblue', alpha = 0.5)
plt.title('Daily CF_wind empirical distribution')
ax.set_xlabel('CF_wind')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(CF_solar_dly.values.flatten(), bins=30, ax=ax, label = 'Solar', fill = True, color = 'gold', alpha = 0.5)
plt.title('Daily (daytime) CF_solar empirical distribution')
plt.show()

# %%
