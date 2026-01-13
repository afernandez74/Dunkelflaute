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
import matplotlib.colors as colors
import geopandas as gpd
import regionmask

#%% Load CF datasets 

CF_wind_path = os.path.expanduser('./../Results/CF_wind/')
CF_solar_path = os.path.expanduser('./../Results/CF_solar/')

CF_wind_file = os.listdir(CF_wind_path)

CF_wind = xr.open_dataarray(CF_wind_path+CF_wind_file[0])

CF_solar_file = os.listdir(CF_solar_path)

CF_solar = xr.open_dataarray(CF_solar_path+CF_solar_file[0])

#%% parameters to plot maps

global_countries_path = os.path.expanduser('./../Data/countries_shp/ne_110m_admin_0_countries.shp')
countries_gdf = gpd.read_file(global_countries_path)
NL_shape = countries_gdf[countries_gdf['ADM0_A3'] == 'NLD']


lat_min = CF_solar['latitude'].min().item()
lat_max = CF_solar['latitude'].max().item()
lon_min = CF_solar['longitude'].min().item()
lon_max = CF_solar['longitude'].max().item()

plot_region = [lon_min-1, lon_max+1, lat_min-1, lat_max+1]


#%% Resample CF from hourly to daily timestep

# daily mean of CF wind
CF_wind_dly = CF_wind.resample(valid_time = "1D").mean()

# daily mean of CF solar daytime hours
CF_solar_dly = CF_solar.resample(valid_time = '1D').mean()

#%% Calculate empirical quantiles for each gridcell

CF_wind_dly_ranks = CF_wind_dly.rank(dim='valid_time',pct=True)
CF_solar_dly_ranks = CF_solar_dly.rank(dim='valid_time',pct=True)

#%% calculate the lower tail dependence coefficient (lambda_L)

def lower_tail_dependence_coefficient(U, V, p, time_dim):
    """
    Calculates the empirical lower tail dependence coefficient (lambda_L) 
    between two empirical quantile DataArrays (U and V) for a given quantile limit p.
    """
    # 1. Count the number of simultaneous low events: I(U <= p AND V <= p)
    # The condition (U <= p) & (V <= p) creates a boolean DataArray (True/False)
    # .sum() treats True as 1 and False as 0, counting the events
    simultaneous_low_count = ((U <= p) & (V <= p)).sum(dim=time_dim)

    # 2. Total number of time steps N
    N = U.valid_time.size

    # The denominator for the simpler (and often used) estimator: N * p
    # N_p_denominator = N * p
    
    # 3. Calculate the empirical lambda_L estimate
    # The empirical P(U <= p, V <= p) is simultaneous_low_count / N
    # The estimate is (1/p) * P(U <= p, V <= p)
    lambda_L_estimate = simultaneous_low_count / (N * p)
    
    # Alternative estimator (more robust for extremely small p):
    # N_events_V = (V <= p).sum(dim=time_dim, dtype=np.int64)
    # lambda_L_estimate_alt = simultaneous_low_count / N_events_V

    return lambda_L_estimate

# Example calculation:
p_limit = 0.05 # The quantile limit, e.g., 10th percentile
lambda_L_da = lower_tail_dependence_coefficient(CF_wind_dly_ranks, CF_solar_dly_ranks, p=p_limit, time_dim='valid_time')

# lambda_L_da will be a DataArray with dimensions (lat, lon), 
# containing the estimated lower tail dependence coefficient for each grid cell.

#%% map distribution of lower tail dependence coefficient

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
lambda_L_da.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Data Value'})

# Add a title
ax.set_title('Empirical lower tail dependence coefficient between CF_Solar and CF_Wind')

# Show the plot
plt.show()
#%% plot uniformly distributed ranks in 2d 
# first histograms: 

# fig, ax = plt.subplots(figsize=(12,8))
# sns.histplot(CF_wind_dly_ranks.values.flatten(), bins=50,ax=ax, label = 'CF wind', fill = True, color = 'royalblue', alpha = 0.5)
# plt.title('Daily CF_wind empirical distribution')
# ax.set_xlabel('CF_wind')

# fig, ax = plt.subplots(figsize=(12,8))
# sns.histplot(CF_solar_dly_ranks.values.flatten(), bins=30, ax=ax, label = 'Solar', fill = True, color = 'gold', alpha = 0.5)
# plt.title('Daily (daytime) CF_solar empirical distribution')

# and a scatter plot: 

plt.scatter(
    CF_wind_dly_ranks.sel(latitude = 52, longitude = 5).values.flatten(), 
    CF_solar_dly_ranks.sel(latitude = 52, longitude = 5).values.flatten(), 
    alpha=0.1, # Use low alpha because you have many points
    s=5,       # Use small size markers
    color='darkblue'
)

# Highlight the lower tail region defined by the limit 'p_limit'
p_limit = 0.1 # Using the example limit from the previous step

# Draw a box representing the lower tail (U <= p and V <= p)
plt.gca().add_patch(plt.Rectangle((0, 0), p_limit, p_limit,
                                  edgecolor='red', facecolor='red', 
                                  linestyle='--', alpha=0.3, 
                                  label=f'Lower Tail (p={p_limit})'))

plt.title('Joint Distribution of Empirical Quantiles')
plt.xlabel('Wind CF Empirical Quantiles')
plt.ylabel('Solar CF Empirical Quantiles')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()



