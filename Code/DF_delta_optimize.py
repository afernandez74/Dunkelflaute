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

#%% Histogram of daily CF values 

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(CF_wind_dly.values.flatten(), bins=50,ax=ax, label = 'CF wind', fill = True, color = 'royalblue', alpha = 0.5)
plt.title('Daily CF_wind empirical distribution')
ax.set_xlabel('CF_wind')

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(CF_solar_dly.values.flatten(), bins=30, ax=ax, label = 'Solar', fill = True, color = 'gold', alpha = 0.5)
plt.title('Daily (daytime) CF_solar empirical distribution')

#%% 2d density plot for a single spatial point 

CF_wind_dly_pnt = CF_wind_dly.sel(latitude = 52, longitude = 5, method = 'nearest')
CF_solar_dly_pnt = CF_solar_dly.sel(latitude = 52, longitude = 5, method = 'nearest')

# Calculate the 2D density array
counts, xedges, yedges = np.histogram2d(
    CF_wind_dly_pnt.values.flatten(),    
    CF_solar_dly_pnt.values.flatten(),  
    bins=50,                      
    density=True                  # Normalize the counts to show probability density
)
# Transpose the array (np.histogram2d returns X, Y; plotting expects Y, X)
H = counts.T 


plt.figure(figsize=(10, 8))

# Use pcolormesh to plot the 50x50 density grid
# Using LogNorm for the color scale helps visualize variations when peaks are high
mesh = plt.pcolormesh(
    xedges,
    yedges,
    H,
    cmap='viridis')

# Add a color bar
cbar = plt.colorbar(mesh)
cbar.set_label('Count')

# Set plot titles and labels
plt.title('2D Density Plot CF_wind vs. CF_solar')
plt.xlabel('CF_Wind')
plt.ylabel('CF_solar')

# Display the plot
plt.tight_layout()
plt.show()

#%% define function to decluster DF events and count them in each gridcell of the CF dataarray

def decluster_events(CF_sys, threshold, duration, min_gap):
    """
    Decluster low-capacity-factor events.

    Parameters
    ----------
    CF_sys : xr.DataArray
        System capacity factor with dims (time, lat, lon).
    threshold : float
        Threshold for defining a dunkelflaute event 
    min_duration : int
        Minimum number of consecutive days to count as an event 
    min_gap : int
        Minimum number of days above threshold required between independent events.

    Returns
    -------
    xr.Dataset
        With variables:
        - event_count: number of events per gridcell
        - mean_duration: mean duration (days) of events
        - mean_min_cf: mean of minimum CF across events
    """

    def _decluster_1d(cf_ts):
        # boolean mask of DF event
        mask = cf_ts <= threshold
        if not mask.any():
            return np.array([0, np.nan, np.nan])

        # Identify start/end of clusters
        diff = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # return events as pair of (start, end), if duration is sufficient
        events = []
        for s, e in zip(starts, ends):
            dur = e - s
            if dur >= duration:
                events.append([s, e])
        
        # no events, return NaN
        if len(events) == 0:
            return np.array([0, np.nan, np.nan])

        # Merge events closer than min_gap
        merged = []
        cur_start, cur_end = events[0]
        for s, e in events[1:]:
            if s - cur_end <= min_gap:  # too close, merge
                cur_end = e
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # Collect stats
        durations = []
        min_cfs = []
        for s, e in merged:
            cf_event = cf_ts[s:e]
            durations.append(len(cf_event))
            min_cfs.append(cf_event.min())

        return np.array([len(merged), np.mean(durations), np.mean(min_cfs)])

    # Apply function across grid
    results = xr.apply_ufunc(
        _decluster_1d,
        CF,
        input_core_dims=[["valid_time"]],
        output_core_dims=[["metric"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={"metric": 3},
    )

    results = results.assign_coords(metric=["event_count", "mean_duration", "mean_min_cf"])
    return results.to_dataset(dim="metric")

#%% Optimize value of delta_solar

#define search space for delta_solar
d_solar_range = np.arange(0,1,0.05)

# Store results
opt_event_count = None
opt_d_solar = None

# Parameters for declustering
threshold = 0.05
duration = 3
min_gap = 2

# parameters for declustering

X = 0.05
duration = 1
min_gap = 2

for i, d_solar in enumerate (d_solar_range):
    d_wind = 1-d_solar
    CF = d_wind * CF_wind_dly + d_solar * CF_solar_dly
    
    DF = decluster_events(
        CF,
        threshold = X, 
        duration = duration,
        min_gap = min_gap
        )
    if i == 0:
        opt_event_count = DF.event_count
        opt_d_solar = xr.full_like(DF.event_count, d_solar)
    else: 
        mask = DF.event_count < opt_event_count
        opt_event_count = xr.where(mask, DF.event_count, opt_event_count)
        opt_d_solar = xr.where(mask, d_solar, opt_d_solar)
        
# put results in dataset
DF_opt = xr.Dataset({
    "opt_event_count": opt_event_count,
    "opt_d_solar": opt_d_solar
    })

#%% plot map of optimized d_solar

mask = regionmask.Regions([NL_shape.geometry.values[0]]).mask(CF.longitude, CF.latitude)
DF_opt_nl = DF_opt.where(mask == 0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
    central_longitude=5, central_latitude=40
))
ax.set_extent(plot_region, crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

DF_opt_nl.opt_d_solar.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='viridis',
    cbar_kwargs={'label': 'Optimal Solar Share δ_solar'}
)
ax.set_title('Optimal δ_solar (min DF event count)')
plt.show()
