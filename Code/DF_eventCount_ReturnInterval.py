#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:04:03 2025

@author: afer
"""

import os 
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
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

#%% Joint CF analysis
# proportion of installed capacity from van der Wiel et al., 2019
delta_solar = 0.25
delta_wind = 1 - delta_solar

# total capacity factor
CF = delta_wind * CF_wind_dly + delta_solar * CF_solar_dly
#%% histogram of total CF
sns.set_theme(style="whitegrid")  # base style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.linewidth': 1.2
})

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

sns.histplot(
    CF.values.flatten(),
    bins=50,
    ax=ax,
    label='CF (wind + solar)',
    color='steelblue',
    fill=True,
    alpha=0.6,
    kde=True,
    line_kws={'lw': 2, 'color': 'black'}  # KDE line styling
)

# --- Labels and title ---
ax.set_title('Empirical Distribution of Daily Wind + Solar Capacity Factors in NL gridcells (from ERA5)', 
             fontweight='bold', 
             pad=15)
ax.set_xlabel('Combined Capacity Factor (CF)', labelpad=10)
ax.set_ylabel('Frequency', labelpad=10)

# --- Aesthetics ---
ax.legend(frameon=False, fontsize=13)
sns.despine(ax=ax)  # remove top/right spines
ax.grid(True, linestyle='--', alpha=0.3)

# --- Optional: save high-quality figure ---
plt.tight_layout()

# --- save figure

path_save = os.path.expanduser('./../Figures/ERA5/')
name = 'CF_wind_solar_histogram'
save_toggle = input("Save {name} ? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    plt.savefig(path_save+name+'.png', dpi=300)
    plt.savefig(path_save+name+'.svg')

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

#%%Run the decluster function on CF dataset
# parameters for empirical analysis 
X = 0.05
duration = 3
min_gap = 1

DF_events = decluster_events(
    CF,
    threshold=X,
    duration=duration,
    min_gap=min_gap
)

#%% Map geographic distribution of captured events

mask = regionmask.Regions([NL_shape.geometry.values[0]]).mask(CF.longitude, CF.latitude)

# mask out everything outside NL
DF_events_nl = DF_events.where(mask == 0)   # 0 means "inside the first polygon"

n_years = (CF.valid_time[-1].dt.year.values-CF.valid_time[0].dt.year.values)+1

# compute return interval: years / count
RI = n_years / DF_events_nl.event_count

# add it back into your dataset
DF_events_nl = DF_events_nl.assign(return_interval=RI)


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
DF_events_nl.return_interval.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'years'})

# Add a title
ax.set_title('DF NL  Return Interval')

# Show the plot
plt.show()


#%% Nicer maps? 

# --- Style and font settings ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,
})

# --- Figure setup ---
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=5, central_latitude=40))
ax.set_extent(plot_region, crs=ccrs.PlateCarree())

# --- Base map features ---
ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=0)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='none', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='gray', zorder=2)

# --- Gridlines ---
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# --- Color map customization ---
cmap = plt.cm.viridis  # or "magma_r" / "plasma_r" for higher contrast
# norm = matplotlib.colors.LogNorm(vmin=DF_events_nl.return_interval.min(), vmax=DF_events_nl.return_interval.max())

# --- Plot the data ---
im = DF_events_nl.return_interval.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    # norm=norm,
    cbar_kwargs={
        'label': 'Return interval (years)',
        'shrink': 0.8,
        'pad': 0.03,
        'aspect': 25,
    },
)


# --- Country labels ---
ax.text(5.3, 52.2, 'Netherlands', fontsize=13, fontweight='bold', color='dimgray',
        ha='center', transform=ccrs.PlateCarree())
ax.text(4.5, 50.8, 'Belgium', fontsize=12, fontweight='bold', color='dimgray',
        ha='center', transform=ccrs.PlateCarree())
ax.text(7.5, 51.5, 'Germany', fontsize=12, fontweight='bold', color='dimgray',
        ha='left', transform=ccrs.PlateCarree())

# --- Title ---
ax.set_title(
    'Empirical Return Interval of Renewable Energy Shortfall Days',
    fontsize=16,
    fontweight='bold',
    pad=15,
)

# --- Final touches ---
plt.tight_layout()
path_save = os.path.expanduser('./../Figures/ERA5/')
name = 'CF_wind_solar_empirical_RI'
save_toggle = input("Save {name} ? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    plt.savefig(path_save+name+'.png', dpi=300)
    plt.savefig(path_save+name+'.svg')

plt.show()


#%% statistics of DF events

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(DF_events_nl.event_count.values.flatten(), 
             bins=25,ax=ax, label = 'number of DF events', 
             fill = True, color = 'slategray', alpha = 0.5)
plt.title('Distribution of DF event counts')
ax.set_xlabel('Count of events per gridcell')
#%% Map geographic distribution of empirical return intervals

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
DF_events_nl.mean_duration.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Data Value'})

# Add a title
ax.set_title('DF NL event mean duration 1980-2024')

# Show the plot
plt.show()

