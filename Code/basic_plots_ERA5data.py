#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:49:16 2025

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

#%% load 2 datasets: wind and temp files AND solar rad files

dat_path = os.path.expanduser('./../Data/ERA5/')

files = os.listdir(dat_path)

name_file = 'ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8'

file_path = dat_path + name_file

ds = xr.open_mfdataset(file_path)

u100 = ds.u100
v100 = ds.v100

ssrd = ds.ssrd / 3600

t2m = ds.t2m -273.15
#%% maximum coordinates in dataset

lat_min = ds['latitude'].min().item()
lat_max = ds['latitude'].max().item()
lon_min = ds['longitude'].min().item()
lon_max = ds['longitude'].max().item()

plot_region = [lon_min-1, lon_max+1, lat_min-1, lat_max+1]

#%% calculations 

# net windspeed
ws100 = np.sqrt(u100**2 + v100**2)

# windspeed over whole area
ws100_ts = ws100.mean(dim = ['latitude','longitude'])

# monthly mean of windspeed
ws_100_monthly = ws100_ts.resample(valid_time = '1M').mean()

# surface temperature over area
t2m_ts = t2m.mean(dim=['latitude','longitude'])

# monthly surface air temperature
t2m_monthly = t2m_ts.resample(valid_time='1ME').mean()

# solar irradiance over area
ssrd_ts = ssrd.mean(dim = ['latitude','longitude'])

# monthly mean solar irradiance 
ssrd_monthly = ssrd_ts.resample(valid_time = '1ME').mean()

#%% plot monthly means of data

# plot wind speed 
fig, ax = plt.subplots(figsize=(12,6))
ws_100_monthly.plot(ax=ax,linewidth=0.9, linestyle = '-', label = '100m wind')
ax.set_title('ERA5 monthly net 100m windspeed')
ax.set_ylabel('Wind Speed [m/s]')

#  plot temp
fix, ax = plt.subplots(figsize = (12,6))
t2m_monthly.plot(ax=ax,linewidth=0.9,color='darkred',label = '2m air temperature')
ax.set_title('ERA 5 monthly surface air temperature')
ax.set_ylabel('2m Temperature [deg C]')

# plot ssrd
fix, ax = plt.subplots(figsize = (12,6))
ssrd_monthly.plot(ax=ax,linewidth=1.0,color='goldenrod',label = 'surface solar radiation')
ax.set_title('ERA 5 monthly surface solar radiation over NL')
ax.set_ylabel('radiation [W/m2]')


#%% map average wind power over time period

# --- Figure and projection ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(
    1, 1, 1,
    projection=ccrs.LambertConformal(central_longitude=5, central_latitude=40)
)

# --- Map extent ---
ax.set_extent(plot_region, crs=ccrs.PlateCarree())

# --- Map features ---
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=3)
ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='dimgray', zorder=3)

# --- Gridlines ---
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.4, linestyle='--', linewidth=0.5)
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}

# --- Plot data ---
ws100.mean(dim='valid_time').plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='RdPu',
    cbar_kwargs={'label': 'Wind Speed m/s ', 'shrink': 0.8, 'pad': 0.05}
)

# --- Title and fonts ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

ax.set_title(
    'Average Hourly Wind Speed @100 m 1980-2024',
    fontsize=16, fontweight='bold', pad=15
)

# --- Country labels ---
ax.text(5.3, 52.2, 'Netherlands', fontsize=14, fontweight='bold',
        color='dimgray', ha='center', transform=ccrs.PlateCarree())
ax.text(4.5, 50.8, 'Belgium', fontsize=13, fontweight='bold',
        color='dimgray', ha='center', transform=ccrs.PlateCarree())
ax.text(7.5, 51.5, 'Germany', fontsize=13, fontweight='bold',
        color='dimgray', ha='left', transform=ccrs.PlateCarree())

# --- Layout and show ---
plt.tight_layout()

path_save = os.path.expanduser('./../Figures/ERA5/')
name = 'u100_map'
save_toggle = input("Save {name} ? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    plt.savefig(path_save+name+'.png', dpi=300)
    plt.savefig(path_save+name+'.svg')

plt.show()

#%% map average ssrd over time period
# --- Figure and projection ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(
    1, 1, 1,
    projection=ccrs.LambertConformal(central_longitude=5, central_latitude=40)
)

# --- Map extent ---
ax.set_extent(plot_region, crs=ccrs.PlateCarree())

# --- Map features ---
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=3)
ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='dimgray', zorder=3)

# --- Gridlines ---
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.4, linestyle='--', linewidth=0.5)
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}

# Plot your xarray data on the map
ssrd.mean(dim='valid_time').plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cbar_kwargs={'label': 'Solar Rad J/m2'}, cmap='YlOrBr_r')

# --- Title and fonts ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

ax.set_title(
    'Average Hourly Surface SW Solar Rad 1980-2024',
    fontsize=16, fontweight='bold', pad=15
)

# --- Country labels ---
ax.text(5.3, 52.2, 'Netherlands', fontsize=14, fontweight='bold',
        color='dimgray', ha='center', transform=ccrs.PlateCarree())
ax.text(4.5, 50.8, 'Belgium', fontsize=13, fontweight='bold',
        color='dimgray', ha='center', transform=ccrs.PlateCarree())
ax.text(7.5, 51.5, 'Germany', fontsize=13, fontweight='bold',
        color='dimgray', ha='left', transform=ccrs.PlateCarree())

# --- Layout and show ---
plt.tight_layout()

path_save = os.path.expanduser('./../Figures/ERA5/')
name = 'ssrd_map'
save_toggle = input("Save {name} ? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    plt.savefig(path_save+name+'.png', dpi=300)
    plt.savefig(path_save+name+'.svg')

plt.show()



#%% dunkelflaute? 

#standardize data
ws_10_stdrd = (ws100 - ws100.mean(dim=['latitude','longitude','valid_time']))/(ws100.std(dim=['latitude','longitude','valid_time']))
ssrd_stdrd = (ssrd - ssrd.mean(dim=['latitude','longitude','valid_time']))/(ssrd.std(dim=['latitude','longitude','valid_time']))

ws_point = ws_10_stdrd.sel(latitude = 52, longitude = 5, method = 'nearest')
ssrd_point = ssrd_stdrd.sel(latitude = 52, longitude = 5, method = 'nearest')

counts, xedges, yedges = np.histogram2d(ws_point.values.flatten(),
                                        ssrd_point.values.flatten(),
                                        bins=50,
                                        density=True)
H = counts.T

plt.figure(figsize=(10, 8))

# Use pcolormesh to plot the 50x50 grid of counts directly
# This is orders of magnitude faster than a scatter plot of millions of points.
mesh = plt.pcolormesh(
    xedges,
    yedges,
    H,
    cmap='viridis',
    norm=colors.LogNorm(vmin=H.min()+1e-10, vmax=H.max()) # Set vmin above zero
)

# Add a color bar
cbar = plt.colorbar(mesh)
cbar.set_label('Probability Density')

# Set plot titles and labels
plt.title('2D Histogram (Density Plot) of Standardized Wind vs. Solar at 52N, 5E')
plt.xlabel('Standardized Wind Speed')
plt.ylabel('Standardized Solar Rad')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
