#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:36:52 2025

@author: afer
"""
import os 
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#%%
dat_path = os.path.expanduser('./../Data/ninja_NL/')

pv_dat = pd.read_csv(dat_path+'ninja_PV_NL_national.csv',parse_dates=['time'], skiprows=3)
wind_dat = pd.read_csv(dat_path+'ninja_WIND_NL_total.csv',parse_dates=['time'], skiprows=3)

#%% mask hourly data for 2024
year = 2024
date_mask = pv_dat['time'].dt.year == 2024
pv_dat_24 = pv_dat[date_mask]
wind_dat_24 = wind_dat[date_mask]

#%% plot hourly CFpv for Ninja data

fig, ax = plt.subplots()

# Plot the data
pv_dat_24.plot(x='time', y='NATIONAL', ax=ax, legend=False)

# Set plot title and labels
ax.set_title('Hourly CF solar')
ax.set_xlabel('Date')
ax.set_ylabel('Renewable Ninja CF Value')

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Save the plot to a file
plt.savefig('time_series_plot.png')

# Display the plot
plt.show()
#%% plot Hourly CFwind for Ninja data

fig, ax = plt.subplots()

# Plot the data
wind_dat_24.plot(x='time', y='NATIONAL', ax=ax, legend=False)

# Set plot title and labels
ax.set_title('Hourly CF wind')
ax.set_xlabel('Date')
ax.set_ylabel('Renewable Ninja CF Value')

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Save the plot to a file
plt.savefig('time_series_plot.png')

# Display the plot
plt.show()

#%% scatter plot
plt.figure()

# Plot the 2D histogram
# The 'bins' argument controls the number of bins in both dimensions
plt.scatter(pv_dat_24['NATIONAL'], wind_dat_24['NATIONAL'])

# Add a colorbar to show the density
# cbar = plt.colorbar()
# cbar.set_label('Count')

# Set titles and labels
plt.title('2D Histogram of CFsolar and CFwind')
plt.xlabel('CFsolar')
plt.ylabel('CFwind')

# Adjust layout and save the figure
plt.tight_layout()
# plt.savefig('2d_histogram.png')

# Display the plot
plt.show()

#%% 2d histogram scatter plot

counts, xedges, yedges = np.histogram2d(pv_dat_24['NATIONAL'], wind_dat_24['NATIONAL'], bins=50)

# Find the bin index for each data point
x_indices = np.searchsorted(xedges[:-1], pv_dat_24['NATIONAL']) - 1
y_indices = np.searchsorted(yedges[:-1], wind_dat_24['NATIONAL']) - 1

# Clip indices to prevent out-of-bounds errors for edge cases
x_indices = np.clip(x_indices, 0, counts.shape[0]-1)
y_indices = np.clip(y_indices, 0, counts.shape[1]-1)

# Get the density value for each point from the counts array
density = counts[x_indices, y_indices]

# Create the scatter plot, coloring points by their density
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pv_dat_24['NATIONAL'], wind_dat_24['NATIONAL'], c=density, cmap='viridis', s=10)

# Add a color bar to show the density scale
cbar = plt.colorbar(scatter)
cbar.set_label('Point Density')

# Set plot titles and labels
plt.title('2d Histogram scatter plot of CF values')
plt.xlabel('CF solar')
plt.ylabel('CF wind')

# Adjust layout and save the figure
plt.tight_layout()

# Display the plot
plt.show()

#%% Set time index for pandas resample
#set time index
pv_dat_24.set_index('time', inplace=True)
wind_dat_24.set_index('time', inplace=True)
#%% calculate daily averages
pv_dat_24_daily = pv_dat_24['NATIONAL'].resample('D').mean()
wind_dat_24_daily = wind_dat_24['NATIONAL'].resample('D').mean()

#%%2d histogram of daily data
counts, xedges, yedges = np.histogram2d(pv_dat_24_daily, wind_dat_24_daily, bins=50)

# Find the bin index for each data point
x_indices = np.searchsorted(xedges[:-1], pv_dat_24_daily) - 1
y_indices = np.searchsorted(yedges[:-1], wind_dat_24_daily) - 1

# Clip indices to prevent out-of-bounds errors for edge cases
x_indices = np.clip(x_indices, 0, counts.shape[0]-1)
y_indices = np.clip(y_indices, 0, counts.shape[1]-1)

# Get the density value for each point from the counts array
density = counts[x_indices, y_indices]

# Create the scatter plot, coloring points by their density
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pv_dat_24_daily, wind_dat_24_daily, c=density, cmap='viridis', s=10)

# Add a color bar to show the density scale
cbar = plt.colorbar(scatter)
cbar.set_label('Point Density')

# Set plot titles and labels
plt.title('2d Histogram scatter plot of Daily CF values')
plt.xlabel('CF solar')
plt.ylabel('CF wind')

# Adjust layout and save the figure
plt.tight_layout()

# Display the plot
plt.show()

#%% kernel density plots for both variables

fig, ax = plt.subplots()
sns.histplot(pv_dat_24['NATIONAL'], bins=30,ax=ax, label = 'Solar', fill = True, color = 'red', alpha = 0.5)
sns.histplot(wind_dat_24['NATIONAL'],bins=30, ax=ax, label = 'Wind', fill = True, color = 'blue', alpha = 0.5)
plt.title('hourly data')


fig, ax = plt.subplots()
sns.histplot(pv_dat_24_daily, bins=30,ax=ax, label = 'Solar', fill = True, color = 'red', alpha = 0.5)
sns.histplot(wind_dat_24_daily,bins=30,ax=ax, label = 'Wind', fill = True, color = 'blue', alpha = 0.5)
plt.title('daily averages')
