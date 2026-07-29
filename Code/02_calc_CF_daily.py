#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_calc_CF_daily.py
===================
Compute daily capacity factor aggregates from the hourly CF Zarr stores
produced by 01_calc_CF.py.

Three daily aggregates are produced
------------------------------------
1. CF_wind_dly — daily mean wind CF (simple 24-hour mean).

2. CF_solar_24h_dly      — daily mean solar CF averaged over all 24 hours,
                             including nighttime zeros.
                             
3. CF_solar_daytime_dly  — daily mean solar CF averaged only over hours when
                             CF_solar > 0 (panels are actively generating).
                             NOT used as SSDI input; saved for the delta-
                             optimisation analysis in 04_DF_ID.py.

Inputs
------
  ./../Results/CF_wind/CF_wind_hrly.zarr   — from 01_calc_CF.py
  ./../Results/CF_solar/CF_solar_hrly.zarr — from 01_calc_CF.py

Outputs
-------
  ./../Results/CF_daily/CF_wind_daily.zarr          — daily wind CF
  ./../Results/CF_daily/CF_solar_24h_daily.zarr     — daily solar CF (24h)
  ./../Results/CF_daily/CF_solar_daytime_daily.zarr — daily solar CF (daytime)

"""

#%%
# Imports

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask


#%%
# Load hourly CFs from Zarr lazily

CF_wind_zarr  = Path('./../Results/CF_wind/CF_wind_hrly.zarr')
CF_solar_zarr = Path('./../Results/CF_solar/CF_solar_hrly.zarr')

print(f"Loading CF_wind  : {CF_wind_zarr}")
print(f"Loading CF_solar : {CF_solar_zarr}")

# Extract dataarrays preserving native on-disk chunking
CF_wind_hr  = xr.open_zarr(CF_wind_zarr,  consolidated=True, chunks={})['CF_wind']
CF_solar_hr = xr.open_zarr(CF_solar_zarr, consolidated=True, chunks={})['CF_solar']

print(f"\nCF_wind  — shape: {CF_wind_hr.shape}")
print(f"  time: {str(CF_wind_hr.time.values[0])[:10]} → {str(CF_wind_hr.time.values[-1])[:10]}")
print(f"CF_solar — shape: {CF_solar_hr.shape}")
print(f"  time: {str(CF_solar_hr.time.values[0])[:10]} → {str(CF_solar_hr.time.values[-1])[:10]}")


#%%
# Daily aggregation via coarsen

# 1. Daily mean wind CF — 24-hour mean
CF_wind_dly = (
    CF_wind_hr
    .coarsen(time=24, boundary='trim', coord_func='min')
    .mean()
)

# 2. Daily mean solar CF — 24-hour average 
solar = CF_solar_hr

CF_solar_24h_dly = (
    solar
    .coarsen(time=24, boundary='trim', coord_func='min')
    .mean()
)

# 3. Daily mean solar CF — daytime hours only (CF_solar > 0)
solar_masked = solar.where(solar > 0)

CF_solar_daytime_dly = (
    solar_masked
    .coarsen(time=24, boundary='trim', coord_func='min')
    .mean()
)

print(f"\n  CF_wind_dly          — shape: {CF_wind_dly.shape}")
print(f"  CF_solar_24h_dly     — shape: {CF_solar_24h_dly.shape}")
print(f"  CF_solar_daytime_dly — shape: {CF_solar_daytime_dly.shape}")

# Attach metadata
CF_wind_dly = CF_wind_dly.rename('CF_wind_dly')
CF_wind_dly.attrs.update({
    'long_name' : 'Daily mean wind capacity factor — Vestas V90-2.0 MW at 100 m',
    'units'     : 'dimensionless',
    'note'      : '24-hour mean of hourly CF_wind from 01_calc_CF.py',
})

CF_solar_24h_dly = CF_solar_24h_dly.rename('CF_solar_24h_dly')
CF_solar_24h_dly.attrs.update({
    'long_name' : 'Daily mean solar CF — 24-hour average including nighttime zeros',
    'units'     : 'dimensionless',
    'note'      : 'Input to SSDI in 03_calc_SWDI_SSDI.py; '
                  'DOY climatology absorbs day-length structure (Van der Wiel 2019)',
})

CF_solar_daytime_dly = CF_solar_daytime_dly.rename('CF_solar_daytime_dly')
CF_solar_daytime_dly.attrs.update({
    'long_name' : 'Daily mean solar CF — daytime hours only (CF_solar > 0)',
    'units'     : 'dimensionless',
    'note'      : 'Input to delta optimisation in 04_DF_ID.py; '
                  'NaN on days with no generating hours',
})


#%%
# Save daily CFs to Zarr 
# Writing directly to disk triggers Dask computations safely in chunks
# without blowing up system memory.

out_dir = Path('./../Results/CF_daily')
out_dir.mkdir(parents=True, exist_ok=True)

out_paths = {
    'CF_wind_daily'         : (CF_wind_dly,          out_dir / 'CF_wind_daily.zarr'),
    'CF_solar_24h_daily'    : (CF_solar_24h_dly,     out_dir / 'CF_solar_24h_daily.zarr'),
    'CF_solar_daytime_daily': (CF_solar_daytime_dly, out_dir / 'CF_solar_daytime_daily.zarr'),
}

print("\nComputing and streaming results to disk...")
for name, (da, path) in out_paths.items():
    print(f"Saving {name} → {path}")
    da.to_dataset().to_zarr(path, mode='w', consolidated=True)
    print("  Done.")

print("\nAll daily CF Zarr stores saved successfully:")
for name, (_, path) in out_paths.items():
    print(f"  {name}: {path}")


#%%
# Diagnostic: sample grid cell distributions 

lat_idx = CF_wind_dly.sizes['latitude']  // 2
lon_idx = CF_wind_dly.sizes['longitude'] // 2
central_lat = float(CF_wind_dly.latitude.values[lat_idx])
central_lon = float(CF_wind_dly.longitude.values[lon_idx])

print(f"\nSample grid cell: ({central_lat:.2f}°N, {central_lon:.2f}°E)")

wind_cell    = CF_wind_dly.isel(latitude=lat_idx, longitude=lon_idx).dropna('time')
solar24_cell = CF_solar_24h_dly.isel(latitude=lat_idx, longitude=lon_idx).dropna('time')
solardt_cell = CF_solar_daytime_dly.isel(latitude=lat_idx, longitude=lon_idx).dropna('time')

print(f"  CF_wind_dly      mean = {float(wind_cell.mean().values):.4f}")
print(f"  CF_solar_24h     mean = {float(solar24_cell.mean().values):.4f}")
print(f"  CF_solar_daytime mean = {float(solardt_cell.mean().values):.4f}")

fig, axes = plt.subplots(3, 2, figsize=(12, 11))
fig.suptitle(
    f'Daily CF distributions — ({central_lat:.2f}°N, {central_lon:.2f}°E)',
    fontsize=13, fontweight='bold'
)

for row, (da, name, color) in enumerate([
    (wind_cell,    'CF_wind_dly  (24h mean)',               'steelblue'),
    (solar24_cell, 'CF_solar_24h_dly  (incl. night zeros)', 'goldenrod'),
    (solardt_cell, 'CF_solar_daytime_dly  (daylight only)', 'darkorange'),
]):
    vals = da.values

    # Box and whisker
    axes[row, 0].boxplot(
        vals, vert=True, patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.7),
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
    )
    axes[row, 0].set_ylabel('CF [ ]')
    axes[row, 0].set_title(f'Box & Whisker — {name}')

    # Histogram + KDE
    axes[row, 1].hist(vals, bins=50, density=True, alpha=0.6, color=color,
                      edgecolor='black', linewidth=0.3, label='Histogram')
    sns.kdeplot(vals, ax=axes[row, 1], color='black', linewidth=1.5, label='KDE')
    axes[row, 1].set_xlabel('CF [ ]')
    axes[row, 1].set_ylabel('Density')
    axes[row, 1].set_title(f'Distribution — {name}')
    axes[row, 1].legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#%%
# Diagnostic: DOY seasonal cycle at central grid cell 
# Mean-year profiles (DOY mean ± 1 std) for each daily CF aggregate.

def doy_stats(da, max_doy=365):
    """Compute DOY mean and std for a 1-D daily time series (single grid cell)."""
    da_doy = da.assign_coords(doy=da.time.dt.dayofyear).groupby('doy')
    return da_doy.mean().values[:max_doy], da_doy.std().values[:max_doy]


wind_m,  wind_s   = doy_stats(wind_cell)
s24_m,   s24_s    = doy_stats(solar24_cell)
sdt_m,   sdt_s    = doy_stats(solardt_cell)

doy_x = np.arange(1, 366)
month_starts = [pd.Timestamp(year=2001, month=m, day=1).dayofyear for m in range(1, 13)]
month_names  = [calendar.month_abbr[m] for m in range(1, 13)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'Mean-year daily CF seasonal cycle (DOY climatology) — '
    f'({central_lat:.2f}°N, {central_lon:.2f}°E)',
    fontsize=13, fontweight='bold'
)

# Left: wind
axes[0].plot(doy_x, wind_m, color='steelblue', linewidth=1.5, label='DOY mean')
axes[0].fill_between(doy_x, wind_m - wind_s, wind_m + wind_s,
                     color='steelblue', alpha=0.3, label='± 1 std')
axes[0].set_xticks(month_starts)
axes[0].set_xticklabels(month_names, fontsize=9)
axes[0].set_ylabel('CF_wind [ ]')
axes[0].set_title('Wind CF — daily 24h mean')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, linestyle='--')

# Right: solar — 24h vs daytime-only comparison
axes[1].plot(doy_x, s24_m, color='goldenrod', linewidth=1.5,
             label='24h mean (incl. night zeros)')
axes[1].fill_between(doy_x, s24_m - s24_s, s24_m + s24_s,
                     color='goldenrod', alpha=0.25)
axes[1].plot(doy_x, sdt_m, color='darkorange', linewidth=1.5, linestyle='--',
             label='Daytime only (CF > 0 hours)')
axes[1].fill_between(doy_x, sdt_m - sdt_s, sdt_m + sdt_s,
                     color='darkorange', alpha=0.25)
axes[1].set_xticks(month_starts)
axes[1].set_xticklabels(month_names, fontsize=9)
axes[1].set_ylabel('CF_solar [ ]')
axes[1].set_title('Solar CF — 24h mean vs daytime-only mean')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%%
# Diagnostic: spatial maps of time-mean daily CFs 

CF_wind_mean_map  = CF_wind_dly.mean(dim='time')
CF_s24_mean_map   = CF_solar_24h_dly.mean(dim='time')
CF_sdt_mean_map   = CF_solar_daytime_dly.mean(dim='time')

lat_min = float(CF_wind_dly.latitude.min())
lat_max = float(CF_wind_dly.latitude.max())
lon_min = float(CF_wind_dly.longitude.min())
lon_max = float(CF_wind_dly.longitude.max())
plot_region = [lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5]

fig, axes = plt.subplots(
    1, 3, figsize=(18, 7),
    subplot_kw={'projection': ccrs.LambertConformal(central_longitude=5, central_latitude=48)}
)

for ax, data, title, cmap, cbar_lbl in zip(
    axes,
    [CF_wind_mean_map,  CF_s24_mean_map,   CF_sdt_mean_map],
    ['CF_wind_dly (24h mean)',
     'CF_solar_24h_dly (incl. night zeros)',
     'CF_solar_daytime_dly (daylight only)'],
    ['Blues',   'YlOrRd',  'YlOrRd'],
    ['CF_wind [ ]', 'CF_solar_24h [ ]', 'CF_solar_daytime [ ]'],
):
    ax.set_extent(plot_region, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle='--', edgecolor='gray')
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', alpha=0.4)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightblue', alpha=0.4)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
              cbar_kwargs={'label': cbar_lbl, 'shrink': 0.75})
    ax.set_title(title, fontsize=10, fontweight='bold')

plt.suptitle('ERA5 time-mean daily capacity factors — full record',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()