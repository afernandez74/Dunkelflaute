#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_DF_raw_CF.py
================
Compound dunkelflaute detection based on a combined system capacity factor,
following the approach of the early DF analysis but extended to the full
multi-country domain and with improved event detection logic.

Unlike 04_DF_ID.py (which uses deseasonalised SWDI/SSDI indices), this script
works directly on raw daily CF values.  A single weighted system CF is computed
from wind and solar using a fixed delta parameter, and events are identified
as periods where CF_sys falls below an empirical percentile threshold.

@author: afer
"""
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy.ndimage import label


#%%
# ─── Parameters ───────────────────────────────────────────────────────────────

DELTA_SOLAR    = 0.25
THRESHOLD_PCT  = 10
MIN_DURATION   = 3
MAX_GAP        = 1   
CF_DAILY_DIR   = Path('./../Results/CF_daily')
OUT_DIR = Path("./../Figs/NAC")

#%%
# ─── Load daily capacity factors ──────────────────────────────────────────────

CHUNKS = {'time': -1, 'latitude': 20, 'longitude': 20}

CF_wind_dly      = xr.open_zarr(CF_DAILY_DIR / 'CF_wind_daily.zarr',
                                  consolidated=True, chunks=CHUNKS)['CF_wind_dly']
CF_solar_24h_dly = xr.open_zarr(CF_DAILY_DIR / 'CF_solar_24h_daily.zarr',
                                  consolidated=True, chunks=CHUNKS)['CF_solar_24h_dly']

CF_wind_dly      = CF_wind_dly.compute()
CF_solar_24h_dly = CF_solar_24h_dly.compute()

print(f"CF_wind_dly— shape: {CF_wind_dly.shape}")
print(f"CF_solar_24h_dly — shape: {CF_solar_24h_dly.shape}")
print(f"Time: {str(CF_wind_dly.time.values[0])[:10]} → {str(CF_wind_dly.time.values[-1])[:10]}")


#%%
# ─── Combined system capacity factor ──────────────────────────────────────────

delta_wind = 1.0 - DELTA_SOLAR

CF_sys = delta_wind * CF_wind_dly + DELTA_SOLAR * CF_solar_24h_dly
CF_sys.name = 'CF_sys'
CF_sys.attrs.update({
    'long_name'   : 'System capacity factor',
    'description' : f'δ_wind={delta_wind:.2f} × CF_wind + δ_solar={DELTA_SOLAR:.2f} × CF_solar_24h',
    'units'       : 'fraction',
})

print(f"\nCF_sys — shape: {CF_sys.shape}")
print(f"  mean : {float(CF_sys.mean().values):.3f}")
print(f"  std  : {float(CF_sys.std().values):.3f}")



#%%
# ─── Event detection functions ────────────────────────────────────────────────

def _merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """Merge runs separated by <= max_gap days; discard runs < min_duration."""
    if not np.any(flag):
        return np.zeros_like(flag, dtype=bool)

    T, runs, in_run, start = len(flag), [], False, None
    for t in range(T):
        if flag[t] and not in_run:
            start, in_run = t, True
        elif not flag[t] and in_run:
            runs.append((start, t - 1))
            in_run = False
    if in_run:
        runs.append((start, T - 1))

    if not runs:
        return np.zeros_like(flag, dtype=bool)

    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = np.zeros_like(flag, dtype=bool)
    for s, e in merged:
        if e - s + 1 >= min_duration:
            out[s : e + 1] = True
    return out


def identify_events(cf_sys, threshold, max_gap, min_duration):
    """
    Identify low-CF events using a GLOBAL threshold.
    """
    cf_np = cf_sys.values
    n_lat = cf_sys.sizes['latitude']
    n_lon = cf_sys.sizes['longitude']
    T     = cf_sys.sizes['time']

    out = np.zeros((T, n_lat, n_lon), dtype=bool)

    for i in range(n_lat):
        for j in range(n_lon):
            raw_flag = cf_np[:, i, j] < threshold
            out[:, i, j] = _merge_and_filter_runs_1d(
                raw_flag, max_gap, min_duration
            )

    return xr.DataArray(
        out,
        coords=cf_sys.coords,
        dims=cf_sys.dims,
        attrs={
            'long_name'        : 'Low-CF dunkelflaute event day',
            'threshold_pct'    : THRESHOLD_PCT,
            'threshold_value'  : float(threshold),
            'delta_solar'      : DELTA_SOLAR,
            'max_gap_days'     : max_gap,
            'min_duration_days': min_duration,
        },
    )
def compute_event_stats(event_mask):
    """
    Compute event frequency and return period more robustly.
    """
    n_days   = event_mask.sizes['time']
    n_years  = len(np.unique(event_mask.time.dt.year.values))

    n_lat = event_mask.sizes['latitude']
    n_lon = event_mask.sizes['longitude']

    counts = np.zeros((n_lat, n_lon), dtype=int)

    for i in range(n_lat):
        for j in range(n_lon):
            _, n_ev = label(event_mask.isel(latitude=i, longitude=j).values)
            counts[i, j] = n_ev

    coords = {'latitude': event_mask.latitude, 'longitude': event_mask.longitude}

    # Events per year (primary metric)
    events_per_year = xr.DataArray(
        counts / n_years,
        coords=coords,
        dims=['latitude', 'longitude'],
        attrs={'long_name': 'Events per year'}
    )

    # Return period (derived, smoother)
    return_period = xr.where(events_per_year > 0,
                             1 / events_per_year,
                             np.nan)

    return events_per_year, return_period

#%%
# ─── Threshold and event detection (winter only) ──────────────────────────────

winter_months  = [12, 1, 2]  
CF_sys_winter  = CF_sys.sel(time=CF_sys.time.dt.month.isin(winter_months))

# ─── Climatology (winter only) ────────────────────────────────────────────────

# Mean and std per grid cell (over time)
CF_mean = CF_sys_winter.mean(dim='time')
CF_std  = CF_sys_winter.std(dim='time')
CF_std = CF_std.where(CF_std > 1e-6)

print("Climatology computed:")
print(f"  Mean CF range: {float(CF_mean.min()):.3f} – {float(CF_mean.max()):.3f}")
print(f"  Std  CF range: {float(CF_std.min()):.3f} – {float(CF_std.max()):.3f}")

# ─── Normalize (z-score) ──────────────────────────────────────────────────────

CF_anom = CF_sys_winter - CF_mean
CF_norm = CF_anom / CF_std

CF_norm.name = 'CF_norm'
CF_norm.attrs['description'] = 'Normalized CF anomaly (z-score)'


threshold = np.nanpercentile(CF_norm.values, THRESHOLD_PCT)

print(f"Global normalized threshold ({THRESHOLD_PCT}th pct): {threshold:.2f}")

print("Identifying events ...")
event_mask = identify_events(CF_norm, threshold,
                             max_gap=MAX_GAP,
                             min_duration=MIN_DURATION)
                             

n_event_days = int(event_mask.sum())
print(f"  Total event days (all cells): {n_event_days:,}")
print(f"  Fraction of all cell-days  : {float(event_mask.mean())*100:.2f}%")

yr0     = int(CF_sys.time.dt.year.min().values)
yr1     = int(CF_sys.time.dt.year.max().values)
n_years = yr1 - yr0 + 1

events_per_year, return_period_da = compute_event_stats(event_mask)

events_per_year = events_per_year.rolling(latitude=3, longitude=3, center=True).mean()

print(f"\nSummary ({yr0}–{yr1}, {n_years} years):")
print(f"  Mean events/year (spatial avg) : {float(events_per_year.mean(skipna=True).values):.2f}")
print(f"  Mean return period             : {float(return_period_da.mean(skipna=True).values):.1f} years")
#%%
#%%
# ─── Land mask (cartopy + shapely, no regionmask needed) ──────────────────────

import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.ops import unary_union
import shapely.vectorized

land_shp  = shpreader.natural_earth(resolution='110m', category='physical', name='land')
land_geom = unary_union(list(shpreader.Reader(land_shp).geometries()))

lons, lats = np.meshgrid(CF_sys.longitude.values, CF_sys.latitude.values)
is_land    = shapely.vectorized.contains(land_geom, lons.ravel(), lats.ravel()
                                         ).reshape(lons.shape)

is_land_da = xr.DataArray(is_land,
    coords={'latitude': CF_sys.latitude, 'longitude': CF_sys.longitude},
    dims=['latitude', 'longitude'])

events_per_year_land = events_per_year.where(is_land_da)
return_period_land   = return_period_da.where(is_land_da)

print(f"Land cells: {int(is_land.sum())}")


#%%
#%%
# ─── Map: events per year only ────────────────────────────────────────────────

proj = ccrs.LambertConformal(central_longitude=5, central_latitude=52)

def _base_map(ax):
    ax.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d6e8f5', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.8, linestyle=':', zorder=3)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.4,
                      linestyle='--', linewidth=0.4)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

# ── Bounds ────────────────────────────────────────────────────────────────────
epy_land  = events_per_year_land.values
epy_land  = epy_land[np.isfinite(epy_land) & (epy_land > 0)]

vmin_epy   = np.percentile(epy_land, 2)
vmax_epy   = np.percentile(epy_land, 98)
levels_epy = np.linspace(vmin_epy, vmax_epy, 12)

print(f"Events/year land range : {vmin_epy:.2f} – {vmax_epy:.2f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                       subplot_kw={'projection': proj})

_base_map(ax)

im = events_per_year_land.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='YlOrRd',
    levels=levels_epy,
    vmin=vmin_epy,
    vmax=vmax_epy,
    add_colorbar=False,
    zorder=1,
    extend='both',
)

cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03, extend='both')
cbar.set_label('Mean events per year', fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    f'Dunkelflaute frequency (events/year, land)\n'
    f'{yr0}–{yr1}  |  δ_solar={DELTA_SOLAR}  |  DJF  |  '
    f'{THRESHOLD_PCT}th pct (normalized)  |  min {MIN_DURATION} d',
    fontsize=11, fontweight='bold', pad=8,
)

plt.tight_layout()
fig.savefig(OUT_DIR / 'df_events_per_year_map.svg', bbox_inches='tight')
plt.show()
#%%
# ─── Select grid cell ─────────────────────────────────────────────────────────

lat_sel = 52
lon_sel = 5

CF_wind_pt  = CF_wind_dly.sel(latitude=lat_sel, longitude=lon_sel, method='nearest')
CF_solar_pt = CF_solar_24h_dly.sel(latitude=lat_sel, longitude=lon_sel, method='nearest')

# Restrict to winter
CF_wind_pt  = CF_wind_pt.sel(time=CF_wind_pt.time.dt.month.isin([12,1,2]))
CF_solar_pt = CF_solar_pt.sel(time=CF_solar_pt.time.dt.month.isin([12,1,2]))

# Match event mask
event_pt = event_mask.sel(latitude=lat_sel, longitude=lon_sel, method='nearest')
# %%
# ─── Normalize wind & solar separately ────────────────────────────────────────

wind_anom  = (CF_wind_pt  - CF_wind_pt.mean())  / CF_wind_pt.std()
solar_anom = (CF_solar_pt - CF_solar_pt.mean()) / CF_solar_pt.std()
# ─── Scatter plot ─────────────────────────────────────────────────────────────

plt.figure(figsize=(7, 6))

# All days
plt.scatter(wind_anom, solar_anom,
            s=10, alpha=0.3, label='All days')

# Dunkelflaute days
plt.scatter(wind_anom.where(event_pt),
            solar_anom.where(event_pt),
            s=15, alpha=0.8, label='Dunkelflaute days')

plt.axvline(0)
plt.axhline(0)

plt.xlabel('Wind CF anomaly (σ)')
plt.ylabel('Solar CF anomaly (σ)')
plt.title('Wind vs Solar anomalies (DJF)\nLocation: 52N, 5E')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(OUT_DIR / 'df_scatter_anomalies.svg')
plt.show()
# %%
# ─── Select CF_norm at grid cell ──────────────────────────────────────────────

CF_norm_pt = CF_norm.sel(latitude=lat_sel, longitude=lon_sel, method='nearest')

# ─── Split days into event / non-event index arrays ───────────────────────────

is_event     = event_pt.values.astype(bool)
is_non_event = ~is_event

wind_vals  = wind_anom.values
solar_vals = solar_anom.values
norm_vals  = CF_norm_pt.values   # used to colour event days

# ─── Scatter plot ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))

# Non-dunkelflaute days — neutral gray background points
ax.scatter(
    wind_vals[is_non_event],
    solar_vals[is_non_event],
    s=10, color='#b0b0b0', alpha=0.35, linewidths=0,
    label='Non-dunkelflaute days',
    zorder=1,
)

# Dunkelflaute days — coloured by how low CF_norm is
#   lower (more negative) CF_norm → more intense colour
cmap  = plt.cm.plasma          # reversed: low CF_norm → dark red
vmin  = np.nanpercentile(norm_vals[is_event], 2)
vmax  = np.nanpercentile(norm_vals[is_event], 98)

sc = ax.scatter(
    wind_vals[is_event],
    solar_vals[is_event],
    c=norm_vals[is_event],
    cmap=cmap, vmin=vmin, vmax=vmax,
    s=20, alpha=0.9, linewidths=0.2, edgecolors='k',
    label='Dunkelflaute days',
    zorder=2,
)

cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
cbar.set_label('CF$_{norm}$ (z-score)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Reference lines at zero anomaly
ax.axvline(0, color='k', linewidth=0.7, linestyle='--', alpha=0.5)
ax.axhline(0, color='k', linewidth=0.7, linestyle='--', alpha=0.5)

ax.set_xlabel('Wind CF anomaly (σ)', fontsize=11)
ax.set_ylabel('Solar CF anomaly (σ)', fontsize=11)
ax.set_title(
    f'Wind vs Solar CF anomalies — DJF\n'
    f'Location: {lat_sel}°N, {lon_sel}°E  |  '
    f'{THRESHOLD_PCT}th pct threshold  |  min {MIN_DURATION} d',
    fontsize=11, fontweight='bold',
)
ax.legend(fontsize=9, markerscale=1.5)
ax.grid(color='#e0e0e0', linewidth=0.5)

plt.tight_layout()
fig.savefig(OUT_DIR / 'df_scatter_anomalies.svg', bbox_inches='tight')
plt.show()
# %%
