#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_2018_anomalies.py
======================
Diagnostic plots for 2018 dunkelflaute and agricultural drought signals
over the Netherlands.

Figures
-------
  1. Full year 2018 — CF_comb z-score, t2m z-score, SMDI, SVDI (no shading)
  2. Jan–Jun 2018  — CF_comb & t2m z-scores with cold/low-CF co-occurrence shading
  3. Apr–Sep 2018  — SMDI & SVDI with compound agricultural stress shading

All series smoothed with a centred ROLLING_DAYS rolling mean before plotting.
"""

# %%
# ── Imports ───────────────────────────────────────────────────────────────────
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.rcParams['axes.labelsize']  = 14   # x and y axis labels
mpl.rcParams['xtick.labelsize'] = 14    # x tick marks
mpl.rcParams['ytick.labelsize'] = 14    # y tick marks
mpl.rcParams['axes.titlesize']  = 16   # subplot titles


# %%
# ── Paths ─────────────────────────────────────────────────────────────────────
ERA5_path = Path(os.environ["ERA5_dat"])
DF_path   = Path("./../Results/")
CDHW_path = Path("~/CDHW_ag/Results").expanduser()

cf_wind_path  = DF_path   / "CF_wind"  / "CF_wind_hrly.zarr"
cf_solar_path = DF_path   / "CF_solar" / "CF_solar_hrly.zarr"
era5_path     = ERA5_path / "df_dat"   / "processed" / "df_dat_cleaned.zarr"
smdi_path     = CDHW_path / "SMDI"    / "smdi.zarr"
svdi_path     = CDHW_path / "SVDI"    / "svdi.zarr"

countries_path = Path("../../../CDHW_ag/Data/countries_shp/ne_110m_admin_0_countries.shp")

# ── Parameters ────────────────────────────────────────────────────────────────
YEAR         = 2018
ROLLING_DAYS = 4
W_WIND       = 0.6
W_SOLAR      = 0.4

# ── Plot colours ──────────────────────────────────────────────────────────────
COLOR_CF   = '#0A8A64'   #    — combined CF
COLOR_T2M  = '#C00000'   #     — air temperature
COLOR_SMDI = '#3B5BA5'   #   — soil moisture deficit
COLOR_SVDI = '#C00000'   #   — VPD index

# %%
# ── Load Zarr stores ──────────────────────────────────────────────────────────
print("Opening Zarr stores...")
cf_wind_ds  = xr.open_zarr(cf_wind_path,  consolidated=True, chunks={})
cf_solar_ds = xr.open_zarr(cf_solar_path, consolidated=True, chunks={})
era5_ds     = xr.open_zarr(era5_path,     consolidated=True, chunks={})
smdi_ds     = xr.open_zarr(smdi_path,     consolidated=True, chunks={})
svdi_ds     = xr.open_zarr(svdi_path,     consolidated=True, chunks={})

cf_wind_da  = cf_wind_ds['CF_wind']
cf_solar_da = cf_solar_ds['CF_solar']
t2m_da      = era5_ds['t2m']
smdi_da     = smdi_ds[list(smdi_ds.data_vars)[0]]
svdi_da     = svdi_ds[list(svdi_ds.data_vars)[0]]

countries_gdf = gpd.read_file(countries_path)

# %%
# ── Clip to NL and spatially average ─────────────────────────────────────────
nl_geom = countries_gdf[countries_gdf["SOVEREIGNT"] == "Netherlands"].dissolve()

def nl_mean(da: xr.DataArray) -> xr.DataArray:
    """Clip to NL interior cells (all_touched=False) and return spatial mean."""
    clipped = (
        da
        .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        .rio.write_crs("EPSG:4326")
        .rio.clip(nl_geom.geometry, nl_geom.crs, all_touched=False)
    )
    return clipped.mean(dim=["latitude", "longitude"])

print("Clipping and averaging over the Netherlands...")
cf_wind_nl  = nl_mean(cf_wind_da).compute()
cf_solar_nl = nl_mean(cf_solar_da).compute()
t2m_nl      = nl_mean(t2m_da).compute()
smdi_nl     = nl_mean(smdi_da).compute()
svdi_nl     = nl_mean(svdi_da).compute()

# %%
# ── Combined CF and daily means ───────────────────────────────────────────────
CF_comb_nl    = W_WIND * cf_wind_nl + W_SOLAR * cf_solar_nl
CF_comb_daily = CF_comb_nl.resample(time='1D').mean()
t2m_daily     = t2m_nl.resample(time='1D').mean()
# SMDI and SVDI are already daily

# %%
# DOY climatology (full record) for CF_comb and t2m 
def doy_clim(da: xr.DataArray):
    """DOY mean and std arrays (length 365) from a daily DataArray."""
    grp  = da.assign_coords(doy=da.time.dt.dayofyear).groupby('doy')
    mean = grp.mean().values[:365]
    std  = grp.std().values[:365]
    return mean, std

clim_cf_mean,  clim_cf_std  = doy_clim(CF_comb_daily)
clim_t2m_mean, clim_t2m_std = doy_clim(t2m_daily)

clim_years = (
    int(CF_comb_daily.time.dt.year.min()),
    int(CF_comb_daily.time.dt.year.max()),
)

# %%
# ── Helpers ───────────────────────────────────────────────────────────────────
def sel_months(da: xr.DataArray, year: int, months) -> xr.DataArray:
    mask = (da.time.dt.year == year) & (da.time.dt.month.isin(months))
    return da.sel(time=mask)

def z_score(values: np.ndarray, doy: np.ndarray,
            mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    idx = doy - 1   # DOY is 1-based
    return (values - mean[idx]) / std[idx]

def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

def month_ticks(year: int, months):
    return [pd.Timestamp(year, m, 1) for m in months]

# %%
# ── Compute z-scores for full year 2018 ──────────────────────────────────────
cf_full   = sel_months(CF_comb_daily, YEAR, range(1, 13))
t2m_full  = sel_months(t2m_daily,     YEAR, range(1, 13))
smdi_full = sel_months(smdi_nl,        YEAR, range(1, 13))
svdi_full = sel_months(svdi_nl,        YEAR, range(1, 13))

doy_full  = cf_full.time.dt.dayofyear.values
time_full = cf_full.time.values

z_cf_full  = smooth(z_score(cf_full.values,  doy_full, clim_cf_mean,  clim_cf_std),  ROLLING_DAYS)
z_t2m_full = smooth(z_score(t2m_full.values, doy_full, clim_t2m_mean, clim_t2m_std), ROLLING_DAYS)
smdi_full_s = smooth(smdi_full.values, ROLLING_DAYS)
svdi_full_s = smooth(svdi_full.values, ROLLING_DAYS)

# %%
# ════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full year 2018, all four variables, no shading
# ════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(16, 5))

ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

ax.plot(time_full, z_cf_full,   color=COLOR_CF,   linewidth=1.6,
        label=f'CF_comb z-score')
ax.plot(time_full, z_t2m_full,  color=COLOR_T2M,  linewidth=1.6,
        label=f't2m z-score')
ax.plot(time_full, smdi_full_s, color=COLOR_SMDI, linewidth=1.6,
        label='SMDI')
ax.plot(time_full, svdi_full_s, color=COLOR_SVDI, linewidth=1.6,
        label='SVDI')

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_xlim(time_full[0], time_full[-1])
ax.set_ylim(-4, 4)
ax.set_ylabel('Standardised index / z-score  [ ]')
ax.legend(fontsize=9, loc='lower right', ncol=4)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_title(
    f'{YEAR} — CF_comb & t2m z-scores, SMDI, SVDI — Netherlands spatial mean\n'
    f'CF climatology: {clim_years[0]}–{clim_years[1]}  |  '
    f'{ROLLING_DAYS}-day rolling smooth',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.show()

# %%
# ════════════════════════════════════════════════════════════════════
# FIGURE 2 — Jan–Jun 2018, CF_comb & t2m z-scores, co-occurrence shading
# ════════════════════════════════════════════════════════════════════
cf_H1   = sel_months(CF_comb_daily, YEAR, range(1, 7))
t2m_H1  = sel_months(t2m_daily,     YEAR, range(1, 7))

doy_H1  = cf_H1.time.dt.dayofyear.values
time_H1 = cf_H1.time.values

z_cf_H1  = smooth(z_score(cf_H1.values,  doy_H1, clim_cf_mean,  clim_cf_std),  ROLLING_DAYS)
z_t2m_H1 = smooth(z_score(t2m_H1.values, doy_H1, clim_t2m_mean, clim_t2m_std), ROLLING_DAYS)
both_neg = (z_cf_H1 < 0) & (z_t2m_H1 < 0)

fig2, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(time_H1, -4, 4, where=both_neg,
                color='#FAE08A', alpha=0.8, linewidth=0,
                label='Energy Shortfall')
ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax.plot(time_H1, z_t2m_H1, color=COLOR_T2M, alpha = 0.75, linewidth=3.0, label='Temperature Anom.')
ax.plot(time_H1, z_cf_H1,  color=COLOR_CF,  linewidth=3.0, label='Renewable Generation Anom.')

for tick in month_ticks(YEAR, range(1, 7)):
    ax.axvline(tick, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_xlim(time_H1[0], time_H1[-1])
ax.set_ylim(-4, 4)
ax.set_ylabel('Deviation from mean  [ ]')
ax.legend(fontsize=14, loc='lower right')
ax.grid(True, alpha=0.9, linestyle='--')
# ax.set_title(
#     f'Jan–Jun {YEAR} — Combined CF & t2m standardised anomalies — Netherlands\n'
#     f'CF climatology: {clim_years[0]}–{clim_years[1]}  |  '
#     f'{ROLLING_DAYS}-day rolling smooth',
#     fontsize=12, fontweight='bold'
# )
plt.tight_layout()
plt.show()

# %%
# ════════════════════════════════════════════════════════════════════
# FIGURE 3 — Apr–Sep 2018, SMDI & SVDI, compound stress shading
# Shading: SMDI < 0 (soil moisture deficit) AND SVDI > 0 (VPD excess)
# ════════════════════════════════════════════════════════════════════
smdi_H2 = sel_months(smdi_nl, YEAR, range(4, 10))
svdi_H2 = sel_months(svdi_nl, YEAR, range(4, 10))

time_H2  = smdi_H2.time.values
smdi_H2_s = smooth(smdi_H2.values, 1)
svdi_H2_s = smooth(svdi_H2.values, ROLLING_DAYS)
compound  = (smdi_H2_s < 0) & (svdi_H2_s > 0)

fig3, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(time_H2, -4, 4, where=compound,
                color='#FAE08A', alpha=0.8, linewidth=0,
                label='Crop stressed')
ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax.plot(time_H2, smdi_H2_s, color=COLOR_SMDI, linewidth=3.0, label='Soil Water')
ax.plot(time_H2, svdi_H2_s, color=COLOR_SVDI, linewidth=3.0, label='Heat Stress')

for tick in month_ticks(YEAR, range(4, 10)):
    ax.axvline(tick, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_xlim(time_H2[0], time_H2[-1])
ax.set_ylim(-4, 4)
ax.set_ylabel('Deviation from mean  [ ]')
ax.legend(fontsize=14, loc='lower left')
ax.grid(True, alpha=0.2, linestyle='--')
# ax.set_title(
#     f'Apr–Sep {YEAR} — SMDI & SVDI — Netherlands spatial mean\n'
#     f'{ROLLING_DAYS}-day rolling smooth',
#     fontsize=12, fontweight='bold'
# )
plt.tight_layout()
plt.show()

# %%
