#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_calc_SWDI_SSDI.py
====================
Compute the Standardized Wind Deficit Index (SWDI) and Standardized Solar
Deficit Index (SSDI) from the previously computed hourly capacity factor files.

Scientific background
---------------------
Both indices follow the deseasonalised z-score approach of Van der Wiel et al.
(2019, Environ. Res. Lett.):

    index_d = ( CF_d  −  µ(DOY_d) ) / σ(DOY_d)

where µ(DOY) and σ(DOY) are the climatological mean and standard deviation of
the capacity factor across the full record for each calendar day (day-of-year).
Negative values indicate a generation deficit; positive values indicate
above-normal generation.  This is analogous to the SVDI used in the ag
subproject (Gamelin et al. 2022/2025), just applied to wind and solar CFs
rather than vapour-pressure deficit.

Daily solar CF aggregation
--------------------------
Two daily solar CF aggregates are computed and saved:
  - CF_solar_24h : mean over all 24 hours (including nighttime zeros).
    Used as input to SSDI.  The seasonal day-length pattern is encoded in the
    DOY climatology, so the standardisation correctly expresses "how anomalously
    low is today's solar generation relative to what this calendar day
    typically looks like", including the structural zero-padding at night
    (Van der Wiel et al. 2019).
  - CF_solar_daytime : mean restricted to hours where CF_solar > 0 (i.e.,
    hours when panels are actually generating).  NOT used for SSDI; saved for
    the delta-optimisation analysis in 04_DF_ID.py, where the daytime-only mean
    allows a fair wind/solar comparison at hours when both resources can
    simultaneously contribute.

Winter SSDI NaNs
----------------
In winter, solar CF is near zero on virtually every day, making the DOY
standard deviation σ → 0.  A guard (σ < 1e-6 → NaN) prevents spurious
division-by-zero amplification.  A small fraction of NaN SSDI days in winter
is expected and physically meaningful: compound dunkelflaute events in winter
are primarily identified through the wind deficit component (SWDI).

Inputs
------
  ./../Results/CF_wind/   — hourly CF_wind  (DataArray, dim: valid_time or time)
  ./../Results/CF_solar/  — hourly CF_solar (DataArray, dim: valid_time or time)

Outputs
-------
  ./../Results/SWDI/swdi.zarr               — SWDI (daily, same grid as CFs)
  ./../Results/SSDI/ssdi.zarr               — SSDI (daily, same grid as CFs)
  ./../Results/CF_solar_daytime/            — CF_solar_daytime daily Zarr

@author: afer
"""

#%%
# Imports

import os
import calendar
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# Load daily CF datasets 
#
# from 02_calc_CF_daily.py 

CF_daily_dir = Path('./../Results/CF_daily')

# Keep time contiguous; chunk only spatially
CHUNKS = {'time': -1, 'latitude': 20, 'longitude': 20}

CF_wind_dly          = xr.open_zarr(CF_daily_dir / 'CF_wind_daily.zarr',
                                     consolidated=True, chunks=CHUNKS)['CF_wind_dly']
CF_solar_24h_dly     = xr.open_zarr(CF_daily_dir / 'CF_solar_24h_daily.zarr',
                                     consolidated=True, chunks=CHUNKS)['CF_solar_24h_dly']
CF_solar_daytime_dly = xr.open_zarr(CF_daily_dir / 'CF_solar_daytime_daily.zarr',
                                     consolidated=True, chunks=CHUNKS)['CF_solar_daytime_dly']

print(f"Loaded daily CFs from {CF_daily_dir}")
print(f"  CF_wind_dly          — shape: {CF_wind_dly.shape}")
print(f"  CF_solar_24h_dly     — shape: {CF_solar_24h_dly.shape}")
print(f"  CF_solar_daytime_dly — shape: {CF_solar_daytime_dly.shape}")
print(f"  time: {str(CF_wind_dly.time.values[0])[:10]} → {str(CF_wind_dly.time.values[-1])[:10]}")

#%%
# Daily distributions central cell

# Central NL grid cell used for all single-point diagnostics throughout the script
# lat_idx = CF_wind_dly.latitude.size // 2
# lon_idx = CF_wind_dly.longitude.size // 2
# central_lat = float(CF_wind_dly.latitude.values[lat_idx])
# central_lon = float(CF_wind_dly.longitude.values[lon_idx])
central_lat = 52
central_lon = 5
print(f"\nDiagnostic grid cell: ({central_lat:.2f}°N, {central_lon:.2f}°E)")

wind_cell  = CF_wind_dly.sel(latitude=central_lat, longitude=central_lon)
solar_cell = CF_solar_24h_dly.sel(latitude=central_lat, longitude=central_lon)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Daily CF distributions — ({central_lat:.2f}°N, {central_lon:.2f}°E)', fontsize=13)

axs[0].hist(wind_cell.values, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='k')
sns.kdeplot(wind_cell.values, ax=axs[0], color='navy')
axs[0].set_xlabel('CF_wind (daily 24h mean)')
axs[0].set_ylabel('Density')
axs[0].set_title('Wind capacity factor')
axs[0].text(0.97, 0.97, f'mean = {float(wind_cell.mean().values):.3f}',
            transform=axs[0].transAxes, ha='right', va='top', fontsize=10)

solar_vals = solar_cell.dropna(dim='time').values
axs[1].hist(solar_vals, bins=40, density=True, alpha=0.6, color='gold', edgecolor='k')
sns.kdeplot(solar_vals, ax=axs[1], color='darkorange')
axs[1].set_xlabel('CF_solar (daily 24h mean)')
axs[1].set_ylabel('Density')
axs[1].set_title('Solar capacity factor')
axs[1].text(0.97, 0.97, f'mean = {float(solar_cell.mean(skipna=True).values):.3f}',
            transform=axs[1].transAxes, ha='right', va='top', fontsize=10)

plt.tight_layout()
plt.show()


#%%
# DOY climatology functions 

def calculate_doy_climatology(cf_daily, baseline_start=None, baseline_end=None):
    """
    Compute day-of-year (DOY) climatological mean and standard deviation of CF.

    For each DOY (1–365/366) the mean µ(DOY) and standard deviation σ(DOY) are
    computed over all years in the baseline period and at every grid cell
    independently.  The results are indexed by integer DOY and are aligned back
    onto a full time series using .sel(doy=da.time.dt.dayofyear).
    
    Parameters
    ----------
    cf_daily       : xr.DataArray   dims: (time, latitude, longitude)
    baseline_start : str or None    first year of baseline, e.g. '1980'
    baseline_end   : str or None    last  year of baseline, e.g. '2023'

    Returns
    -------
    cf_mean_doy : xr.DataArray   dims: (doy, latitude, longitude)
    cf_std_doy  : xr.DataArray   dims: (doy, latitude, longitude)
    """
    years = pd.DatetimeIndex(cf_daily['time'].values).year
    baseline_start = baseline_start or str(years.min())
    baseline_end   = baseline_end   or str(years.max())

    cf_baseline = (
        cf_daily
        .sel(time=slice(baseline_start, baseline_end))
        .chunk({'time': -1, 'latitude': -1, 'longitude': -1})
        .assign_coords(doy=lambda x: x.time.dt.dayofyear)
    )

    cf_mean_doy = cf_baseline.groupby('doy').mean('time').compute()
    cf_std_doy  = cf_baseline.groupby('doy').std('time').compute()

    # Guard: near-zero σ causes division-by-zero 
    # Setting σ → NaN propagates NaN to the index 
    cf_std_doy = xr.where(cf_std_doy > 1e-6, cf_std_doy, np.nan)

    return cf_mean_doy, cf_std_doy


def calculate_standardized_index(cf_daily, cf_mean_doy, cf_std_doy):
    """
    Compute the deseasonalised z-score standardised index.

        index_d = ( CF_d  −  µ(DOY_d) ) / σ(DOY_d)

    Negative values = below seasonal normal (generation deficit).
    Positive values = above seasonal normal (generation surplus).

    The .sel(doy=cf_daily.doy) call maps each time step in cf_daily to its
    corresponding DOY climatological statistic, so the result has the same
    (time, latitude, longitude) shape as cf_daily.

    Parameters
    ----------
    cf_daily     : xr.DataArray   dims: (time, latitude, longitude)
    cf_mean_doy  : xr.DataArray   dims: (doy, latitude, longitude)
    cf_std_doy   : xr.DataArray   dims: (doy, latitude, longitude)

    Returns
    -------
    index : xr.DataArray   dims: (time, latitude, longitude)
    """
    # Attach DOY as a non-dimension coordinate so we can use it for selection
    cf_daily = cf_daily.assign_coords(doy=cf_daily.time.dt.dayofyear)

    # Map each time step to its DOY climatological statistics.
    mean_aligned = cf_mean_doy.sel(doy=cf_daily.doy)
    std_aligned  = cf_std_doy.sel(doy=cf_daily.doy)

    index = (cf_daily - mean_aligned) / std_aligned
    return index


#%%
# Compute SWDI 

# Standardized Wind Deficit Index:
#   SWDI < 0    : wind generation below seasonal normal
#   SWDI < −1.0 : moderate deficit  (~16th percentile) 
#   SWDI < −1.5 : severe deficit    (~7th  percentile) 

print("Computing SWDI DOY climatology ...")
wind_mean_doy, wind_std_doy = calculate_doy_climatology(CF_wind_dly)

print("Computing SWDI ...")
SWDI = calculate_standardized_index(CF_wind_dly, wind_mean_doy, wind_std_doy).compute()
SWDI.name = 'SWDI'
SWDI.attrs.update({
    'long_name'       : 'Standardized Wind Deficit Index',
    'units'           : 'dimensionless (z-score)',
    'description'     : 'Deseasonalised z-score of daily mean wind CF at 100 m. ',
    'source_variable' : 'CF_wind — daily mean of hourly wind CF (Vestas V90-2MW power curve)',
    'baseline_period' : f"{int(CF_wind_dly.time.dt.year.min().values)}"
                        f"–{int(CF_wind_dly.time.dt.year.max().values)}",
})

print(f"SWDI computed — shape: {SWDI.shape}")
print(f"  range : {float(SWDI.min(skipna=True).values):.2f} "
      f"to {float(SWDI.max(skipna=True).values):.2f}")
print(f"  mean  : {float(SWDI.mean(skipna=True).values):.3f}  (should be ≈ 0)")
print(f"  std   : {float(SWDI.std(skipna=True).values):.3f}   (should be ≈ 1)")


#%%
# SWDI diagnostics 
swdi_cell = SWDI.sel(latitude=central_lat, longitude=central_lon).dropna(dim='time')

# Box & Whisker + empirical distribution
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(f'SWDI — central grid cell ({central_lat:.2f}°N, {central_lon:.2f}°E)',
             fontsize=13)

axs[0].boxplot(swdi_cell.values, vert=True, patch_artist=True,
               boxprops={'facecolor': 'steelblue', 'alpha': 0.6}, showmeans=True)
axs[0].axhline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='−1 (moderate)')
axs[0].axhline(-1.5, color='red',    linestyle='--', linewidth=1.5, label='−1.5 (severe)')
axs[0].set_ylabel('SWDI')
axs[0].set_title('Box & Whisker')
axs[0].legend(fontsize=9)

axs[1].hist(swdi_cell.values, bins=40, density=True,
            alpha=0.6, color='steelblue', edgecolor='k')
sns.kdeplot(swdi_cell.values, ax=axs[1], color='navy', label='KDE')
axs[1].axvline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='−1 (moderate)')
axs[1].axvline(-1.5, color='red',    linestyle='--', linewidth=1.5, label='−1.5 (severe)')
axs[1].set_xlabel('SWDI')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical distribution')
axs[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


#%%
# SWDI full time series for the central NL grid cell during 2016-2017 episode
# Shading around known major European renewable energy drought episodes
# validates that SWDI captures the events described in the literature.

# Reference dunkelflaute / wind drought periods from Li et al. (2021)
# and Mockert et al. (2023): December 2016 and January 2017 are among the most
# cited severe events in the North Sea region.
known_events = {
    'Dec 2016': ('2016-12-01', '2016-12-31'),
    'Jan 2017': ('2017-01-01', '2017-01-31'),
    # 'Dec 2021': ('2021-12-01', '2021-12-31'),
}

fig, ax = plt.subplots(figsize=(16, 5))

swdi_cell.sel(time=slice("2016","2018")).plot(ax=ax, color='steelblue', linewidth=0.8, alpha=0.5, label='SWDI (daily)')
swdi_cell.sel(time=slice("2016","2018")).rolling(time=14, center=True).mean().plot(
    ax=ax, color='navy', linewidth=2, label='14-day rolling mean')

ax.axhline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='−1 (moderate deficit)')
ax.axhline(-1.5, color='red',    linestyle='--', linewidth=1.5, label='−1.5 (severe deficit)')
ax.axhline( 0,   color='black',  linewidth=0.5,  alpha=0.5)

for label_txt, (t0, t1) in known_events.items():
    ax.axvspan(pd.to_datetime(t0), pd.to_datetime(t1),
               alpha=0.2, color='tomato',
               label='Known wind drought' if label_txt == 'Dec 2016' else '_nolegend_')
    ax.text(pd.to_datetime(t0), ax.get_ylim()[0] + 0.2, label_txt,
            fontsize=8, color='firebrick', rotation=90, va='bottom')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('SWDI', fontsize=12)
ax.set_title(f'SWDI time series — ({central_lat:.2f}°N, {central_lon:.2f}°E)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# Compute SSDI 
#
# Standardized Solar Deficit Index:
#   SSDI < 0    : solar generation below seasonal normal
#   SSDI < −1.0 : moderate deficit — initial event flag
#   SSDI < −1.5 : severe deficit   — severity filter
#
# Note on NaNs: the σ guard (see calculate_doy_climatology) sets σ → NaN
# for winter DOYs where solar CF ≈ 0 every day.  These NaN SSDI days are
# expected.  Compound events in winter are identified via SWDI; SSDI acts
# as a secondary discriminator that becomes meaningful from spring onwards.

print("Computing SSDI DOY climatology ...")
solar_mean_doy, solar_std_doy = calculate_doy_climatology(CF_solar_24h_dly)

print("Computing SSDI ...")
SSDI = calculate_standardized_index(CF_solar_24h_dly, solar_mean_doy, solar_std_doy).compute()
SSDI.name = 'SSDI'
SSDI.attrs.update({
    'long_name'       : 'Standardized Solar Deficit Index',
    'units'           : 'dimensionless (z-score)',
    'description'     : ('Deseasonalised z-score of daily 24h-mean solar CF. '
                         'Negative = below seasonal normal (solar drought). '
                         'NaN in winter when DOY σ ≈ 0.'),
    'reference'       : 'Van der Wiel et al. (2019, Environ. Res. Lett.)',
    'source_variable' : 'CF_solar — daily 24h mean of hourly solar CF (Brown et al. 2021)',
    'baseline_period' : f"{int(CF_solar_24h_dly.time.dt.year.min().values)}"
                        f"–{int(CF_solar_24h_dly.time.dt.year.max().values)}",
})

nan_frac = float(SSDI.isnull().mean().values) * 100
print(f"SSDI computed — shape: {SSDI.shape}")
print(f"  range        : {float(SSDI.min(skipna=True).values):.2f} "
      f"to {float(SSDI.max(skipna=True).values):.2f}")
print(f"  mean         : {float(SSDI.mean(skipna=True).values):.3f}  (should be ≈ 0)")
print(f"  std          : {float(SSDI.std(skipna=True).values):.3f}   (should be ≈ 1)")
print(f"  NaN fraction : {nan_frac:.1f}%  (expected ~10–25% for NL winter days)")


#%%
# ─── SSDI diagnostics ─────────────────────────────────────────────────────────

ssdi_cell = SSDI.sel(latitude=central_lat, longitude=central_lon).dropna(dim='time')

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(f'SSDI — central grid cell ({central_lat:.2f}°N, {central_lon:.2f}°E)',
             fontsize=13)

axs[0].boxplot(ssdi_cell.values, vert=True, patch_artist=True,
               boxprops={'facecolor': 'gold', 'alpha': 0.7}, showmeans=True)
axs[0].axhline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='−1 (moderate)')
axs[0].axhline(-1.5, color='red',    linestyle='--', linewidth=1.5, label='−1.5 (severe)')
axs[0].set_ylabel('SSDI')
axs[0].set_title('Box & Whisker')
axs[0].legend(fontsize=9)

axs[1].hist(ssdi_cell.values, bins=40, density=True,
            alpha=0.6, color='gold', edgecolor='k')
sns.kdeplot(ssdi_cell.values, ax=axs[1], color='darkorange', label='KDE')
axs[1].axvline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='−1 (moderate)')
axs[1].axvline(-1.5, color='red',    linestyle='--', linewidth=1.5, label='−1.5 (severe)')
axs[1].set_xlabel('SSDI')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical distribution (non-NaN days only)')
axs[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


#%%
# Seasonal DOY envelope for solar CF — this plot is particularly important
# because it shows WHY winter SSDI is NaN: the solar CF DOY std collapses
# to near zero in November–January at Netherlands latitudes (~52°N).
month_starts = [sum(calendar.monthrange(2001, m)[1] for m in range(1, i+1)) - 
                calendar.monthrange(2001, i)[1] + 1 for i in range(1, 13)]
month_names  = [calendar.month_abbr[m] for m in range(1, 13)]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f'SSDI seasonal cycle — ({central_lat:.2f}°N, {central_lon:.2f}°E)', fontsize=13)

solar_doy_mean = solar_mean_doy.sel(latitude=central_lat, longitude=central_lon).values
solar_doy_std  = solar_std_doy.sel(latitude=central_lat, longitude=central_lon).values
doy_ax         = solar_mean_doy['doy'].values

# Raw solar CF DOY mean ± 1σ
axes[0].plot(doy_ax, solar_doy_mean, color='darkorange', linewidth=2, label='DOY mean CF_solar')
axes[0].fill_between(doy_ax, solar_doy_mean - solar_doy_std, solar_doy_mean + solar_doy_std,
                     color='moccasin', alpha=0.6, label='±1 σ')
axes[0].set_ylabel('CF_solar (24h mean)')
axes[0].set_title('Raw solar CF — DOY climatology')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# DOY standard deviation — shows where σ → 0 (winter NaN region)
axes[1].plot(doy_ax, solar_doy_std, color='firebrick', linewidth=2, label='DOY σ(CF_solar)')
axes[1].axhline(1e-6, color='gray', linestyle='--', linewidth=1, label='σ guard threshold (1e-6)')
axes[1].set_ylabel('σ(CF_solar)')
axes[1].set_title('DOY std — near zero in winter → SSDI = NaN (expected)')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# SSDI DOY mean after standardisation (non-NaN DOYs only)
ssdi_doy_mean = (
    SSDI.sel(latitude=central_lat, longitude=central_lon)
    .dropna(dim='time')
    .assign_coords(doy=lambda x: x.time.dt.dayofyear)
    .groupby('doy').mean('time')
)
axes[2].plot(ssdi_doy_mean['doy'].values, ssdi_doy_mean.values,
             color='darkorange', linewidth=2, label='DOY mean SSDI (should ≈ 0)')
axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
axes[2].set_ylabel('SSDI')
axes[2].set_title('SSDI after standardisation (non-NaN DOYs only)')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

axes[2].set_xticks(month_starts)
axes[2].set_xticklabels(month_names)
axes[2].set_xlabel('Month')
plt.tight_layout()
plt.show()


#%%
# SSDI full time series

fig, ax = plt.subplots(figsize=(16, 5))

ssdi_cell.plot(ax=ax, color='gold', linewidth=0.5, alpha=0.5, label='SSDI (daily)')
ssdi_cell.rolling(time=30, center=True, min_periods=5).mean().plot(
    ax=ax, color='darkorange', linewidth=2, label='30-day rolling mean')

ax.axhline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='−1 (moderate deficit)')
ax.axhline(-1.5, color='red',    linestyle='--', linewidth=1.5, label='−1.5 (severe deficit)')
ax.axhline( 0,   color='black',  linewidth=0.5,  alpha=0.3)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('SSDI', fontsize=12)
ax.set_title(f'SSDI time series — ({central_lat:.2f}°N, {central_lon:.2f}°E)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# ─── Joint SWDI–SSDI distribution (percentile-based thresholds) ───────────────
# Thresholds are derived from each index's own marginal distribution so that
# "moderate" = below 10th percentile and "severe" = below 5th percentile.
# This is more robust than fixed z-score cuts because the empirical
# percentiles are by construction dataset-independent.

# ─── Joint SWDI–SSDI distribution (percentile-based thresholds) ───────────────

swdi_vals = SWDI.sel(latitude=central_lat, longitude=central_lon).values
ssdi_vals = SSDI.sel(latitude=central_lat, longitude=central_lon).values
time_vals = SSDI.time.values  # shared time axis

# ── Winter mask (NDJ: November, December, January) ────────────────────────────
months = pd.DatetimeIndex(time_vals).month
winter_mask = np.isin(months, [12, 1, 2])

valid_mask = np.isfinite(swdi_vals) & np.isfinite(ssdi_vals) & winter_mask
sw = swdi_vals[valid_mask]
ss = ssdi_vals[valid_mask]


# ── Percentile thresholds (marginal, computed independently) ──────────────────
MODERATE_PCT = 5   # % — initial / moderate deficit flag
SEVERE_PCT   =  1   # % — severity filter

sw_mod = np.nanpercentile(sw, MODERATE_PCT)
sw_sev = np.nanpercentile(sw, SEVERE_PCT)
ss_mod = np.nanpercentile(ss, MODERATE_PCT)
ss_sev = np.nanpercentile(ss, SEVERE_PCT)

print(f"\nPercentile thresholds at ({central_lat:.2f}°N, {central_lon:.2f}°E):")
print(f"  SWDI  {MODERATE_PCT}th pct (moderate) = {sw_mod:.3f}")
print(f"  SWDI  {SEVERE_PCT}th  pct (severe)   = {sw_sev:.3f}")
print(f"  SSDI  {MODERATE_PCT}th pct (moderate) = {ss_mod:.3f}")
print(f"  SSDI  {SEVERE_PCT}th  pct (severe)   = {ss_sev:.3f}")

# ── PMF stats at moderate threshold ───────────────────────────────────────────
p_wind  = np.mean(sw < sw_mod)
p_solar = np.mean(ss < ss_mod)
p_joint = np.mean((sw < sw_mod) & (ss < ss_mod))
pmf     = p_joint / (p_wind * p_solar) if (p_wind * p_solar) > 0 else np.nan

print(f"\nJoint distribution stats (moderate threshold = {MODERATE_PCT}th pct):")
print(f"  P(SWDI < {sw_mod:.3f})                     = {p_wind:.4f}")
print(f"  P(SSDI < {ss_mod:.3f})                     = {p_solar:.4f}")
print(f"  P(SWDI < {sw_mod:.3f} AND SSDI < {ss_mod:.3f}) = {p_joint:.4f}")
print(f"  P_independence                            = {p_wind * p_solar:.4f}")
print(f"  PMF (actual / independent)                = {pmf:.2f}  "
      f"(>1 = compound amplification; Van der Wiel et al. 2019)")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

combined_deficit = np.clip(-0.5 * sw - 0.5 * ss, 0, None)
sc = ax.scatter(sw, ss, c=combined_deficit, cmap='YlOrRd', s=3, alpha=0.35)
plt.colorbar(sc, ax=ax, label='Combined deficit (higher = more severe)')

# Moderate and severe threshold lines — separate per index
for sw_thresh, ss_thresh, col, lbl in [
    (sw_mod, ss_mod, 'orange', f'Moderate ({MODERATE_PCT}th pct)'),
    (sw_sev, ss_sev, 'red',    f'Severe ({SEVERE_PCT}th pct)'),
]:
    ax.axvline(sw_thresh, color=col, linestyle='--', linewidth=1.2, alpha=0.8,
               label=f'SWDI {lbl}: {sw_thresh:.2f}')
    ax.axhline(ss_thresh, color=col, linestyle='--', linewidth=1.2, alpha=0.8,
               label=f'SSDI {lbl}: {ss_thresh:.2f}')

# Shade moderate compound zone — boolean mask handles asymmetric thresholds
compound_mod = (sw < sw_mod) & (ss < ss_mod)
ax.scatter(sw[compound_mod], ss[compound_mod],
           color='red', s=4, alpha=0.12, zorder=0,
           label=f'Compound zone ({MODERATE_PCT}th pct)\nP_joint={p_joint:.4f}, PMF={pmf:.2f}')

ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

ax.set_xlabel('SWDI  (wind deficit ← left)', fontsize=12)
ax.set_ylabel('SSDI  (solar deficit ← down)', fontsize=12)
ax.set_title(
    f'Joint SWDI–SSDI distribution\n({central_lat:.2f}°N, {central_lon:.2f}°E)\n'
    f'n = {valid_mask.sum():,} days  |  thresholds: {MODERATE_PCT}th / {SEVERE_PCT}th pct',
    fontsize=12)
ax.set_xlim(-5, 4)
ax.set_ylim(-5, 4)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

#%%
# Save outputs 

print("\nSaving outputs ...")

# SWDI
swdi_out = Path('./../Results/SWDI')
swdi_out.mkdir(parents=True, exist_ok=True)
SWDI.to_zarr(swdi_out / 'swdi.zarr', mode='w', consolidated=True)
print(f"  SWDI saved → {swdi_out / 'swdi.zarr'}")

# SSDI
ssdi_out = Path('./../Results/SSDI')
ssdi_out.mkdir(parents=True, exist_ok=True)
SSDI.to_zarr(ssdi_out / 'ssdi.zarr', mode='w', consolidated=True)
print(f"  SSDI saved → {ssdi_out / 'ssdi.zarr'}")

# Daytime solar CF — used only in 04_DF_ID.py delta-optimisation section
cf_dt_out = Path('./../Results/CF_solar_daytime')
cf_dt_out.mkdir(parents=True, exist_ok=True)
CF_solar_daytime_dly.attrs.update({
    'long_name'   : 'Solar capacity factor — daytime hours mean',
    'description' : ('Daily mean CF_solar restricted to hours where CF_solar > 0. '
                     'Used for delta-optimisation in 04_DF_ID.py only; '
                     'NOT used for SSDI standardisation.'),
})
CF_solar_daytime_dly.to_zarr(cf_dt_out / 'CF_solar_daytime.zarr', mode='w', consolidated=True)
print(f"  CF_solar_daytime saved → {cf_dt_out / 'CF_solar_daytime.zarr'}")


# %%
