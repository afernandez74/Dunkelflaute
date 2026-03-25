#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_DF_ID.py
===========
Identify, characterise, and visualise compound dunkelflaute events using the
Standardized Wind Deficit Index (SWDI) and Standardized Solar Deficit Index
(SSDI) computed in 03_calc_SWDI_SSDI.py.

Scientific framework
--------------------
A compound dunkelflaute event is defined as a period in which BOTH wind and
solar generation are simultaneously below their seasonal normal — the
intersection (w-AND-s) type in Zscheischler et al. (2020, Nat. Rev. Earth
Environ.).  This maps directly to the Shan et al. (2024, HESS) algorithmic
framework, adapted for the energy sector:

  • Dual deficit days : SWDI < −1.0  AND  SSDI < −1.0 (initial flag)
  • Gap merge        : runs separated by ≤ MAX_GAP days are merged
                       (captures persistent blocking regimes with brief
                       interruptions)
  • Duration filter  : events shorter than MIN_DURATION days are discarded
  • Severity filter  : events with no day where SWDI < −1.5 OR SSDI < −1.5
                       are discarded (removes trivially weak co-occurrences)

No forward-looking temporal lag is applied.  Unlike the DTH (drought-to-heat)
cascade in the agricultural subproject, the physics of dunkelflaute events is
simultaneous co-suppression: blocking high-pressure systems reduce both wind
speed and solar irradiance at the same time (Grams et al. 2017, Nat. Clim.
Change), so a causal lag window is not warranted.

Delta-optimisation analysis (daytime CF)
-----------------------------------------
A supplementary analysis computes the raw system capacity factor:

    CF_sys = δ_wind × CF_wind_daily + δ_solar × CF_solar_daytime_daily

using the daytime-only solar CF (saved by 03_calc_SWDI_SSDI.py), which masks
nighttime hours where panels cannot generate power.  This provides a fair
comparison between wind and solar at hours when both resources can
simultaneously contribute, answering: "what proportion of wind vs. solar
capacity minimises low-generation events during daylight periods?"
See discussion in comments below for the rationale (previous session notes).

Inputs
------
  ./../Results/SWDI/swdi.zarr
  ./../Results/SSDI/ssdi.zarr
  ./../Results/CF_wind/          — hourly wind CF NetCDF  (for delta analysis)
  ./../Results/CF_solar_daytime/ — daily daytime solar CF (for delta analysis)

Outputs
-------
  Figures only (no intermediate data files).

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
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import label
from scipy.stats import linregress


#%%
# ─── Load SWDI and SSDI ────────────────────────────────────────────────────────

results_path = Path('./../Results/')

swdi_ds = xr.open_zarr(results_path / 'SWDI' / 'swdi.zarr', consolidated=True)
ssdi_ds = xr.open_zarr(results_path / 'SSDI' / 'ssdi.zarr', consolidated=True)

# Extract the data variables (each Zarr holds one DataArray saved as a dataset)
SWDI = swdi_ds[list(swdi_ds.data_vars)[0]]
SSDI = ssdi_ds[list(ssdi_ds.data_vars)[0]]

# Align time axes — both should cover the same period, but make it explicit
SWDI, SSDI = xr.align(SWDI, SSDI, join='inner')

# Load into memory: the event-identification loops below operate on numpy
# arrays, so we compute upfront to avoid repeated dask scheduler overhead.
SWDI = SWDI.compute()
SSDI = SSDI.compute()

print(f"SWDI loaded — shape: {SWDI.shape}")
print(f"SSDI loaded — shape: {SSDI.shape}")
print(f"Time range: {str(SWDI.time.values[0])[:10]} → {str(SWDI.time.values[-1])[:10]}")


#%%
# ─── Country shapefile for masking and plotting ────────────────────────────────

global_countries_path = Path('./../Data/countries_shp/ne_110m_admin_0_countries.shp')
countries_gdf = gpd.read_file(str(global_countries_path))

# Spatial extent for map plots
lat_min = float(SWDI.latitude.min())
lat_max = float(SWDI.latitude.max())
lon_min = float(SWDI.longitude.min())
lon_max = float(SWDI.longitude.max())
plot_region = [lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5]

# Netherlands mask for single-country statistics (expand as needed)
NL_shape = countries_gdf[countries_gdf['ADM0_A3'] == 'NLD']
NL_mask  = regionmask.Regions([NL_shape.geometry.values[0]]).mask(
    SWDI.longitude, SWDI.latitude
)


#%%
# ─── Central grid cell for time-series diagnostics ────────────────────────────

lat_idx = SWDI.latitude.size // 2
lon_idx = SWDI.longitude.size // 2
central_lat = float(SWDI.latitude.values[lat_idx])
central_lon = float(SWDI.longitude.values[lon_idx])
print(f"Diagnostic grid cell: ({central_lat:.2f}°N, {central_lon:.2f}°E)")


#%%
# ─── Event identification functions ───────────────────────────────────────────

def _merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """
    Given a 1D boolean array, merge runs of True separated by <= max_gap
    calendar days, then discard runs shorter than min_duration days.
    Returns a 1D boolean array of the same length.

    This helper mirrors the function of the same name in 03_CDHW.py (ag
    subproject), generalised here for any binary event flag.
    """
    if not np.any(flag):
        return np.zeros_like(flag, dtype=bool)

    T = len(flag)
    runs = []
    in_run = False
    start  = None

    for t in range(T):
        if flag[t] and not in_run:
            start  = t
            in_run = True
        elif not flag[t] and in_run:
            runs.append((start, t - 1))
            in_run = False
    if in_run:
        runs.append((start, T - 1))

    if not runs:
        return np.zeros_like(flag, dtype=bool)

    # Merge runs whose gap is <= max_gap
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e      # extend the current run
        else:
            merged.append([s, e])

    # Keep only runs that meet the minimum duration criterion
    out = np.zeros_like(flag, dtype=bool)
    for s, e in merged:
        if e - s + 1 >= min_duration:
            out[s : e + 1] = True
    return out


def _apply_severity_filter_1d(compound_1d, swdi_1d, ssdi_1d,
                               sev_threshold_swdi, sev_threshold_ssdi):
    """
    Remove compound event runs that contain no day severe enough.
    An event is kept only if at least one day satisfies:
        SWDI < sev_threshold_swdi  OR  SSDI < sev_threshold_ssdi

    This is the two-stage severity filter of Shan et al. (2024, HESS):
    the permissive initial threshold flags candidate periods; this filter
    ensures only statistically extreme events survive.
    """
    if not np.any(compound_1d):
        return compound_1d.copy()

    out = np.zeros_like(compound_1d, dtype=bool)
    labeled, n_events = label(compound_1d)

    for ev_id in range(1, n_events + 1):
        ev_mask = labeled == ev_id
        # Check whether this event has at least one severe day in either index
        swdi_severe = np.any(swdi_1d[ev_mask] < sev_threshold_swdi)
        ssdi_severe = np.nansum(ssdi_1d[ev_mask] < sev_threshold_ssdi) > 0
        if swdi_severe or ssdi_severe:
            out[ev_mask] = True

    return out


def identify_compound_events(
    swdi,
    ssdi,
    threshold_initial  = -1.0,
    threshold_severity = -1.5,
    max_gap            = 1,
    min_duration       = 3,
):
    """
    Full compound dunkelflaute event identification pipeline following
    Shan et al. (2024, HESS) adapted for the intersection (w-AND-s) type.

    Steps:
      1. Flag days where SWDI < threshold_initial AND SSDI < threshold_initial.
      2. Per grid cell: merge runs separated by <= max_gap days.
      3. Discard runs shorter than min_duration days.
      4. Apply severity filter: keep only events with at least one day where
         SWDI < threshold_severity OR SSDI < threshold_severity.

    Parameters
    ----------
    swdi               : xr.DataArray  (time, latitude, longitude)
    ssdi               : xr.DataArray  (time, latitude, longitude)
    threshold_initial  : float   initial dual-deficit threshold (default −1.0)
    threshold_severity : float   severity filter threshold (default −1.5)
    max_gap            : int     max days between runs to merge (default 1)
    min_duration       : int     minimum event length in days (default 3)

    Returns
    -------
    compound_mask : xr.DataArray  bool, (time, latitude, longitude)
        True on compound dunkelflaute days after all filters.
    """
    n_lat = swdi.sizes['latitude']
    n_lon = swdi.sizes['longitude']
    T     = swdi.sizes['time']

    swdi_np = swdi.values
    ssdi_np = ssdi.values

    # Step 1: raw dual-deficit flag (NaN in SSDI → not a compound day)
    raw_flag = (swdi_np < threshold_initial) & (ssdi_np < threshold_initial)

    compound_out = np.zeros((T, n_lat, n_lon), dtype=bool)

    for i in range(n_lat):
        for j in range(n_lon):
            ts = raw_flag[:, i, j]

            # Steps 2 & 3: merge gaps, discard short runs
            ts = _merge_and_filter_runs_1d(ts, max_gap, min_duration)

            # Step 4: severity filter
            ts = _apply_severity_filter_1d(
                ts,
                swdi_np[:, i, j],
                ssdi_np[:, i, j],
                threshold_severity,
                threshold_severity,
            )

            compound_out[:, i, j] = ts

    compound_mask = xr.DataArray(
        compound_out,
        coords={
            'time'     : swdi['time'],
            'latitude' : swdi['latitude'],
            'longitude': swdi['longitude'],
        },
        dims=['time', 'latitude', 'longitude'],
        attrs={
            'long_name'        : 'Compound dunkelflaute event day',
            'event_type'       : 'intersection (w-AND-s): Zscheischler et al. 2020',
            'threshold_initial': threshold_initial,
            'threshold_sev'    : threshold_severity,
            'max_gap_days'     : max_gap,
            'min_duration_days': min_duration,
        },
    )
    return compound_mask


def characterize_compound_events(compound_mask, swdi, ssdi):
    """
    Build a per-event catalogue with duration and severity metrics.

    Returns
    -------
    events_df : pd.DataFrame
        One row per event: latitude, longitude, start_date, end_date,
        duration, peak_swdi (most negative), peak_ssdi (most negative),
        mean_swdi, mean_ssdi, integrated_severity.
    """
    events     = []
    time_coord = compound_mask['time']

    for i in range(compound_mask.sizes['latitude']):
        for j in range(compound_mask.sizes['longitude']):
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            if not np.any(mask_1d):
                continue

            labeled, n_ev = label(mask_1d)
            for ev_id in range(1, n_ev + 1):
                ev_days    = labeled == ev_id
                swdi_ev    = swdi.isel(latitude=i, longitude=j).values[ev_days]
                ssdi_ev    = ssdi.isel(latitude=i, longitude=j).values[ev_days]
                times_ev   = time_coord.values[ev_days]

                # Integrated severity: sum of (|SWDI| × |SSDI|) over event days.
                # NaN-safe: use nansum so winter events with NaN SSDI are handled.
                integrated = float(np.nansum(np.abs(swdi_ev) * np.abs(ssdi_ev)))

                events.append({
                    'latitude'           : float(swdi.latitude.values[i]),
                    'longitude'          : float(swdi.longitude.values[j]),
                    'start_date'         : pd.Timestamp(times_ev[0]),
                    'end_date'           : pd.Timestamp(times_ev[-1]),
                    'duration'           : int(len(times_ev)),
                    'peak_swdi'          : float(np.nanmin(swdi_ev)),   # most negative
                    'peak_ssdi'          : float(np.nanmin(ssdi_ev)),
                    'mean_swdi'          : float(np.nanmean(swdi_ev)),
                    'mean_ssdi'          : float(np.nanmean(ssdi_ev)),
                    'integrated_severity': integrated,
                })

    return pd.DataFrame(events)


def compute_annual_metrics(compound_mask, swdi, ssdi):
    """
    Spatially averaged annual compound event statistics.

    Returns
    -------
    metrics_df : pd.DataFrame  indexed by year
        n_compound_days, integrated_severity
    """
    years   = np.unique(compound_mask.time.dt.year.values)
    metrics = {}

    for year in years:
        year_mask = compound_mask.sel(time=compound_mask.time.dt.year == year)
        year_swdi = swdi.sel(time=swdi.time.dt.year == year)
        year_ssdi = ssdi.sel(time=ssdi.time.dt.year == year)

        # Spatially averaged compound day count for this year
        n_days = float(year_mask.sum(dim='time').mean())

        # Integrated severity: |SWDI| × |SSDI| on compound days (NaN-safe)
        compound_swdi = year_swdi.where(year_mask, 0)
        compound_ssdi = year_ssdi.where(year_mask, 0)
        severity      = float(np.nanmean(np.abs(compound_swdi.values) *
                                          np.abs(compound_ssdi.values)) *
                              year_mask.sum(dim='time').mean().values)

        metrics[int(year)] = {
            'n_compound_days'     : n_days,
            'integrated_severity' : severity,
        }

    return pd.DataFrame(metrics).T


def compute_return_periods(compound_mask):
    """
    Empirical mean return period (years between events) per grid cell.

    Returns
    -------
    return_period_da : xr.DataArray  (latitude, longitude)  — years per event
    event_count_arr  : np.ndarray    (latitude, longitude)  — total events
    """
    n_years        = len(np.unique(compound_mask.time.dt.year.values))
    n_lat, n_lon   = compound_mask.sizes['latitude'], compound_mask.sizes['longitude']
    return_periods = np.full((n_lat, n_lon), np.nan)
    event_counts   = np.zeros((n_lat, n_lon), dtype=int)

    for i in range(n_lat):
        for j in range(n_lon):
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            _, n_ev = label(mask_1d)
            event_counts[i, j] = n_ev
            if n_ev > 0:
                return_periods[i, j] = n_years / n_ev

    return_period_da = xr.DataArray(
        return_periods,
        coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
        dims=['latitude', 'longitude'],
        attrs={'long_name': 'Empirical mean return period', 'units': 'years/event'},
    )
    return return_period_da, event_counts


#%%
# ─── Run event identification ──────────────────────────────────────────────────
# Default parameters follow Shan et al. (2024) adapted for dunkelflaute.
# Sensitivity tests (threshold, gap, duration) should be run in a separate
# dedicated script or as a loop here if needed.

THRESHOLD_INITIAL  = -1.0   # both SWDI and SSDI must be below this (initial flag)
THRESHOLD_SEVERITY = -1.5   # at least one index must reach this (severity filter)
MAX_GAP            =  1     # merge events separated by ≤ 1 day
MIN_DURATION       =  3     # discard events shorter than 3 days

print("Identifying compound dunkelflaute events ...")
compound_mask = identify_compound_events(
    SWDI, SSDI,
    threshold_initial  = THRESHOLD_INITIAL,
    threshold_severity = THRESHOLD_SEVERITY,
    max_gap            = MAX_GAP,
    min_duration       = MIN_DURATION,
)

n_total_days = int(compound_mask.sum())
frac_days    = float(compound_mask.mean()) * 100
print(f"Compound mask complete — {n_total_days:,} compound event days total")
print(f"  = {frac_days:.2f}% of all (time × grid cell) combinations")


#%%
# ─── Event catalogue ──────────────────────────────────────────────────────────

events_df = characterize_compound_events(compound_mask, SWDI, SSDI)
print(f"\nTotal compound events catalogued: {len(events_df)}")
print(events_df[['start_date', 'end_date', 'duration',
                  'peak_swdi', 'peak_ssdi', 'integrated_severity']].head(10))

# Duration distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(events_df['duration'], bins=np.arange(0.5, 30.5, 1),
        color='steelblue', edgecolor='k', alpha=0.7)
ax.set_xlabel('Event duration (days)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of compound dunkelflaute event durations', fontsize=13)
ax.axvline(events_df['duration'].mean(), color='red', linestyle='--', linewidth=1.5,
           label=f"Mean = {events_df['duration'].mean():.1f} d")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# ─── Time-series plot for central grid cell ────────────────────────────────────
# This panel plot mirrors the SMDI/SVDI time-series diagnostic in the ag
# subproject (03_CDHW.py).  Compound event periods are shaded grey.

compound_days_cell = compound_mask.isel(latitude=lat_idx, longitude=lon_idx).values
swdi_cell          = SWDI.isel(latitude=lat_idx, longitude=lon_idx)
ssdi_cell          = SSDI.isel(latitude=lat_idx, longitude=lon_idx)
dates              = SWDI.time.values

fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

# Shade compound event periods on both panels
for ax in axes:
    ax.fill_between(
        dates, -8, 8,
        where=compound_days_cell,
        color='gray', alpha=0.25, zorder=0,
        label='Compound event days' if ax is axes[0] else '_nolegend_',
    )

# ── SWDI panel ─────────────────────────────────────────────────────────────────
axes[0].plot(dates, swdi_cell.values,
             color='steelblue', linewidth=0.6, alpha=0.5, label='SWDI (daily)', zorder=1)
axes[0].plot(dates, swdi_cell.rolling(time=30, center=True).mean().values,
             color='navy', linewidth=2.5, label='30-day rolling mean', zorder=2)
axes[0].axhline(THRESHOLD_INITIAL,  color='orange', linestyle='--', linewidth=1.8,
                label=f'Initial threshold ({THRESHOLD_INITIAL})')
axes[0].axhline(THRESHOLD_SEVERITY, color='red',    linestyle='--', linewidth=1.8,
                label=f'Severity threshold ({THRESHOLD_SEVERITY})')
axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.3)
axes[0].set_ylabel('SWDI', fontsize=14)
axes[0].set_title(
    f'Compound dunkelflaute events — ({central_lat:.2f}°N, {central_lon:.2f}°E)',
    fontsize=15, fontweight='bold',
)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=12)

# ── SSDI panel ─────────────────────────────────────────────────────────────────
axes[1].plot(dates, ssdi_cell.values,
             color='gold', linewidth=0.6, alpha=0.5, label='SSDI (daily)', zorder=1)
axes[1].plot(dates, ssdi_cell.rolling(time=30, center=True, min_periods=5).mean().values,
             color='darkorange', linewidth=2.5, label='30-day rolling mean', zorder=2)
axes[1].axhline(THRESHOLD_INITIAL,  color='orange', linestyle='--', linewidth=1.8,
                label=f'Initial threshold ({THRESHOLD_INITIAL})')
axes[1].axhline(THRESHOLD_SEVERITY, color='red',    linestyle='--', linewidth=1.8,
                label=f'Severity threshold ({THRESHOLD_SEVERITY})')
axes[1].axhline(0, color='black', linewidth=0.5, alpha=0.3)
axes[1].set_ylabel('SSDI', fontsize=14)
axes[1].set_xlabel('Date', fontsize=14)
axes[1].legend(fontsize=10, loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=12)

plt.tight_layout()
plt.show()


#%%
# ─── Annual metrics and trends ────────────────────────────────────────────────

annual_metrics = compute_annual_metrics(compound_mask, SWDI, SSDI)
years_arr      = annual_metrics.index.values.astype(float)

# Dual-axis: compound days per year (bars) + integrated severity (line)
fig, ax1 = plt.subplots(figsize=(15, 6))

ax1.bar(annual_metrics.index, annual_metrics['n_compound_days'],
        alpha=0.6, color='steelblue', label='Compound days/year (spatial avg)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Compound days per year (spatial avg)', fontsize=12, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(annual_metrics.index, annual_metrics['integrated_severity'],
         color='navy', linewidth=2.5, marker='o', markersize=4,
         label='Integrated severity')
ax2.set_ylabel('Integrated severity (|SWDI|×|SSDI|)', fontsize=12, color='navy')
ax2.tick_params(axis='y', labelcolor='navy')

# Annotate known major dunkelflaute years from Li et al. (2021)
for yr, lbl in {2016: 'Dec 2016', 2017: 'Jan 2017', 2021: 'Dec 2021'}.items():
    ax1.axvline(yr, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.text(yr + 0.1, ax1.get_ylim()[1] * 0.93, lbl,
             ha='left', fontsize=9, style='italic', color='black')

yr0 = int(SWDI.time.dt.year.min().values)
yr1 = int(SWDI.time.dt.year.max().values)
ax1.set_title(f'Compound Dunkelflaute Events ({yr0}–{yr1})',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()


#%%
# Trend analysis: 10-year rolling mean + linear regression
# (mirrors the trend plot in 03_CDHW.py from the ag subproject)

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

rolling_days = annual_metrics['n_compound_days'].rolling(10, center=True).mean()
rolling_sev  = annual_metrics['integrated_severity'].rolling(10, center=True).mean()

# ── Compound days panel ────────────────────────────────────────────────────────
axes[0].plot(annual_metrics.index, annual_metrics['n_compound_days'],
             'o-', alpha=0.3, color='gray', markersize=4, label='Annual')
axes[0].plot(rolling_days.index, rolling_days.values,
             linewidth=3, color='steelblue', label='10-year rolling mean')

slope, intercept, _, p, _ = linregress(years_arr, annual_metrics['n_compound_days'])
axes[0].plot(years_arr, slope * years_arr + intercept,
             '--', color='black', linewidth=2,
             label=f'Trend: {slope:.3f} days/yr  (p = {p:.3f})')
axes[0].set_ylabel('Compound days/year', fontsize=12)
axes[0].set_title('Temporal evolution of compound dunkelflaute events',
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.3)

# ── Integrated severity panel ──────────────────────────────────────────────────
valid       = ~np.isnan(annual_metrics['integrated_severity'])
yrs_valid   = years_arr[valid]
sev_valid   = annual_metrics['integrated_severity'].values[valid]

axes[1].plot(annual_metrics.index, annual_metrics['integrated_severity'],
             'o-', alpha=0.3, color='gray', markersize=4, label='Annual')
axes[1].plot(rolling_sev.index, rolling_sev.values,
             linewidth=3, color='navy', label='10-year rolling mean')

sl_s, ic_s, _, p_s, _ = linregress(yrs_valid, sev_valid)
axes[1].plot(years_arr, sl_s * years_arr + ic_s,
             '--', color='black', linewidth=2, label=f'Trend: p = {p_s:.3f}')
axes[1].set_ylabel('Integrated severity', fontsize=12)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%%
# ─── Seasonal analysis ────────────────────────────────────────────────────────
# Dunkelflaute events are predominantly a winter phenomenon in NW Europe.
# The seasonal heatmap shows which calendar months carry the most compound event
# days, averaged across grid cells.  Shan et al. (2024) provide a direct
# methodological precedent for seasonal stratification of compound event results.

# Monthly compound-day counts: average over space, sum over each year×month cell
monthly_counts = (
    compound_mask.mean(dim=['latitude', 'longitude'])   # spatial mean fraction
    .resample(time='1ME').sum()                          # days per calendar month
    .rename('compound_days')
)

monthly_df = pd.DataFrame({
    'year' : monthly_counts.time.dt.year.values,
    'month': monthly_counts.time.dt.month.values,
    'days' : monthly_counts.values,
})

# Pivot to year × month heatmap
heatmap_data = monthly_df.pivot(index='year', columns='month', values='days')
heatmap_data.columns = [calendar.month_abbr[m] for m in heatmap_data.columns]

fig, ax = plt.subplots(figsize=(14, max(6, len(heatmap_data) * 0.28)))
sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap='YlOrRd',
    linewidths=0.3,
    linecolor='white',
    cbar_kws={'label': 'Compound days (spatial avg)'},
)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Year', fontsize=12)
ax.set_title('Seasonal distribution of compound dunkelflaute days\n'
             '(spatially averaged)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Bar chart: mean compound days per calendar month (climatological seasonality)
monthly_mean = monthly_df.groupby('month')['days'].mean()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(1, 13), monthly_mean.values,
              color='steelblue', edgecolor='k', alpha=0.75)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Mean compound days/month (spatial avg)', fontsize=12)
ax.set_title('Climatological seasonal cycle of compound dunkelflaute\n'
             f'({yr0}–{yr1})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


#%%
# ─── Return period spatial maps ───────────────────────────────────────────────
# Empirical mean return period = record length (years) / event count per cell.
# Cells with very few or zero events are masked out for clarity.

return_periods, event_counts = compute_return_periods(compound_mask)

# Mask cells with fewer than 2 events (return period estimate unreliable)
rp_masked = return_periods.where(return_periods < 50)   # cap at 50 yr for colour scale
ec_da = xr.DataArray(
    np.where(event_counts >= 2, event_counts, np.nan),
    coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
    dims=['latitude', 'longitude'],
)

fig, axes = plt.subplots(1, 2, figsize=(17, 7),
                         subplot_kw={'projection': ccrs.LambertConformal(
                             central_longitude=5, central_latitude=52)})

for ax in axes:
    ax.set_extent(plot_region, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor='whitesmoke',  zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='aliceblue',   zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0,            zorder=2)
    ax.add_feature(cfeature.BORDERS,   linestyle=':', linewidth=0.8, zorder=2)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3,
                      linestyle='--', linewidth=0.5)
    gl.top_labels   = False
    gl.right_labels = False

# Return periods
rp_masked.plot(
    ax=axes[0], transform=ccrs.PlateCarree(),
    cmap='RdYlGn_r',
    cbar_kwargs={'label': 'Mean return period (years)', 'shrink': 0.8},
)
axes[0].set_title('Compound dunkelflaute\nMean return period (lower = more frequent)',
                  fontsize=12, fontweight='bold')

# Total event counts
ec_da.plot(
    ax=axes[1], transform=ccrs.PlateCarree(),
    cmap='YlOrRd',
    cbar_kwargs={'label': f'Total compound events ({yr0}–{yr1})', 'shrink': 0.8},
)
axes[1].set_title(f'Compound dunkelflaute\nTotal events ({yr0}–{yr1})',
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


#%%
# Return period map masked to NL only (for Netherlands-focused results)

rp_nl = rp_masked.where(NL_mask == 0)

fig = plt.figure(figsize=(9, 8))
ax  = fig.add_subplot(1, 1, 1,
                      projection=ccrs.LambertConformal(central_longitude=5, central_latitude=52))
ax.set_extent(plot_region, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND,      facecolor='whitesmoke', zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor='aliceblue',  zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=1.2,           zorder=2)
ax.add_feature(cfeature.BORDERS,   linestyle=':', linewidth=0.9, zorder=2)

rp_nl.plot(
    ax=ax, transform=ccrs.PlateCarree(),
    cmap='RdYlGn_r',
    cbar_kwargs={'label': 'Mean return period (years)', 'shrink': 0.8},
)
ax.set_title('Compound dunkelflaute return period — Netherlands',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


#%%
# ─── Raw index maps: spatial distribution of mean SWDI and SSDI ───────────────
# These maps show the spatial structure of the drivers themselves before
# compound event filtering.  Areas of persistently low SWDI (e.g. inland,
# sheltered from westerlies) versus persistently low SSDI (e.g. frequently
# overcast coastal zones) help contextualise the return-period patterns.

swdi_mean = SWDI.mean(dim='time')
ssdi_mean = SSDI.mean(dim='time', skipna=True)

fig, axes = plt.subplots(1, 2, figsize=(17, 7),
                         subplot_kw={'projection': ccrs.LambertConformal(
                             central_longitude=5, central_latitude=52)})

for ax in axes:
    ax.set_extent(plot_region, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,  facecolor='whitesmoke',  zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue',   zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0,        zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8, zorder=2)

swdi_mean.plot(
    ax=axes[0], transform=ccrs.PlateCarree(),
    cmap='RdBu', center=0,
    cbar_kwargs={'label': 'Mean SWDI (dimensionless)', 'shrink': 0.8},
)
axes[0].set_title('Long-run mean SWDI\n(blue = wind-rich; red = wind-poor)', fontsize=12)

ssdi_mean.plot(
    ax=axes[1], transform=ccrs.PlateCarree(),
    cmap='RdBu', center=0,
    cbar_kwargs={'label': 'Mean SSDI (dimensionless)', 'shrink': 0.8},
)
axes[1].set_title('Long-run mean SSDI\n(blue = solar-rich; red = solar-poor)', fontsize=12)

plt.suptitle(f'Mean standardised deficit indices ({yr0}–{yr1})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


#%%
# ─── Delta-optimisation analysis (daytime CF) ─────────────────────────────────
#
# This section answers: "among hours when both wind and solar can generate power,
# what proportion of each minimises the frequency of low system-CF events?"
#
# We use the daytime-only solar CF (hours where CF_solar > 0) so that nighttime
# structural zeros do not artificially suppress the solar term relative to wind.
# This gives a fair comparison between the two resources at hours when both can
# simultaneously contribute to generation.
#
# The system CF is CF_sys = δ_wind × CF_wind + δ_solar × CF_solar_daytime,
# restricted to daytime hours (days where CF_solar_daytime is non-NaN).
# For each δ_solar we count dunkelflaute events using the same decluster logic
# from DF_eventCount_ReturnInterval.py, and find the δ_solar that minimises
# event count per grid cell.
#
# NOTE: this analysis is SEPARATE from the SWDI/SSDI compound event analysis
# above.  It uses raw CF thresholds on a weighted system CF, not standardised
# indices.  The two framings answer different but complementary questions.

print("\nRunning delta-optimisation analysis ...")

# Load hourly wind CF and resample to daily (same as in 03_calc_SWDI_SSDI.py)
CF_wind_path  = Path('./../Results/CF_wind/')
CF_wind_file  = sorted(CF_wind_path.glob('*.nc'))[0]
CF_wind_hr    = xr.open_dataarray(CF_wind_file)
if 'valid_time' in CF_wind_hr.dims:
    CF_wind_hr = CF_wind_hr.rename({'valid_time': 'time'})
CF_wind_dly = CF_wind_hr.resample(time='1D').mean()

# Load daytime solar CF (saved by 03_calc_SWDI_SSDI.py)
CF_solar_dt_path  = Path('./../Results/CF_solar_daytime/CF_solar_daytime.zarr')
CF_solar_dt_ds    = xr.open_zarr(CF_solar_dt_path, consolidated=True)
CF_solar_dt_dly   = CF_solar_dt_ds[list(CF_solar_dt_ds.data_vars)[0]].compute()

# Align time axes (wind daily vs daytime solar daily should match, but be explicit)
CF_wind_dly, CF_solar_dt_dly = xr.align(CF_wind_dly, CF_solar_dt_dly, join='inner')

# Restrict analysis to days where solar can contribute (daytime non-NaN days only)
# This ensures the comparison is between generating periods, not a whole-year average.
daytime_days = CF_solar_dt_dly.notnull()


def decluster_events_1d(cf_ts, threshold, min_duration, min_gap):
    """
    Count declustered below-threshold events in a 1D CF time series.
    Mirrors the _decluster_1d helper in DF_eventCount_ReturnInterval.py.
    Returns (event_count, mean_duration).
    """
    mask  = cf_ts <= threshold
    if not mask.any():
        return 0, np.nan

    diff   = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    events = [[s, e] for s, e in zip(starts, ends) if e - s >= min_duration]
    if not events:
        return 0, np.nan

    merged = [events[0]]
    for s, e in events[1:]:
        if s - merged[-1][1] <= min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    durations = [e - s for s, e in merged]
    return len(merged), float(np.mean(durations))


# Grid search over δ_solar ∈ [0.0, 1.0] at 0.05 increments
d_solar_range = np.arange(0, 1.05, 0.05)

# Parameters for the raw-CF event counting (deliberately simple thresholds)
RAW_CF_THRESHOLD  = 0.10   # system CF below 10% = stress event
RAW_MIN_DURATION  = 3      # minimum 3 consecutive days
RAW_MIN_GAP       = 1      # merge events within 1 day

n_lat = CF_wind_dly.sizes['latitude']
n_lon = CF_wind_dly.sizes['longitude']

# Initialise storage arrays
opt_event_count = np.full((n_lat, n_lon), np.inf)
opt_d_solar     = np.full((n_lat, n_lon), np.nan)

for d_solar in d_solar_range:
    d_wind = 1.0 - d_solar

    # System CF restricted to daytime hours for the comparison
    CF_sys = (d_wind * CF_wind_dly + d_solar * CF_solar_dt_dly).where(daytime_days)

    # Apply decluster per grid cell using apply_ufunc
    # We extract counts only (index 0 of the 2-element return)
    results = xr.apply_ufunc(
        lambda ts: np.array([decluster_events_1d(
            ts[np.isfinite(ts)], RAW_CF_THRESHOLD, RAW_MIN_DURATION, RAW_MIN_GAP
        )[0]]),
        CF_sys,
        input_core_dims=[['time']],
        output_core_dims=[['metric']],
        vectorize=True,
        output_dtypes=[float],
        output_sizes={'metric': 1},
    ).isel(metric=0)

    ev_count_np = results.values

    # Update grid cells where this δ gives fewer events
    improve_mask = ev_count_np < opt_event_count
    opt_event_count = np.where(improve_mask, ev_count_np, opt_event_count)
    opt_d_solar     = np.where(improve_mask, d_solar,     opt_d_solar)

print("Delta optimisation complete.")
print(f"  Optimal δ_solar (spatial mean): {np.nanmean(opt_d_solar):.3f}")
print(f"  Optimal δ_solar (spatial std) : {np.nanstd(opt_d_solar):.3f}")

# Package results into DataArrays for plotting
opt_d_solar_da = xr.DataArray(
    np.where(np.isfinite(opt_event_count), opt_d_solar, np.nan),
    coords={'latitude': CF_wind_dly.latitude, 'longitude': CF_wind_dly.longitude},
    dims=['latitude', 'longitude'],
    attrs={'long_name': 'Optimal solar share δ_solar (daytime CF analysis)',
           'units'    : 'fraction [0, 1]'},
)
opt_events_da = xr.DataArray(
    np.where(opt_event_count < np.inf, opt_event_count, np.nan),
    coords={'latitude': CF_wind_dly.latitude, 'longitude': CF_wind_dly.longitude},
    dims=['latitude', 'longitude'],
    attrs={'long_name': 'Minimum event count at optimal δ_solar'},
)


#%%
# ─── Delta-optimisation maps ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(17, 7),
                         subplot_kw={'projection': ccrs.LambertConformal(
                             central_longitude=5, central_latitude=52)})

for ax in axes:
    ax.set_extent(plot_region, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor='whitesmoke', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='aliceblue',  zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0,           zorder=2)
    ax.add_feature(cfeature.BORDERS,   linestyle=':', linewidth=0.8, zorder=2)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3,
                      linestyle='--', linewidth=0.5)
    gl.top_labels   = False
    gl.right_labels = False

opt_d_solar_da.plot(
    ax=axes[0], transform=ccrs.PlateCarree(),
    cmap='RdYlGn', vmin=0, vmax=1,
    cbar_kwargs={'label': 'Optimal solar share δ_solar', 'shrink': 0.8},
)
axes[0].set_title(
    'Optimal solar share δ_solar\n(daytime CF, minimising low-CF events)',
    fontsize=12, fontweight='bold',
)
axes[0].set_title(
    'Interpretation: δ_solar → 0.5 signals diversification benefit;\n'
    'δ_solar → 0 means wind dominates daytime resilience',
    fontsize=9, loc='left', style='italic', pad=2,
)

opt_events_da.plot(
    ax=axes[1], transform=ccrs.PlateCarree(),
    cmap='YlOrRd_r',
    cbar_kwargs={'label': 'Min event count at optimal δ', 'shrink': 0.8},
)
axes[1].set_title('Minimum achievable event count\nat optimal δ_solar',
                  fontsize=12, fontweight='bold')

plt.suptitle(
    f'Delta-optimisation: daytime CF wind vs. solar\n'
    f'(CF_sys = δ_wind × CF_wind + δ_solar × CF_solar_daytime, threshold = {RAW_CF_THRESHOLD})',
    fontsize=12, fontweight='bold', y=1.01,
)
plt.tight_layout()
plt.show()

# Histogram of optimal δ_solar across all grid cells
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(opt_d_solar_da.values.flatten()[np.isfinite(opt_d_solar_da.values.flatten())],
        bins=np.arange(-0.025, 1.075, 0.05),
        color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(np.nanmean(opt_d_solar_da.values), color='red', linestyle='--', linewidth=2,
           label=f"Mean = {np.nanmean(opt_d_solar_da.values):.2f}")
ax.set_xlabel('Optimal δ_solar', fontsize=12)
ax.set_ylabel('Grid cell count', fontsize=12)
ax.set_title('Distribution of optimal solar share across the domain\n'
             '(daytime CF analysis)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# ─── Summary statistics ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("COMPOUND DUNKELFLAUTE EVENT SUMMARY")
print("=" * 60)
print(f"Record length          : {yr0}–{yr1}  ({yr1-yr0+1} years)")
print(f"Domain                 : {lat_min:.1f}–{lat_max:.1f}°N, "
      f"{lon_min:.1f}–{lon_max:.1f}°E")
print(f"Threshold (initial)    : SWDI < {THRESHOLD_INITIAL}  AND  "
      f"SSDI < {THRESHOLD_INITIAL}")
print(f"Threshold (severity)   : SWDI < {THRESHOLD_SEVERITY}  OR  "
      f"SSDI < {THRESHOLD_SEVERITY}")
print(f"Min duration           : {MIN_DURATION} days")
print(f"Max gap to merge       : {MAX_GAP} day(s)")
print(f"Total events catalogued: {len(events_df):,}")
if len(events_df) > 0:
    print(f"Mean event duration    : {events_df['duration'].mean():.1f} days")
    print(f"Max event duration     : {events_df['duration'].max()} days")
    print(f"Mean peak SWDI         : {events_df['peak_swdi'].mean():.2f}")
    print(f"Mean peak SSDI         : {events_df['peak_ssdi'].mean():.2f}")
print(f"Mean spatial return pd : {float(return_periods.mean(skipna=True).values):.1f} years")
print("=" * 60)
