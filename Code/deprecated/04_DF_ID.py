#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_DF_ID.py
===========
Identify and characterise compound dunkelflaute events using SWDI and SSDI.
Thresholds are derived from the empirical marginal distributions of each index
(percentile-based) rather than fixed z-score values.
"""

#%%
# Imports

import calendar
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
import geopandas as gpd
import rioxarray

#%%
# ─── Load SWDI and SSDI ────────────────────────────────────────────────────────

results_path = Path('./../Results/')

swdi_ds = xr.open_zarr(results_path / 'SWDI' / 'swdi.zarr', consolidated=True)
ssdi_ds = xr.open_zarr(results_path / 'SSDI' / 'ssdi.zarr', consolidated=True)

global_countries_path = Path('~/CDHW_ag/Data/countries_shp/ne_110m_admin_0_countries.shp')

SWDI = swdi_ds[list(swdi_ds.data_vars)[0]]
SSDI = ssdi_ds[list(ssdi_ds.data_vars)[0]]

SWDI, SSDI = xr.align(SWDI, SSDI, join='inner')

countries_of_interest = ['France', 'Belgium', 'Netherlands', 'Germany']

countries_gdf = gpd.read_file(str(global_countries_path))

region_parts = (
    countries_gdf[countries_gdf['SOVEREIGNT'].isin(countries_of_interest)]
    .dissolve().explode(index_parts=True).reset_index(drop=True)
)
# Remove Corsica
corsica_bbox = (8.5, 41.3, 9.6, 43.1)
is_corsica = region_parts.geometry.apply(
    lambda g: (corsica_bbox[0] <= g.centroid.x <= corsica_bbox[2] and
               corsica_bbox[1] <= g.centroid.y <= corsica_bbox[3])
)
region = region_parts[~is_corsica].dissolve()

def clip_to_region(da, region_gdf):
    da = da.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(region_gdf.geometry, region_gdf.crs,
                       all_touched=False, drop=False)

SWDI = clip_to_region(SWDI, region)
SSDI = clip_to_region(SSDI, region)

n_land = int(np.isfinite(SWDI.isel(time=0).values).sum())
print(f"Land cells after clip: {n_land:,}")

SWDI = SWDI.compute()
SSDI = SSDI.compute()

print(f"SWDI loaded — shape: {SWDI.shape}")
print(f"SSDI loaded — shape: {SSDI.shape}")
print(f"Time range: {str(SWDI.time.values[0])[:10]} → {str(SWDI.time.values[-1])[:10]}")


#%%
#%%
# ─── Percentile-based thresholds (per grid cell) ──────────────────────────────
# Percentiles are computed independently at each grid cell along the time axis.
# This means the threshold at each location reflects that cell's own distribution
# rather than the domain-wide pooled distribution, which is more appropriate
# given spatial gradients in mean wind and solar resource across the domain.
# Ocean/masked cells produce NaN thresholds and are naturally skipped in the
# event identification loop.

INITIAL_PCT  = 10
SEVERITY_PCT =  5

# Shape: (latitude, longitude) — one threshold value per grid cell
THRESH_SWDI_INITIAL  = np.nanpercentile(SWDI.values, INITIAL_PCT,  axis=0)
THRESH_SWDI_SEVERITY = np.nanpercentile(SWDI.values, SEVERITY_PCT, axis=0)
THRESH_SSDI_INITIAL  = np.nanpercentile(SSDI.values, INITIAL_PCT,  axis=0)
THRESH_SSDI_SEVERITY = np.nanpercentile(SSDI.values, SEVERITY_PCT, axis=0)

# Wrap in DataArrays for inspection / saving
def _thresh_da(arr, name, pct):
    return xr.DataArray(arr,
        coords={'latitude': SWDI.latitude, 'longitude': SWDI.longitude},
        dims=['latitude', 'longitude'],
        attrs={'long_name': name, 'percentile': pct})

thresh_swdi_init_da = _thresh_da(THRESH_SWDI_INITIAL,  'SWDI initial threshold',  INITIAL_PCT)
thresh_ssdi_init_da = _thresh_da(THRESH_SSDI_INITIAL,  'SSDI initial threshold',  INITIAL_PCT)

print(f"Per-cell SWDI {INITIAL_PCT}th pct — mean: {np.nanmean(THRESH_SWDI_INITIAL):.3f}, "
      f"std: {np.nanstd(THRESH_SWDI_INITIAL):.3f}")
print(f"Per-cell SSDI {INITIAL_PCT}th pct — mean: {np.nanmean(THRESH_SSDI_INITIAL):.3f}, "
      f"std: {np.nanstd(THRESH_SSDI_INITIAL):.3f}")

MAX_GAP      = 1
MIN_DURATION = 3
#%%
# ─── Event identification functions ───────────────────────────────────────────

def _merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """
    Given a 1D boolean array, merge runs of True separated by <= max_gap
    calendar days, then discard runs shorter than min_duration days.
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


def _apply_severity_filter_1d(compound_1d, swdi_1d, ssdi_1d,
                               sev_thresh_swdi, sev_thresh_ssdi):
    """
    Remove compound event runs that contain no day severe enough.
    An event is kept only if at least one day satisfies:
        SWDI < sev_thresh_swdi  OR  SSDI < sev_thresh_ssdi

    Per-index severity thresholds are accepted separately to correctly
    handle asymmetric percentile-derived cuts.
    """
    if not np.any(compound_1d):
        return compound_1d.copy()

    out = np.zeros_like(compound_1d, dtype=bool)
    labeled, n_events = label(compound_1d)

    for ev_id in range(1, n_events + 1):
        ev_mask     = labeled == ev_id
        swdi_severe = np.any(swdi_1d[ev_mask] < sev_thresh_swdi)
        ssdi_severe = np.nansum(ssdi_1d[ev_mask] < sev_thresh_ssdi) > 0
        if swdi_severe or ssdi_severe:
            out[ev_mask] = True

    return out


def identify_compound_events(
    swdi,
    ssdi,
    thresh_swdi_initial,    # now (latitude, longitude) np.ndarray or scalar
    thresh_ssdi_initial,
    thresh_swdi_severity,
    thresh_ssdi_severity,
    max_gap      = 1,
    min_duration = 3,
):
    n_lat = swdi.sizes['latitude']
    n_lon = swdi.sizes['longitude']
    T     = swdi.sizes['time']

    swdi_np = swdi.values
    ssdi_np = ssdi.values

    # Broadcast scalar thresholds to 2D for uniform indexing below
    def _broadcast(thresh):
        if np.ndim(thresh) == 0:
            return np.full((n_lat, n_lon), thresh)
        return np.asarray(thresh)

    th_sw_ini = _broadcast(thresh_swdi_initial)
    th_ss_ini = _broadcast(thresh_ssdi_initial)
    th_sw_sev = _broadcast(thresh_swdi_severity)
    th_ss_sev = _broadcast(thresh_ssdi_severity)

    compound_out = np.zeros((T, n_lat, n_lon), dtype=bool)

    for i in range(n_lat):
        for j in range(n_lon):
            # Skip masked / ocean cells — any NaN threshold means no data here
            if np.isnan(th_sw_ini[i, j]) or np.isnan(th_ss_ini[i, j]):
                continue

            raw_flag = (
                (swdi_np[:, i, j] < th_sw_ini[i, j]) &
                (ssdi_np[:, i, j] < th_ss_ini[i, j])
            )

            ts = _merge_and_filter_runs_1d(raw_flag, max_gap, min_duration)
            ts = _apply_severity_filter_1d(
                ts,
                swdi_np[:, i, j],
                ssdi_np[:, i, j],
                th_sw_sev[i, j],
                th_ss_sev[i, j],
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
            'long_name'           : 'Compound dunkelflaute event day',
            'event_type'          : 'intersection (w-AND-s): Zscheischler et al. 2020',
            'initial_percentile'  : INITIAL_PCT,
            'severity_percentile' : SEVERITY_PCT,
            'max_gap_days'        : max_gap,
            'min_duration_days'   : min_duration,
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
        duration, peak_swdi, peak_ssdi, mean_swdi, mean_ssdi,
        integrated_severity.
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
                ev_days  = labeled == ev_id
                swdi_ev  = swdi.isel(latitude=i, longitude=j).values[ev_days]
                ssdi_ev  = ssdi.isel(latitude=i, longitude=j).values[ev_days]
                times_ev = time_coord.values[ev_days]

                integrated = float(np.nansum(np.abs(swdi_ev) * np.abs(ssdi_ev)))

                events.append({
                    'latitude'           : float(swdi.latitude.values[i]),
                    'longitude'          : float(swdi.longitude.values[j]),
                    'start_date'         : pd.Timestamp(times_ev[0]),
                    'end_date'           : pd.Timestamp(times_ev[-1]),
                    'duration'           : int(len(times_ev)),
                    'peak_swdi'          : float(np.nanmin(swdi_ev)),
                    'peak_ssdi'          : float(np.nanmin(ssdi_ev)),
                    'mean_swdi'          : float(np.nanmean(swdi_ev)),
                    'mean_ssdi'          : float(np.nanmean(ssdi_ev)),
                    'integrated_severity': integrated,
                })

    return pd.DataFrame(events)


def compute_return_periods(compound_mask):
    """
    Empirical mean return period (years between events) per grid cell.

    Returns
    -------
    return_period_da : xr.DataArray  (latitude, longitude)  — years per event
    event_count_arr  : np.ndarray    (latitude, longitude)  — total events
    """
    n_years      = len(np.unique(compound_mask.time.dt.year.values))
    n_lat, n_lon = compound_mask.sizes['latitude'], compound_mask.sizes['longitude']
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

print("Identifying compound dunkelflaute events ...")
compound_mask = identify_compound_events(
    SWDI, SSDI,
    thresh_swdi_initial  = THRESH_SWDI_INITIAL,
    thresh_ssdi_initial  = THRESH_SSDI_INITIAL,
    thresh_swdi_severity = THRESH_SWDI_SEVERITY,
    thresh_ssdi_severity = THRESH_SSDI_SEVERITY,
    max_gap              = MAX_GAP,
    min_duration         = MIN_DURATION,
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


#%%
# ─── Return periods ───────────────────────────────────────────────────────────

return_periods, event_counts = compute_return_periods(compound_mask)

yr0     = int(SWDI.time.dt.year.min().values)
yr1     = int(SWDI.time.dt.year.max().values)
n_years = yr1 - yr0 + 1

mean_events_per_year = np.nanmean(event_counts) / n_years
mean_return_period   = float(return_periods.mean(skipna=True).values)

print(f"\nReturn period summary ({yr0}–{yr1}, {n_years} years):")
print(f"  Mean events per year (spatial avg) : {mean_events_per_year:.2f}")
print(f"  Mean return period   (spatial avg) : {mean_return_period:.1f} years/event")
print(f"  Min return period                  : {float(return_periods.min(skipna=True).values):.1f} years")
print(f"  Max return period (≤50 yr cells)   : {float(return_periods.where(return_periods <= 50).max(skipna=True).values):.1f} years")

#%%
#%%
# ─── Spatial map: mean compound dunkelflaute events per year ──────────────────

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# Mean events per year per grid cell
events_per_year = xr.DataArray(
    event_counts / n_years,
    coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
    dims=['latitude', 'longitude'],
    attrs={'long_name': 'Mean compound dunkelflaute events per year', 'units': 'events/year'},
)

# Mask cells with zero events — no information to show
events_per_year = events_per_year.where(events_per_year > 0)

# ── Discrete colour scale ──────────────────────────────────────────────────────
# Boundaries chosen so each bin represents a meaningful climatological frequency.
# The upper bound (2.0) comfortably accommodates the expected range for NW Europe;
# adjust if the domain-mean suggests otherwise.
bounds = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
cmap   = plt.get_cmap('YlOrRd', len(bounds) - 1)
norm   = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)

# ── Figure ─────────────────────────────────────────────────────────────────────
proj = ccrs.LambertConformal(central_longitude=5, central_latitude=52)

fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={'projection': proj})

# Extent: covers NL, BE, DE, FR; widen if domain is broader
ax.set_extent([-5, 16, 46, 56], crs=ccrs.PlateCarree())

# Background features
ax.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor='#d6e8f5', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8,        zorder=3)
ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle=':', zorder=3)
ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor='steelblue',
               alpha=0.4, zorder=2)

# Plot
im = events_per_year.plot(
    ax        = ax,
    transform = ccrs.PlateCarree(),
    cmap      = cmap,
    norm      = norm,
    add_colorbar = False,   # draw manually for more control
    zorder    = 1,
)

# Colorbar
cbar = fig.colorbar(
    im, ax=ax,
    boundaries=bounds,
    ticks=bounds,
    spacing='uniform',
    shrink=0.72,
    pad=0.03,
    extend='max',           # indicate values above the top bound exist
)
cbar.set_label('Mean compound events per year', fontsize=11)
cbar.ax.tick_params(labelsize=9)

# Gridlines
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=0.4, color='gray', alpha=0.5, linestyle='--',
)
gl.top_labels   = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(-4, 17, 4))
gl.ylocator = mticker.FixedLocator(range(46, 57, 2))
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}

ax.set_title(
    f'Compound dunkelflaute — mean events per year\n'
    f'{yr0}–{yr1}  |  initial: {INITIAL_PCT}th pct,  severity: {SEVERITY_PCT}th pct  '
    f'|  min duration: {MIN_DURATION} d',
    fontsize=12, fontweight='bold', pad=10,
)

plt.tight_layout()
plt.show()

# Quick domain summary printed alongside the map
print(f"\nMean events/year across domain: {float(events_per_year.mean(skipna=True).values):.2f}")
print(f"Max events/year (single cell) : {float(events_per_year.max(skipna=True).values):.2f}")
print(f"Cells with ≥1 event/year      : "
      f"{int((events_per_year >= 1.0).sum(skipna=True).values)}")
# %%
