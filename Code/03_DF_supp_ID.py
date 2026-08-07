#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_DF_supply_ID.py
==================
Supply-side dunkelflaute detection on combined capacity factors from ERA5 reanalysis

(Otero et al. 2022 "low wind and solar production" events; cf. Raynaud et al.
2018, Meng et al. 2025).

Purely meteorological: events are periods of low *potential* generation.

Three country-level tracks: 
  1. IRENA current-mix relative CF_sys (national capacity share weighting)
  2. Max potential-weighted relative CF_sys (grid-cell MW weighted, dimensionless)
  3. Max potential absolute fleet capacity (grid-cell MW weighted, total MW generation)

@author: afer
"""

#%%
# Imports

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from scipy.ndimage import label
from scipy.stats import kendalltau

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#%%
# Config and params
WINTER_MONTHS = [11, 12, 1, 2, 3]   # extended winter (NDJFM)
THRESHOLD_PCT = 10                  # lower-tail within-season percentilen threshold (%)
SEVERE_PCT    = 5                   # severe tier percentile threshold (%)
MAX_GAP       = 1                   # Otero'22: events separated by < 2 days are pooled
MIN_DURATION  = 3                   # Min event duration filter (days)

# Absolute CF scarcity threshold (system CF floor)
ABS_CF_THRESHOLD = 0.15

# Spatial parameters
CAP_VINTAGE = 2024              # IRENA dataset reference year

# countries for analysis
COUNTRIES = {                       # Natural Earth ADMIN name -> code
    "France"        : "FR",         # (ADMIN, not SOVEREIGNT: keeps Greenland
    "Belgium"       : "BE",         #  out of DK and overseas out of GB/FR)
    "Netherlands"   : "NL",
    "Germany"       : "DE",
    "Denmark"       : "DK",
    "United Kingdom": "GB",         # NOTE: ENTSO-E GB zone excludes Northern
    "Ireland"       : "IE",         # Ireland (NI is in IE_SEM). TODO before RL
}                                   # validation: move NI polygon to IE.

# Input / Output Directories
CF_DAILY_DIR  = Path('./../Results/CF_daily') # ERA5-based CF data
OUT_DIR       = Path('./../Results/DF_supply') 
IC_PATH       = Path('./../Data/IRENA_IC/IC.csv') # IRENA
POT_PATH      = Path('./../Data/Hu_IC/CF_info_grid.csv') #
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
EEZ_SHP = Path('./../Data/EEZ/World_EEZ_v12_20231025/eez_v12.shp')

OUT_DIR.mkdir(parents=True, exist_ok=True)
#%%
# ====================================================
# Load installed capacity data (IRENA and Hu)
# ====================================================
#
# ====================================================
# IRENA (national ratios)
# ====================================================
IC = pd.read_csv(IC_PATH, index_col=0)

IC.index = IC.index.map({          # IRENA country names -> short EU names
    'France': 'FR', 'Belgium': 'BE', 'Netherlands': 'NL', 'Germany': 'DE',
    'Denmark': 'DK', 'United Kingdom': 'GB', 'Ireland': 'IE',
})

# Per-country capacity shares of the three sources
IC_TOTAL_REN = IC[['Onshore Wind (MW)','Offshore Wind (MW)', 'Solar (MW)']].sum(axis = 1)
W_irena = pd.DataFrame({
    'won': IC['Onshore Wind (MW)'] / IC_TOTAL_REN,
    'woff': IC['Offshore Wind (MW)'] / IC_TOTAL_REN,
    'wsol': IC['Solar (MW)'] / IC_TOTAL_REN
})
print("IRENA 2024 National Capacity Shares:")
print(W_irena.round(3))

# ====================================================
# Hu (maximum installable capacity)
# ====================================================

pot_raw = pd.read_csv(POT_PATH)[['lon', 'lat', 'potential_onshore', 'potential_offshore', 'potential_PV']]

ds_pot = pot_raw.set_index(['lat','lon']).to_xarray()
ds_pot = ds_pot.rename({'lat': 'latitude',
    'lon': 'longitude',
    'potential_onshore': 'pot_on',
    'potential_offshore': 'pot_off',
    'potential_PV': 'pot_pv'})
        
#%%
# Load daily capacity factors and winter masks
# ====================================================

# wind daily 
CF_wind_dly  = xr.open_zarr(CF_DAILY_DIR / 'CF_wind_daily.zarr',
                            consolidated=True)['CF_wind_dly']
CF_solar_dly = xr.open_zarr(CF_DAILY_DIR / 'CF_solar_24h_daily.zarr',
                            consolidated=True)['CF_solar_24h_dly']

# align POT grid to ERA5 grid
pot_grid = ds_pot.reindex(
    latitude = CF_wind_dly.latitude,
    longitude = CF_wind_dly.longitude,
    fill_value = 0.0
)

# extract time details
times    = pd.DatetimeIndex(CF_wind_dly.time.values)
months    = times.month.values
years = times.year.values

#define winter labels (nov/dec belong to next year's winter)
winter_id = np.where(months>=11, years + 1, years)
is_ndjfm = np.isin(months, WINTER_MONTHS)

# only complete winters
winters_with_nov = set(winter_id[months == 11])
winters_with_mar = set(winter_id[months == 3])
valid_winters    = sorted(winters_with_nov & winters_with_mar)

winter_ok = is_ndjfm & np.isin(winter_id, valid_winters)
n_winters = len(valid_winters)

print(f"\nERA5 Grid domain loaded: {len(times)} days ({times[0].date()} to {times[-1].date()})")
print(f"Complete winters found: {n_winters} ({valid_winters[0]} -> {valid_winters[-1]})")

#%%
# =================================================================
# Build Country Spatial Masks (Onshore + Coastal Offshore Buffer)
# =================================================================

def build_masks(template_da, land_shp_path, eez_shp_path):
    land_gdf = gpd.read_file(str(land_shp_path))
    eez_gdf = gpd.read_file(str(eez_shp_path))

    lons, lats = np.meshgrid(template_da.longitude.values, template_da.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())

    all_land = land_gdf.union_all() # all known land so it's not mistaken for offshore
    is_any_land = (shapely.distance(all_land, pts) == 0.0).reshape(lats.shape)

    def _to_da(mask_array):
        return xr.DataArray(
            mask_array, 
            coords={'latitude': template_da.latitude, 'longitude': template_da.longitude},
            dims=['latitude', 'longitude']
        )

    masks_on, masks_off = {}, {}
    for name, code in COUNTRIES.items():
        # ONSHORE MASK
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        
        if code == 'FR':  # Exclude Corsica
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
            
        onshore_mask = shapely.within(pts, land_geom).reshape(lats.shape)
        masks_on[code] = _to_da(onshore_mask)

        # OFFSHORE MASK 
        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
        
        # Exclude Rockall 
        if code == 'GB':  
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))
        if code == 'DK':
            # Exclude Greenland
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))
            # Exclude Faroe Islands
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))        
        in_eez = shapely.within(pts, eez_geom).reshape(lats.shape)
        
        # Offshore condition: Must be inside the legal EEZ, but strictly in the water
        masks_off[code] = _to_da(in_eez & ~is_any_land)

    return masks_on, masks_off

masks_on, masks_off = build_masks(CF_wind_dly.isel(time=0), COUNTRIES_SHP, EEZ_SHP)
print("\nCountry mask build complete. Onshore / Offshore cell count:")
for code in COUNTRIES.values():
    print(f"  {code}: {int(masks_on[code].sum())} onshore / {int(masks_off[code].sum())} offshore cells")

#%%
# Create map of offshore and onshore spatial masks

# #visualize masks 
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.patches as mpatches
# import xarray as xr
# import numpy as np

# def plot_country_masks(masks_on, masks_off):
#     """Plots onshore and offshore xarray masks on a Cartopy map."""
    
#     # 1. Get a template array to initialize the combined map
#     first_code = list(masks_on.keys())[0]
#     template = masks_on[first_code]
    
#     # Create empty arrays filled with NaNs (transparent background)
#     combined_on = xr.full_like(template, fill_value=np.nan, dtype=float)
#     combined_off = xr.full_like(template, fill_value=np.nan, dtype=float)
    
#     # 2. Assign a unique integer to each country so they get distinct colors
#     country_codes = list(masks_on.keys())
#     for i, code in enumerate(country_codes, start=1):
#         combined_on = xr.where(masks_on[code], i, combined_on)
#         combined_off = xr.where(masks_off[code], i, combined_off)

#     # 3. Set up the Map Projection (PlateCarree is standard for lat/lon data)
#     fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

#     # Add real-world map features under the data
#     ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
#     ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
#     ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='darkgray')
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')

#     # 4. Plot the Data
#     # Use a discrete colormap ('tab20' is good for categorical distinction)
#     cmap = plt.get_cmap('tab20', len(country_codes))
    
#     # Plot Offshore (semi-transparent)
#     combined_off.plot(
#         ax=ax,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap,
#         vmin=0.5, vmax=len(country_codes) + 0.5,
#         add_colorbar=False,
#         alpha=0.4  # Transparency distinguishes the offshore buffer
#     )

#     # Plot Onshore (fully opaque)
#     combined_on.plot(
#         ax=ax,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap,
#         vmin=0.5, vmax=len(country_codes) + 0.5,
#         add_colorbar=False,
#         alpha=0.9
#     )

#     # 5. Create a Custom Legend
#     legend_patches = []
#     for i, code in enumerate(country_codes):
#         # Extract the exact color used for this country from the colormap
#         color = cmap(i / max(1, len(country_codes) - 1))
        
#         # Add a patch for both the solid onshore and transparent offshore
#         legend_patches.append(mpatches.Patch(color=color, label=f'{code} Onshore'))
#         legend_patches.append(mpatches.Patch(color=color, alpha=0.4, label=f'{code} Offshore'))

#     # Place legend outside the main plot
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
#     ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
#               fontsize=9, title="Regions", frameon=True)

#     # 6. Final Formatting
#     ax.set_title("Country Energy Spatial Masks (Onshore & Offshore)", fontsize=15, pad=15)
    
#     # Zoom the map strictly to the bounds of our data grid
#     ax.set_extent([
#         template.longitude.min().item(), template.longitude.max().item(),
#         template.latitude.min().item(), template.latitude.max().item()
#     ], crs=ccrs.PlateCarree())

#     # Add Latitude / Longitude gridlines
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False

#     plt.show()

# # Execute the visualization using the dictionaries created in your script
# plot_country_masks(masks_on, masks_off)
#%%
# Country-level generation time series

# initialize dictionaries to store results
cf_irena = {}
cf_pot = {}
gen_pot_abs = {}

# Unified land/sea mask for potential weighing
mask_land_sea = {code: (masks_on[code] | masks_off[code]) for code in COUNTRIES.values()}

for code in COUNTRIES.values():
    m_on = masks_on[code]
    m_off = masks_off[code]
    m_any = mask_land_sea[code]

    # TRACK 1: IRENA mix (relative CF)
    cf_on_mean = CF_wind_dly.where(m_on).mean(['latitude','longitude'])
    cf_off_mean = CF_wind_dly.where(m_off).mean(['latitude','longitude'])
    cf_solar_mean = CF_solar_dly.where(m_on).mean(['latitude','longitude'])

    # weights of RE source based on IRENA database
    w_on, w_off, w_solar = W_irena.loc[code, ['won','woff','wsol']]

    # CF_sys based on national capacity ratios
    cf_irena[code] = (w_on * cf_on_mean +
                w_off * cf_off_mean.fillna(0.0) +
                w_solar * cf_solar_mean).values

    # TRACK 2:  MAX installable capacity (Hu) in MW
    
    # mask MW potential per country 
    p_on = pot_grid['pot_on'].where(m_any, 0.0)
    p_off = pot_grid['pot_off'].where(m_any, 0.0)
    p_sol = pot_grid['pot_pv'].where(m_any, 0.0)
    
    # total sum of absolute installable potential MW for country
    p_tot = float((p_on + p_off + p_sol).sum())

    # Calc abs generation potential per gridcell and sum: 
    gen_on = (CF_wind_dly * p_on).sum(['latitude','longitude'])
    gen_off = (CF_wind_dly * p_off).sum(['latitude','longitude'])    
    gen_solar = (CF_solar_dly * p_sol).sum(['latitude','longitude'])

    tot_gen = gen_on + gen_off + gen_solar

    gen_pot_abs[code] = tot_gen.values

    # TRACK 3: RELATIVE MAX installable capacity (Hu) in ratios
        
    if p_tot > 0:
        cf_pot[code] = (tot_gen / p_tot).values
    else:
        cf_pot[code] = np.zeros_like(tot_gen.values)


# Convert arrays to Pandas df
cf_irena_rel_df = pd.DataFrame(cf_irena, index=times)
cf_pot_rel_df   = pd.DataFrame(cf_pot,   index=times)
capgen_pot_df   = pd.DataFrame(gen_pot_abs,    index=times)

print("Time series generated for all three tracks.")

# %%
# Event Detection and characterization functions

def merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """
    Pool consecutive Trues separated by <=max_gap
    Drop merged runs shorter than min_duration
    
    """
    if not np.any(flag):
        return np.zeros_like(flag, dtype=bool)
    
    T, runs, in_run, start = len(flag), [], False, None
    for t in range(T):
        if flag[t] and not in_run:
            start, in_run = t, True
        elif not flag[t] and in_run:
            runs.append([start, t - 1])
            in_run = False
    if in_run:
        runs.append([start, T - 1])

    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = np.zeros_like(flag, dtype=bool)
    for s, e in merged:
        if e - s + 1 >= min_duration:
            out[s:e + 1] = True
            
    return out

def detect_events_1d(x, mask_valid, thr_pct = None, max_gap = 1, min_duration = 3, abs_thr = None):
    """
    Dunkelflaute event detection
    Computes threshold based on masked winter time series
    """
    wv = x[mask_valid]
    if np.all(np.isnan(wv)):
        return None, np.nan, np.nan

    sigma = np.nanstd(wv)
    thr = abs_thr if abs_thr is not None else np.nanpercentile(wv, thr_pct)

    raw_flag = (x < thr) & mask_valid
    evt_mask = merge_and_filter_runs_1d(raw_flag, max_gap, min_duration)

    return evt_mask, thr, sigma

def characterize_events_1d(evt_mask, x, thr, sigma):
    """
    Extract severity metrics for each pooled event.
    Enforces the 'Biewald variant': gap days (where seg > thr) contribute 0 to Severity.
    """
    rows = []
    lab, num_features = label(evt_mask)
    
    for k in range(1, num_features + 1):
        idx = np.where(lab == k)[0]
        seg = x[idx[0] : idx[-1] + 1]
        
        shortfall = np.clip(thr - seg, 0, None)
        
        rows.append(dict(
            start_idx   = int(idx[0]),
            end_idx     = int(idx[-1]),
            duration    = int(len(seg)),
            min_val     = float(np.nanmin(seg)),
            mean_val    = float(np.nanmean(seg)),
            severity_S  = float(np.sum(shortfall) / sigma),
            deficit_sum = float(np.sum(shortfall))
        ))
    return rows

#%% 
# run the event detection and characterization for 3 tracks

def run_detection(df_series, track_name, threshold_type, thr_val, country_capacities=None):
    """Execute detection logic across a DataFrame of countries."""
    all_events = []
    
    for code in df_series.columns:
        x = df_series[code].values
        
        if threshold_type == 'percentile':
            evt_mask, thr, sigma = detect_events_1d(
                x, winter_ok, thr_pct=thr_val, max_gap=MAX_GAP, min_duration=MIN_DURATION
            )
        elif threshold_type == 'absolute':
            # Translate dimensionless CF floor into an absolute MW floor
            floor = (thr_val * country_capacities[code]) if country_capacities else thr_val
            evt_mask, thr, sigma = detect_events_1d(
                x, winter_ok, abs_thr=floor, max_gap=MAX_GAP, min_duration=MIN_DURATION
            )
            
        ev_rows = characterize_events_1d(evt_mask, x, thr, sigma)
        
        for r in ev_rows:
            s_idx = r.pop('start_idx')
            e_idx = r.pop('end_idx')
            
            all_events.append(dict(
                country    = code,
                track      = track_name,
                start_date = times[s_idx],
                end_date   = times[e_idx],
                winter     = int(winter_id[s_idx]),
                threshold  = float(thr),
                **r
            ))
            
    return pd.DataFrame(all_events)

# Pre-calculate total potential capacity (MW) per country for the Track 3 threshold floor
tot_pot_mw = {
    code: float((pot_grid['pot_on'] + pot_grid['pot_off'] + pot_grid['pot_pv'])
                .where(mask_land_sea[code], 0.0).sum()) 
    for code in COUNTRIES.values()
}

print("Running detection logic...")

# Track 1: IRENA relative mix, relative 10th percentile threshold
events_irena_rel = run_detection(
    cf_irena_rel_df, track_name='IRENA_Relative_10pct', 
    threshold_type='percentile', thr_val=THRESHOLD_PCT
)

# Track 2: Max Potential relative mix, relative 10th percentile threshold
events_pot_rel = run_detection(
    cf_pot_rel_df, track_name='MaxPot_Relative_10pct', 
    threshold_type='percentile', thr_val=THRESHOLD_PCT
)

# Track 3: Max Potential absolute mix (MW), absolute 15% capacity floor
events_pot_abs = run_detection(
    capgen_pot_df, track_name='MaxPot_Absolute_15pct_floor', 
    threshold_type='absolute', thr_val=ABS_CF_THRESHOLD, 
    country_capacities=tot_pot_mw
)

# Combine into a single master event catalogue
master_events_df = pd.concat([events_irena_rel, events_pot_rel, events_pot_abs], ignore_index=True)

print(f"\nDetection complete. Master catalog contains {len(master_events_df)} events.")
# %%
#%%
# Annual Aggregations (Zero-filled climatology)

annual_records = []

# Group by track to process each methodology separately
for track_name, df_track in master_events_df.groupby('track'):
    
    # Calculate seasonal statistics
    agg = (df_track.groupby(['country', 'winter'])
           .agg(n_events     = ('duration', 'size'),
                total_dur    = ('duration', 'sum'),
                mean_dur     = ('duration', 'mean'),
                max_dur      = ('duration', 'max'),
                total_S      = ('severity_S', 'sum'),
                total_deficit= ('deficit_sum', 'sum'))
           .reset_index())
    
    # Create a complete index of all countries and all valid winters
    full_idx = pd.MultiIndex.from_product(
        [list(COUNTRIES.values()), valid_winters], 
        names=['country', 'winter']
    )
    
    # Reindex to force zero-filling for winters with no events
    agg = agg.set_index(['country', 'winter']).reindex(full_idx, fill_value=0).reset_index()
    
    # Clean up NaNs (a mean duration of 0 doesn't make sense if n_events is 0)
    agg.loc[agg['n_events'] == 0, ['mean_dur', 'max_dur']] = np.nan
    
    agg['track'] = track_name
    annual_records.append(agg)
    
annual_df = pd.concat(annual_records, ignore_index=True)

print("\nAnnual aggregations complete.")
print(annual_df.groupby('track')[['n_events', 'total_S']].sum().to_string())
# %%
#%%
#%%
# Diagnostic Visualizations (Relative Tracks Only)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define the relative tracks we want to analyze
rel_tracks = ['IRENA_Relative_10pct', 'MaxPot_Relative_10pct']

# ==============================================================================
# Plot 1: Total Severity Comparison by Country (Relative Tracks)
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Filter annual data for only the relative tracks
relative_annual_df = annual_df[annual_df['track'].isin(rel_tracks)]

# Pivot to group by country and track
sev_pivot = relative_annual_df.groupby(['country', 'track'])['total_S'].sum().unstack()

# Plot the comparison
sev_pivot.plot(kind='bar', ax=ax, width=0.7, colormap='viridis', edgecolor='k')
ax.set_title("Total Supply-Side Dunkelflaute Severity (S) over all valid winters", fontsize=14, pad=15)
ax.set_ylabel("Sum of Severity (S)")
ax.set_xlabel("Country")
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(title="Methodology Track", loc='upper left')

plt.tight_layout()
plt.show()

# ==============================================================================
# Plot 2: Time Series Overlay (IRENA vs. Max Potential)
# ==============================================================================

sample_country = 'DE'  # Germany (Good mix of wind/solar for contrast)

# Filter to a specific 2-year window to see event overlap clearly
t_start = pd.Timestamp('2018-10-01')
t_end   = pd.Timestamp('2020-04-01')
time_mask = (times >= t_start) & (times <= t_end)

# Create a 2-panel plot sharing the same X-axis
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

# Define configurations for the loop
plot_configs = [
    {'ax': axes[0], 'track': 'IRENA_Relative_10pct',  'df': cf_irena_rel_df, 'color': '#1f77b4', 'title': 'IRENA Current Mix'},
    {'ax': axes[1], 'track': 'MaxPot_Relative_10pct', 'df': cf_pot_rel_df,   'color': '#ff7f0e', 'title': 'Max Installable Potential'}
]

for cfg in plot_configs:
    ax = cfg['ax']
    track_name = cfg['track']
    cf_series = cfg['df'].loc[time_mask, sample_country]
    
    # Filter events for this specific track and country
    events_subset = master_events_df[
        (master_events_df['country'] == sample_country) & 
        (master_events_df['track'] == track_name) &
        (master_events_df['start_date'] >= t_start) & 
        (master_events_df['start_date'] <= t_end)
    ]
    
    # Get the threshold used
    sample_thr = events_subset['threshold'].iloc[0] if not events_subset.empty else cf_series.quantile(0.10)

    # Plot the CF time series
    ax.plot(cf_series.index, cf_series.values, color='k', lw=1, label=f'CF_sys')
    
    # Draw the specific percentile threshold line
    ax.axhline(sample_thr, color=cfg['color'], linestyle='--', lw=1.5, label=f'10th Pct Threshold ({sample_thr:.3f})')
    
    # Highlight NDJFM winter periods
    for yr in range(t_start.year, t_end.year + 1):
        ax.axvspan(pd.Timestamp(f'{yr}-11-01'), pd.Timestamp(f'{yr+1}-03-31'), 
                   color='gray', alpha=0.1, label='NDJFM Winter' if yr == t_start.year else "")
    
    # Shade the detected events
    for _, row in events_subset.iterrows():
        ax.axvspan(row['start_date'], row['end_date'], color='red', alpha=0.35)

    # Formatting
    ax.set_title(f"{sample_country} - {cfg['title']}", fontsize=12, fontweight='bold', loc='left')
    ax.set_ylabel("System Capacity Factor")
    ax.grid(alpha=0.3)
    
    # Deduplicate legend handles
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

# Final X-axis formatting
axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

fig.suptitle(f"Event Detection Verification: {sample_country} (IRENA vs. Max Potential)", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()
# %%
#%%
# Save Outputs for Residual Load Pipeline

# Save the country-level generation time series
cf_irena_rel_df.to_parquet(OUT_DIR / 'cf_irena_relative.parquet')
cf_pot_rel_df.to_parquet(OUT_DIR / 'cf_pot_relative.parquet')
capgen_pot_df.to_parquet(OUT_DIR / 'capgen_pot_absolute.parquet')

# Save the event catalogues
master_events_df.to_parquet(OUT_DIR / 'df_supply_events.parquet')
annual_df.to_parquet(OUT_DIR / 'df_supply_annual.parquet')

print(f"\nAll data successfully exported to: {OUT_DIR}")
# %%
