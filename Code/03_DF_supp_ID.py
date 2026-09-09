#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_DF_supply_ID.py
==================
Supply-side dunkelflaute detection on combined capacity factors from ERA5 reanalysis

(Otero et al. 2022 "low wind and solar production" events; cf. Raynaud et al.
2018, Meng et al. 2025).

Purely meteorological: events are periods of low *potential* generation.

TWo country-level tracks: 
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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



#%%
# Config and params
WINTER_MONTHS = [11, 12, 1, 2, 3]   # extended winter (NDJFM)
THRESHOLD_PCT = 10                  # lower-tail within-season percentilen threshold (%)
GRID_CF_THRESHOLD = 0.05            # grid-cell CF threshold (%)
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
    "Ireland"       : "IE",         # Ireland (NI is in IE_SEM). 
}

# Input / Output Directories
CF_DAILY_DIR  = Path('./../Results/CF_daily') # ERA5-based CF data
OUT_DIR       = Path('./../Results/DF_supply') 
FIG_DIR       = Path('./../Figures')
IC_PATH       = Path('./../Data/IRENA_IC/IC.csv') # IRENA
POT_PATH      = Path('./../Data/Hu_IC/CF_info_grid.csv') #
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
EEZ_SHP = Path('./../Data/EEZ/World_EEZ_v12_20231025/eez_v12.shp')

OUT_DIR.mkdir(parents=True, exist_ok=True)
#%%
# Load installed capacity data (IRENA and Hu)

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
# save figure function
def save_panel(fig, stem):
    """Save figure as PNG (300 dpi) and SVG."""
    for ext, kw in [('png', {'dpi': 300}), ('svg', {'format': 'svg'})]:
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches='tight',
                    facecolor='white', transparent=False, **kw)
        print(f"  Saved → {path}")

# #%%
# # Plot maps of max. installable capacity

# countries_gdf = gpd.read_file(str(COUNTRIES_SHP))

# LAEA = ccrs.LambertAzimuthalEqualArea(
#     central_longitude=10.0,
#     central_latitude=52.0
# )
# PC = ccrs.PlateCarree()

# # 
# DOMAIN_EXTENT = [
#     float(pot_grid.longitude.values.min()) - 0.5,
#     float(pot_grid.longitude.values.max()) + 0.5,
#     float(pot_grid.latitude.values.min())  - 0.5,
#     float(pot_grid.latitude.values.max())  + 0.5,
# ]

# def format_capacity_map(ax):
#     """Apply the common map style used throughout the project."""

#     # Background
#     ax.add_feature(
#         cfeature.OCEAN,
#         facecolor='#C6E2F5',
#         zorder=0
#     )
#     ax.add_feature(
#         cfeature.LAND,
#         facecolor='0.93',
#         zorder=0
#     )

#     # Country boundaries
#     ax.add_geometries(
#         countries_gdf.geometry,
#         crs=PC,
#         facecolor='none',
#         edgecolor='0.35',
#         linewidth=0.5,
#         zorder=3
#     )

#     # Major political/coastal boundaries
#     ax.add_feature(
#         cfeature.BORDERS,
#         lw=0.9,
#         edgecolor='0.15',
#         zorder=4
#     )
#     ax.add_feature(
#         cfeature.COASTLINE,
#         lw=0.9,
#         zorder=4
#     )

#     # Spatial extent
#     ax.set_extent(DOMAIN_EXTENT, crs=PC)

#     # Gridlines
#     gl = ax.gridlines(
#         crs=PC,
#         draw_labels=True,
#         lw=0.25,
#         color='0.6',
#         alpha=0.5,
#         ls='--'
#     )

#     gl.top_labels = False
#     gl.right_labels = False

#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER

#     gl.xlabel_style = {'size': 7}
#     gl.ylabel_style = {'size': 7}

# lons = pot_grid.longitude.values
# lats = pot_grid.latitude.values

# lon2d, lat2d = np.meshgrid(lons, lats)

# capacity_fields = {
#     'pot_on': {
#         'title': 'Maximum installable onshore wind capacity',
#         'label': 'Maximum installable capacity (GW)',
#         'cmap': 'YlGn',
#     },
#     'pot_off': {
#         'title': 'Maximum installable offshore wind capacity',
#         'label': 'Maximum installable capacity (GW)',
#         'cmap': 'Blues',
#     },
#     'pot_pv': {
#         'title': 'Maximum installable solar PV capacity',
#         'label': 'Maximum installable capacity (GW)',
#         'cmap': 'YlOrBr',
#     },
# }

# # Convert MW -> GW
# capacity_arrays = {name: pot_grid[name].values / 1000.0 for name in capacity_fields}

# # Use a robust upper limit so a small number of very large cells do not
# # dominate the visual range.
# vmax_capacity = {
#     name: np.nanpercentile(arr[arr > 0], 99)
#     if np.any(arr > 0)
#     else 1.0
#     for name, arr in capacity_arrays.items()
# }

# fig, axes = plt.subplots(
#     1, 3,
#     figsize=(16, 5.8),
#     subplot_kw={'projection': LAEA}
# )

# for ax, (var, cfg) in zip(axes, capacity_fields.items()):

#     data = capacity_arrays[var]

#     format_capacity_map(ax)

#     # Filled contours following the previous map style
#     cf = ax.contourf(
#         lon2d,
#         lat2d,
#         data,
#         levels=12,
#         vmin=0,
#         vmax=vmax_capacity[var],
#         cmap=cfg['cmap'],
#         transform=PC,
#         extend='max',
#         zorder=2
#     )

#     # Colorbar
#     cbar = fig.colorbar(
#         cf,
#         ax=ax,
#         orientation='vertical',
#         pad=0.03,
#         shrink=0.82,
#         aspect=22
#     )

#     cbar.set_label(
#         cfg['label'],
#         fontsize=9
#     )

#     cbar.ax.tick_params(labelsize=8)

#     # Panel title
#     ax.set_title(
#         cfg['title'],
#         fontsize=10,
#         pad=8
#     )

# fig.suptitle('Maximum installable renewable energy capacity', fontsize=13, y=1.02)

# plt.tight_layout()
# # save_panel(fig, 'Max_Cap_Hu_fig')
# plt.show()
#%%
# Build Country Spatial Masks (Onshore + Coastal Offshore Buffer)

def build_masks(template_da, land_shp_path, eez_shp_path):
    # Load shapefiles
    land_gdf = gpd.read_file(str(land_shp_path))
    eez_gdf = gpd.read_file(str(eez_shp_path))
    
    lons, lats = np.meshgrid(template_da.longitude.values, template_da.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel()) # create points from grid

    def _to_da(mask_array): # convert numpy array to xarray DataArray
        return xr.DataArray(
            mask_array, 
            coords={'latitude': template_da.latitude, 'longitude': template_da.longitude},
            dims=['latitude', 'longitude']
        )

    all_land = land_gdf.union_all() # all known land so it's not mistaken for offshore
    is_any_land = (shapely.covers(all_land, pts)).reshape(lats.shape) # True if cell is covered by land

    # Initialize containers
    masks_land, masks_sea, masks_country = {}, {}, {}

    # Loop through all countries and build masks for land, sea, and country
    for name, code in COUNTRIES.items():
        # LAND
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        
        if code == 'FR':  #exclude Corsica
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
            
        in_country_land = shapely.covers(land_geom, pts).reshape(lats.shape)

        #EEZ

        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
         
        if code == 'GB':  #exclude Rockall
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))
        if code == 'DK':  #exclude Greenland and Faroe Islands
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))
        
        in_country_eez = shapely.covers(eez_geom, pts).reshape(lats.shape)

        # Separate land and sea cells per country
        country_sea = (in_country_eez & ~is_any_land)

        masks_land[code] = _to_da(in_country_land)
        masks_sea[code] = _to_da(country_sea)
        masks_country[code] = _to_da(in_country_land | country_sea)
    
    return masks_land, masks_sea, masks_country

# Build masks
masks_land, masks_sea, masks_country = build_masks(CF_wind_dly.isel(time=0), COUNTRIES_SHP, EEZ_SHP)

#%%
# # ============================================================
# # Visual check of final country masks
# # ============================================================

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Build one categorical field:
# # 0 = outside analysis domain
# # 1...N = individual countries
# mask_plot = xr.zeros_like(
#     CF_wind_dly.isel(time=0),
#     dtype=float
# )

# for i, code in enumerate(COUNTRIES.values(), start=1):
#     mask_plot = xr.where(
#         masks_country[code],
#         i,
#         mask_plot
#     )

# # Plot
# fig, ax = plt.subplots(
#     figsize=(10, 8),
#     subplot_kw={"projection": ccrs.PlateCarree()}
# )

# p = mask_plot.where(mask_plot > 0).plot(
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     levels=np.arange(0.5, len(COUNTRIES) + 1.5, 1),
#     cmap="tab10",
#     add_colorbar=False
# )

# # Geographic reference
# ax.coastlines(resolution="10m", linewidth=0.8)
# ax.add_feature(
#     cfeature.BORDERS,
#     linewidth=0.7,
#     linestyle="--"
# )

# # Country labels
# for i, code in enumerate(COUNTRIES.values(), start=1):
#     mask = masks_country[code]

#     yy, xx = np.where(mask.values)

#     if len(xx) > 0:
#         lon = float(mask.longitude.values[xx].mean())
#         lat = float(mask.latitude.values[yy].mean())

#         ax.text(
#             lon, lat, code,
#             ha="center",
#             va="center",
#             fontsize=9,
#             fontweight="bold"
#         )

# ax.set_extent([-16, 17, 40, 61], crs=ccrs.PlateCarree())
# ax.set_title("Final country masks")

# plt.tight_layout()
# plt.show()

#%%
# Country-level generation time series

# initialize dictionaries to store results
GEN_irena_avg, GEN_irena_hu, GEN_max_hu = {}, {}, {}

for code in COUNTRIES.values():
    # Masks
    m_on = masks_land[code]
    m_off = masks_sea[code]
    m_coutry = masks_country[code]

    # Installed capacities per country based on IRENA database
    IC_on, IC_off, IC_sol = IC.loc[code, ['Onshore Wind (MW)','Offshore Wind (MW)','Solar (MW)']]

    # Total country installed capacity
    IC_total = IC_on + IC_off + IC_sol

    # TRACK 1: IRENA mix (country-level average CF)
    cf_on_mean = CF_wind_dly.where(m_on).mean(['latitude','longitude'])
    cf_off_mean = CF_wind_dly.where(m_off).mean(['latitude','longitude'])
    cf_solar_mean = CF_solar_dly.where(m_on).mean(['latitude','longitude'])

    # Absolute generation based on national capacity and raw average weather-based CF
    GEN_irena_avg[code] = (IC_on * cf_on_mean +
                IC_off * cf_off_mean.fillna(0.0) +
                IC_sol * cf_solar_mean).values

    # TRACK 2:  Spatially-resolved generation potential (absolute generation) capped by IRENA installed capacities
    
    # mask MW potential per country 
    pot_on = pot_grid['pot_on'].where(m_coutry, 0.0)
    pot_off = pot_grid['pot_off'].where(m_coutry, 0.0)
    pot_sol = pot_grid['pot_pv'].where(m_coutry, 0.0)
    
    # total sum of absolute installable potential MW for country
    p_tot = float((pot_on + pot_off + pot_sol).sum())

    # Calculate shares of generation per grid cell 
    share_on = pot_on / float(pot_on.sum())
    share_off = pot_off / float(pot_off.sum())
    share_sol = pot_sol / float(pot_sol.sum())

    # Distribute IRENA installed capacities to grid via shares
    IC_on_grid = IC_on * share_on
    IC_off_grid = IC_off * share_off 
    IC_sol_grid = IC_sol * share_sol

    # Generation in Track 1: 
    GEN_irena_hu_on = (IC_on_grid * CF_wind_dly).sum(["latitude","longitude"])
    GEN_irena_hu_off = (IC_off_grid * CF_wind_dly).sum(['latitude','longitude'])
    GEN_irena_hu_sol = (IC_sol_grid * CF_solar_dly).sum(["latitude","longitude"])

    # assign to dict
    GEN_irena_hu[code] = (GEN_irena_hu_on + GEN_irena_hu_off + GEN_irena_hu_sol).values

    # TRACK 3:  Spatially-resolved generation potential (absolute generation) hypothetical maximum

    GEN_max_hu_on = (pot_on * CF_wind_dly).sum(["latitude","longitude"])
    GEN_max_hu_off = (pot_off * CF_wind_dly).sum(['latitude','longitude'])
    GEN_max_hu_sol = (pot_sol * CF_solar_dly).sum(["latitude","longitude"])

    # assign to dict
    GEN_max_hu[code] = (GEN_max_hu_on + GEN_max_hu_off + GEN_max_hu_sol).values

times = pd.DatetimeIndex(CF_wind_dly.time.values)

# Assign time series to pandas dataframes
GEN_irena_avg_df = pd.DataFrame(GEN_irena_avg, index = times)
GEN_irena_hu_df = pd.DataFrame(GEN_irena_hu, index = times)
GEN_max_hu_df = pd.DataFrame(GEN_max_hu, index = times)

# %%
# 
# Event Detection and characterization functions 
# =================================

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


#%%
# Run detection for 3 tracks
tracks = {1: GEN_irena_avg_df,
          2: GEN_irena_hu_df,
          3: GEN_max_hu_df}

events = pd.concat([run_detection(df, name, 'percentile', THRESHOLD_PCT)
                    for name, df in tracks.items()], ignore_index=True)

#%% 
# Annual metrics
index = pd.MultiIndex.from_product(
    [COUNTRIES.values(), tracks.keys(), valid_winters],
    names=['country', 'track', 'winter'])

annual = events.groupby(['country', 'track', 'winter']).agg(
    n_events=('duration', 'size'),
    severity=('severity_S', 'sum'),
    duration=('duration', 'mean')
).reindex(index)

annual[['n_events', 'severity']] = annual[['n_events', 'severity']].fillna(0)
annual = annual.reset_index()
# annual.to_csv(OUT_DIR / 'DF_supply_annual.csv', index=False)

#%%
# Fleet capacity factors (CF) and fixed reference winters

# max capacity for each country based on Hu dataset (MW)
cap_max = pd.Series({
    code: float(pot_grid[['pot_on', 'pot_off', 'pot_pv']]
                .to_array().where(masks_country[code], 0).sum())
    for code in COUNTRIES.values()
})

capacities = {1: IC_TOTAL_REN, 2: IC_TOTAL_REN, 3: cap_max}
cf_tracks = {}

for track, df in tracks.items():
    cap = capacities[track].reindex(df.columns)
    cf_tracks[track] = df.div(cap, axis=1)
    # cf_tracks[track].to_csv(OUT_DIR / f'CF_track_{track}.csv')

REF_FIRST, REF_LAST = 1991, 2020  # winter 1991 = Nov 1990 to Mar 1991
reference = winter_ok & (winter_id >= REF_FIRST) & (winter_id <= REF_LAST)

# Check complete daily coverage and finite winter values
assert times.equals(pd.date_range(times[0], times[-1], freq='D'))
for winter in valid_winters:
    expected = pd.date_range(f'{winter-1}-11-01', f'{winter}-03-31')
    assert expected.isin(times).all(), f'Incomplete winter: {winter}'

for df in cf_tracks.values():
    assert np.isfinite(df.loc[winter_ok].values).all(), 'Missing winter CF'
#%%
# Gridcell level analysis 
print("Starting gridcell-level compound spatial hazard analysis...")

GRID_CF_THRESHOLD = 0.10  # 10% fixed efficiency threshold

# 1. Define complete study domain
study_domain = xr.full_like(masks_on['FR'], False, dtype=bool)
for code in COUNTRIES.values():
    study_domain = study_domain | masks_on[code] | masks_off[code]

# 2. Calculate local gridcell technology weights based on Max Potential
p_tot = pot_grid['pot_on'] + pot_grid['pot_off'] + pot_grid['pot_pv']
p_tot_safe = p_tot.where(p_tot > 0, 1.0)

w_on  = pot_grid['pot_on'] / p_tot_safe
w_off = pot_grid['pot_off'] / p_tot_safe
w_sol = pot_grid['pot_pv'] / p_tot_safe

# 3. Compute blended capacity factor array
cf_blend = (w_on * CF_wind_dly) + (w_off * CF_wind_dly) + (w_sol * CF_solar_dly)

# 4. Transpose dimensions to NumPy arrays (time, latitude, longitude)
lats = CF_wind_dly.latitude.values
lons = CF_wind_dly.longitude.values

cf_wind_np  = CF_wind_dly.transpose('time', 'latitude', 'longitude').values
cf_solar_np = CF_solar_dly.transpose('time', 'latitude', 'longitude').values
cf_blend_np = cf_blend.transpose('time', 'latitude', 'longitude').values

domain_mask = (study_domain.values) & (p_tot.values > 0)
valid_i, valid_j = np.where(domain_mask)

# 5. Initialize result grids for ALL metrics and ALL tracks
# Frequency
freq_wind  = np.full((len(lats), len(lons)), np.nan)
freq_solar = np.full((len(lats), len(lons)), np.nan)
freq_blend = np.full((len(lats), len(lons)), np.nan)
freq_comp  = np.full((len(lats), len(lons)), np.nan)

# Duration
dur_wind  = np.full((len(lats), len(lons)), np.nan)
dur_solar = np.full((len(lats), len(lons)), np.nan)
dur_blend = np.full((len(lats), len(lons)), np.nan)
dur_comp  = np.full((len(lats), len(lons)), np.nan)

# Severity (S) - Standardized Shortfall
sev_wind  = np.full((len(lats), len(lons)), np.nan)
sev_solar = np.full((len(lats), len(lons)), np.nan)
sev_blend = np.full((len(lats), len(lons)), np.nan)
sev_comp  = np.full((len(lats), len(lons)), np.nan)

# 6. Spatial Loop
for i, j in zip(valid_i, valid_j):
    ts_w = cf_wind_np[:, i, j]
    ts_s = cf_solar_np[:, i, j]
    ts_b = cf_blend_np[:, i, j]
    
    # Helper to calculate std dev safely for standardized severity[cite: 1]
    sigma_w = np.nanstd(ts_w[winter_ok])
    sigma_s = np.nanstd(ts_s[winter_ok])
    sigma_b = np.nanstd(ts_b[winter_ok])
    
    # --- Track 1: Wind-Only Drought (CF_wind < 0.10) ---
    mask_w, _, _ = detect_events_1d(ts_w, winter_ok, abs_thr=GRID_CF_THRESHOLD, max_gap=MAX_GAP, min_duration=MIN_DURATION)
    if mask_w is not None and np.any(mask_w):
        rows_w = characterize_events_1d(mask_w, ts_w, GRID_CF_THRESHOLD, sigma_w)
        freq_wind[i, j] = len(rows_w) / n_winters
        dur_wind[i, j]  = np.mean([r['duration'] for r in rows_w])
        sev_wind[i, j]  = np.mean([r['severity_S'] for r in rows_w])
    else:
        freq_wind[i, j] = 0; dur_wind[i, j] = 0; sev_wind[i, j] = 0

    # --- Track 2: Solar-Only Drought (CF_solar < 0.10) ---
    mask_s, _, _ = detect_events_1d(ts_s, winter_ok, abs_thr=GRID_CF_THRESHOLD, max_gap=MAX_GAP, min_duration=MIN_DURATION)
    if mask_s is not None and np.any(mask_s):
        rows_s = characterize_events_1d(mask_s, ts_s, GRID_CF_THRESHOLD, sigma_s)
        freq_solar[i, j] = len(rows_s) / n_winters
        dur_solar[i, j]  = np.mean([r['duration'] for r in rows_s])
        sev_solar[i, j]  = np.mean([r['severity_S'] for r in rows_s])
    else:
        freq_solar[i, j] = 0; dur_solar[i, j] = 0; sev_solar[i, j] = 0

    # --- Track 3: Blended CF Drought (CF_blend < 0.10) ---
    mask_b, _, _ = detect_events_1d(ts_b, winter_ok, abs_thr=GRID_CF_THRESHOLD, max_gap=MAX_GAP, min_duration=MIN_DURATION)
    if mask_b is not None and np.any(mask_b):
        rows_b = characterize_events_1d(mask_b, ts_b, GRID_CF_THRESHOLD, sigma_b)
        freq_blend[i, j] = len(rows_b) / n_winters
        dur_blend[i, j]  = np.mean([r['duration'] for r in rows_b])
        sev_blend[i, j]  = np.mean([r['severity_S'] for r in rows_b])
    else:
        freq_blend[i, j] = 0; dur_blend[i, j] = 0; sev_blend[i, j] = 0

    # --- Track 4: Compound Dunkelflaute (Wind < 0.10 AND Solar < 0.10) ---
    raw_compound_flag = (ts_w < GRID_CF_THRESHOLD) & (ts_s < GRID_CF_THRESHOLD) & winter_ok
    mask_c = merge_and_filter_runs_1d(raw_compound_flag, max_gap=MAX_GAP, min_duration=MIN_DURATION)
    
    if np.any(mask_c):
        # Evaluate severity against the Blended CF time series
        rows_c = characterize_events_1d(mask_c, ts_b, GRID_CF_THRESHOLD, sigma_b)
        freq_comp[i, j] = len(rows_c) / n_winters
        dur_comp[i, j]  = np.mean([r['duration'] for r in rows_c])
        sev_comp[i, j]  = np.mean([r['severity_S'] for r in rows_c])
    else:
        freq_comp[i, j] = 0; dur_comp[i, j] = 0; sev_comp[i, j] = 0

# 7. Package into xarray Dataset
ds_compound = xr.Dataset(
    {
        'freq_wind': (['latitude', 'longitude'], freq_wind), 'dur_wind': (['latitude', 'longitude'], dur_wind), 'sev_wind': (['latitude', 'longitude'], sev_wind),
        'freq_solar':(['latitude', 'longitude'], freq_solar),'dur_solar':(['latitude', 'longitude'], dur_solar),'sev_solar':(['latitude', 'longitude'], sev_solar),
        'freq_blend':(['latitude', 'longitude'], freq_blend),'dur_blend':(['latitude', 'longitude'], dur_blend),'sev_blend':(['latitude', 'longitude'], sev_blend),
        'freq_comp': (['latitude', 'longitude'], freq_comp), 'dur_comp': (['latitude', 'longitude'], dur_comp), 'sev_comp': (['latitude', 'longitude'], sev_comp),
    },
    coords={'latitude': lats, 'longitude': lons}
)
print("12-Panel Compound spatial climatology complete.")


#%%
# Plot: 12-Panel Deconstructed Risk (Frequency, Duration, Severity)
# 

# Create a 3x4 grid: 3 Metrics (Rows) x 4 Tracks (Columns)
fig, axes = plt.subplots(3, 4, figsize=(24, 16), subplot_kw={'projection': ccrs.PlateCarree()})

def format_map(ax, title):
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')

# Configuration arrays to loop over the 3x4 grid systematically
tracks = [
    {'prefix': 'wind',  'col_title': "1. Wind-Only\n(CF < 10%)"},
    {'prefix': 'solar', 'col_title': "2. Solar-Only\n(CF < 10%)"},
    {'prefix': 'blend', 'col_title': "3. Blended CF\n(Mix Weighted CF < 10%)"},
    {'prefix': 'comp',  'col_title': "4. Compound Dunkelflaute\n(Wind AND Solar < 10%)"}
]

metrics = [
    {'prefix': 'freq', 'cmap': 'inferno_r', 'label': 'Frequency (Events / Winter)'},
    {'prefix': 'dur',  'cmap': 'magma_r',   'label': 'Mean Duration (Days)'},
    {'prefix': 'sev',  'cmap': 'cividis_r', 'label': 'Mean Severity (S)'}
]

# Generate the 12 panels
for row_idx, metric in enumerate(metrics):
    for col_idx, track in enumerate(tracks):
        
        ax = axes[row_idx, col_idx]
        var_name = f"{metric['prefix']}_{track['prefix']}"
        data = ds_compound[var_name]
        
        # Only add the Column titles to the top row
        title = track['col_title'] if row_idx == 0 else ""
        format_map(ax, title)
        
        # Plot data (letting pcolormesh autoscale to handle the Solar extremes cleanly)
        im = ax.pcolormesh(lons, lats, data, cmap=metric['cmap'], transform=ccrs.PlateCarree(), shading='auto')
        
        # Add colorbars to the bottom of each panel to keep visual spacing clean
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.046)
        cbar.set_label(metric['label'], fontsize=9)
        cbar.ax.tick_params(labelsize=8)

fig.suptitle("Deconstructing Meteorological Supply Risks: Frequency, Duration, and Severity", 
             fontsize=24, fontweight='bold', y=0.97)

# Add Row Labels to the left side of the figure
fig.text(0.10, 0.81, 'FREQUENCY', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.10, 0.50, 'DURATION',  va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.10, 0.22, 'SEVERITY',  va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Adjust layout to make room for row labels and main title
plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.05, hspace=0.2, wspace=0.1)
# plt.tight_layout()
plt.show()

# Save NetCDF results for manuscript preparation
ds_compound.to_netcdf(OUT_DIR / 'gridcell_compound_climatology_12panel.nc')

#%%
# ==============================================================================
# Figure 5: Dunkelflaute Climatology and Resource Footprint
# ==============================================================================
print("Generating Figure 5: Climatology and Resource Footprint...")

# 1. Load EEZ boundaries for the plot
eez_gdf = gpd.read_file(str(EEZ_SHP))
eez_eu = eez_gdf[eez_gdf['SOVEREIGN1'].isin(COUNTRIES.keys())]

# 2. Define Geographical Exclusion Masks (Corsica & NW Scottish Islands)
lon2d, lat2d = np.meshgrid(lons, lats)

# Bounding boxes for exclusion
corsica = (lon2d >= 8.0) & (lon2d <= 10.0) & (lat2d >= 41.0) & (lat2d <= 43.2)
hebrides = (lon2d >= -8.0) & (lon2d <= -6.0) & (lat2d >= 56.5) & (lat2d <= 58.5)
orkney_shetland = (lon2d >= -4.0) & (lon2d <= -0.5) & (lat2d >= 58.5) & (lat2d <= 61.0)

exclusion_mask = corsica | hebrides | orkney_shetland

# 3. Prepare variables and apply exclusion masks
total_wind_cap = (pot_grid['pot_on'] + pot_grid['pot_off']) / 1000.0 
total_solar_cap = pot_grid['pot_pv'] / 1000.0

data_wind_cap = np.where(exclusion_mask, np.nan, total_wind_cap.values)
data_solar_cap = np.where(exclusion_mask, np.nan, total_solar_cap.values)
data_dur = np.where(exclusion_mask, np.nan, ds_compound['dur_blend'].values)
data_sev = np.where(exclusion_mask, np.nan, ds_compound['sev_blend'].values)

# 4. Set up the figure layout
fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw={'projection': PC})
axes = axes.flatten()

# Configuration for iterative plotting
panels = [
    {
        'data': data_wind_cap,
        'title': 'A) Max Installable Wind Capacity',
        'cmap': 'GnBu',
        'label': 'Capacity (GW)',
        'vmax': np.nanpercentile(data_wind_cap[data_wind_cap > 0], 98)
    },
    {
        'data': data_solar_cap,
        'title': 'B) Max Installable Solar PV Capacity',
        'cmap': 'YlOrRd',
        'label': 'Capacity (GW)',
        'vmax': np.nanpercentile(data_solar_cap[data_solar_cap > 0], 98)
    },
    {
        'data': data_dur,
        'title': 'C) Compound Dunkelflaute Mean Duration',
        'cmap': 'magma_r',
        'label': 'Mean Event Length (Days)',
        'vmax': np.nanmax(data_dur)
    },
    {
        'data': data_sev,
        'title': 'D) Compound Dunkelflaute Severity',
        'cmap': 'cividis_r',
        'label': 'Standardized Severity (S)',
        # Cap at 95th percentile to prevent SE France from ruining the scale
        'vmax': np.nanpercentile(data_sev[data_sev > 0], 95) 
    }
]

# 5. Plotting Loop
for ax, cfg in zip(axes, panels):
    # Apply standard project map formatting
    ax.add_feature(cfeature.OCEAN, facecolor='#C6E2F5', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='0.93', zorder=0)
    
    # EEZ Outlines (Light Blue)
    ax.add_geometries(
        eez_eu.geometry, crs=PC, facecolor='none', 
        edgecolor='#87CEEB', linewidth=0.8, alpha=0.9, zorder=2, linestyle='--'
    )
    
    # Country Borders
    ax.add_geometries(countries_gdf.geometry, crs=PC, facecolor='none', edgecolor='0.35', lw=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, lw=0.9, edgecolor='0.15', zorder=4)
    ax.add_feature(cfeature.COASTLINE, lw=0.9, zorder=4)
    ax.set_extent(DOMAIN_EXTENT, crs=PC)
    
    gl = ax.gridlines(draw_labels=True, lw=0.25, color='0.6', alpha=0.5, ls='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    # Plot Data
    im = ax.pcolormesh(
        lons, lats, cfg['data'], 
        transform=PC, cmap=cfg['cmap'], 
        vmin=0, vmax=cfg['vmax'], shading='auto', zorder=1
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, fraction=0.046)
    cbar.set_label(cfg['label'], fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_title(cfg['title'], fontsize=14, fontweight='bold', pad=12)

plt.subplots_adjust(hspace=0.25, wspace=0.1)
# save_panel(fig, 'Fig5_Dunkelflaute_Climatology_Updated')
plt.show()
#%%
# ==============================================================================
# SE France Anomaly Inspector: Time Series Analysis
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

# 1. Define bounding box for SE France / Alpine foothills
lat_min, lat_max = 43.0, 45.5
lon_min, lon_max = 5.5, 7.5

# 2. Extract the gridcell with maximum severity in SE France
se_france_sev = ds_compound["sev_comp"].where(
    (ds_compound.latitude >= lat_min)
    & (ds_compound.latitude <= lat_max)
    & (ds_compound.longitude >= lon_min)
    & (ds_compound.longitude <= lon_max),
    drop=True,
)

max_cell = se_france_sev.where(se_france_sev == se_france_sev.max(), drop=True)
target_lat = float(max_cell.latitude.values[0])
target_lon = float(max_cell.longitude.values[0])

print(
    f"Peak Severity Gridcell in SE France: Lat {target_lat:.2f}°, Lon {target_lon:.2f}°"
)
print(f"Standardized Severity (S): {float(max_cell.values[0]):.2f}")

# 3. Extract capacity factor time series for this location
ts_wind = (
    CF_wind_dly.sel(latitude=target_lat, longitude=target_lon, method="nearest")
    .compute()
    .values
)
ts_solar = (
    CF_solar_dly.sel(latitude=target_lat, longitude=target_lon, method="nearest")
    .compute()
    .values
)

# Technology weights based on max potential at this cell
p_on = float(
    pot_grid["pot_on"].sel(
        latitude=target_lat, longitude=target_lon, method="nearest"
    )
)
p_off = float(
    pot_grid["pot_off"].sel(
        latitude=target_lat, longitude=target_lon, method="nearest"
    )
)
p_pv = float(
    pot_grid["pot_pv"].sel(
        latitude=target_lat, longitude=target_lon, method="nearest"
    )
)
p_tot = p_on + p_off + p_pv

w_wind = (p_on + p_off) / p_tot if p_tot > 0 else 0.5
w_solar = p_pv / p_tot if p_tot > 0 else 0.5

ts_blend = (w_wind * ts_wind) + (w_solar * ts_solar)

# 4. Detect compound events for this specific cell
raw_compound_flag = (
    (ts_wind < GRID_CF_THRESHOLD) & (ts_solar < GRID_CF_THRESHOLD) & winter_ok
)
evt_mask = merge_and_filter_runs_1d(
    raw_compound_flag, max_gap=MAX_GAP, min_duration=MIN_DURATION
)

# 5. Plot a sample winter season (e.g., Nov 2017 to Mar 2018)
sample_window = (times >= "2016-11-01") & (times <= "2018-03-31")
sample_times = times[sample_window]

fig, ax = plt.subplots(figsize=(14, 5))

# Plot Generation Time Series
ax.plot(
    sample_times,
    ts_wind[sample_window],
    label=f"Wind CF (weight={w_wind:.2f})",
    color="#1f77b4",
    lw=1.2,
    alpha=0.7,
)
ax.plot(
    sample_times,
    ts_solar[sample_window],
    label=f"Solar CF (weight={w_solar:.2f})",
    color="#ff7f0e",
    lw=1.2,
    alpha=0.7,
)
ax.plot(
    sample_times,
    ts_blend[sample_window],
    label="Blended CF",
    color="black",
    lw=2,
)

# Threshold Floor Line
ax.axhline(
    GRID_CF_THRESHOLD,
    color="crimson",
    linestyle="--",
    lw=1.5,
    label=f"Threshold ({GRID_CF_THRESHOLD*100:.0f}%)",
)

# Highlight active Dunkelflaute event days
sample_evt = evt_mask[sample_window]
lab, num_features = label(sample_evt)
for k in range(1, num_features + 1):
    idx = np.where(lab == k)[0]
    ax.axvspan(
        sample_times[idx[0]],
        sample_times[idx[-1]],
        color="crimson",
        alpha=0.25,
        label="Dunkelflaute Event" if k == 1 else "",
    )

ax.set_ylabel("Daily Capacity Factor", fontsize=11)
ax.set_title(
    f"Dunkelflaute Dynamics at SE France Gridcell (Lat {target_lat:.2f}°, Lon {target_lon:.2f}°) — Winter 2017/18",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_ylim(0, max(0.6, np.nanmax(ts_blend[sample_window]) * 1.1))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.show()
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
