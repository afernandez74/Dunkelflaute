#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_DF_supply_ID.py
==================
Supply-side dunkelflaute detection on combined capacity factors from ERA5 reanalysis

(Otero et al. 2022 "low wind and solar production" events; cf. Raynaud et al.
2018, Meng et al. 2025).

Purely meteorological: events are periods of low *potential* generation.

Country-level tracks for generation and fleet CF for five scenarios: 
- IRENA installed capacity in 2017,
- IRENA installed capacity in 2024,
- TYDNP forecasted capacity in 2030,
- TYDNP forecasted capacity in 2040,
- TYNDP forecasted capacity in 2050

Spatial distribution of maximum installable capacities from Hu et al. ()
Used to spatially distribute total country-level capacities in each case. 

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

from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#%%
# Config and params
WINTER_MONTHS = [11, 12, 1, 2, 3]   # extended winter (NDJFM)
THRESHOLD_PCT = 10                  # lower-tail within-season percentilen threshold (%)
GRID_CF_THRESHOLD = 0.05            # grid-cell CF threshold 
SEVERE_PCT    = 5                   # severe tier percentile threshold 
MAX_GAP       = 1                   # Otero'22: events separated by < 2 days are pooled
MIN_DURATION  = 3                   # Min event duration filter (days)

# Years of past and forecast (TYNDP) installed capacity values per country
SCENARIO_YEARS = [2017, 2024, 2030, 2040, 2050]

# countries for analysis
COUNTRIES = {                       # Natural Earth ADMIN name -> code
    "France"        : "FR",         # (ADMIN, not SOVEREIGNT: keeps Greenland
    "Belgium"       : "BE",         #  out of DK and overseas out of UK/FR)
    "Netherlands"   : "NL",
    "Germany"       : "DE",
    "Denmark"       : "DK",
    "United Kingdom": "UK",         # NOTE: ENTSO-E GB zone excludes Northern
    "Ireland"       : "IE",         # Ireland (NI is in IE_SEM). 
}

# Input / Output Directories
CF_DAILY_DIR  = Path('./../Results/CF_daily') # ERA5-based CF data
OUT_DIR       = Path('./../Results/DF_supply/IRENA_TYNDP_scenarios') 
FIG_DIR       = Path('./../Figures')
IC_PATH = Path('./../Data/IRENA_IC/IC_IRENA_TYNDP_comb.csv') #IRENA and TYNDP
POT_PATH      = Path('./../Data/Hu_IC/CF_info_grid.csv') 
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
EEZ_SHP = Path('./../Data/EEZ/World_EEZ_v12_20231025/eez_v12.shp')

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared settings for all maps
PC = ccrs.PlateCarree()
MAP_PROJECTION = ccrs.LambertAzimuthalEqualArea(
    central_longitude=3, central_latitude=53
)
MAP_EXTENT = [-10, 17, 40, 61]
#%%
# Load installed capacity data (IRENA and TYNDP)

# ====================================================
# IRENA and TYNDP data load and cleanup
# ====================================================

IC = pd.read_csv(IC_PATH, index_col=0)

#index country names to abbreviations
IC.index = IC.index.map(COUNTRIES)
IC.index.name = 'country'

# index to year of scenario
IC = (IC.reset_index().rename(columns={'Year':'scenario_year'}))
IC = IC.set_index(['scenario_year', 'country'])

# short column names
tech_short_names = {'Total CAP (MW)': 'total', 'Onshore Wind (MW)':'onshore','Offshore Wind (MW)':'offshore', 'Solar (MW)':'solar'}
IC = IC[['Total CAP (MW)', 'Onshore Wind (MW)','Offshore Wind (MW)', 'Solar (MW)']].rename(columns=tech_short_names)

# calculate proportional shares per country per scenario
IC_shares = IC[['onshore','offshore','solar']].div(IC['total'], axis=0)

# ====================================================
# Hu (maximum installable capacity)
# ====================================================

#load Hu data
pot_raw = pd.read_csv(POT_PATH)[['lon', 'lat', 'potential_onshore', 'potential_offshore', 'potential_PV']]
# xarray dataarray with data
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
         
        if code == 'UK':  #exclude Rockall
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
# PLOT Study region FIG: wind and solar, with linear colour scales

from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

PC = PC
projection = MAP_PROJECTION
circle_scale = 0.01

positions = {
    'FR': (2.3, 47.0), 'BE': (4.5, 50.5), 'NL': (5.3, 52.5),
    'DE': (10.5, 51.0), 'DK': (9.5, 56.5),
    'UK': (-2.5, 54.0), 'IE': (-8.0, 53.3)
}

study_land = xr.zeros_like(masks_land['FR'], dtype=bool)
study_sea = xr.zeros_like(masks_sea['FR'], dtype=bool)

for code in COUNTRIES.values():
    study_land = study_land | masks_land[code]
    study_sea = study_sea | masks_sea[code]

# Fill missing values only in these plotting copies
fields = {}
for var, mask in [('pot_on', study_land),
                  ('pot_off', study_sea),
                  ('pot_pv', study_land)]:
    missing = int((pot_grid[var].isnull() & mask).sum())
    print(f'{var}: {missing} missing cells displayed as zero')
    fields[var] = pot_grid[var].fillna(0).where(mask) / 1000  # GW/cell


# Country totals matching the Hu fields displayed on the maps (GW)
hu_totals = pd.DataFrame({
    var: {
        code: float(field.where(masks_country[code]).sum())
        for code in COUNTRIES.values()
    }
    for var, field in fields.items()
})

dat_IC_plt = pd.DataFrame({
    'solar': IC['solar'],
    'wind': IC[['onshore','offshore']].sum(axis=1)
})

with plt.rc_context({'font.size': 9, 'font.family': 'DejaVu Sans'}):
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 8),
        subplot_kw={'projection': projection}
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90,
                        bottom=0.29, wspace=0.08)

    meshes = {}
    panel_fields = [
        [('pot_on', 'Blues'), ('pot_off', 'Greens')],
        [('pot_pv', 'Oranges')]
    ]

    for i, ax in enumerate(axes):

        ax.set_extent(MAP_EXTENT, crs=PC)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#edf3f7', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#e5e5e5', zorder=0)

        for var, cmap in panel_fields[i]:
            field = fields[var]
            meshes[var] = ax.pcolormesh(
                field.longitude, field.latitude, field.values,
                cmap=cmap,
                norm=Normalize(vmin=0, vmax=max(float(field.max()), 0.001)),
                shading='auto', transform=PC, rasterized=True, zorder=2
            )

        for code in COUNTRIES.values():
            sea = masks_sea[code]
            if bool(sea.any()):
                ax.contour(
                    sea.longitude, sea.latitude, sea.astype(int),
                    levels=[0.5], colors='#456779',
                    linewidths=0.7, linestyles='--',
                    transform=PC, zorder=3
                )

        ax.coastlines(resolution='50m', linewidth=0.55,
                      color='0.25', zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                       linewidth=0.5, edgecolor='0.35', zorder=4)

        technology = ['wind','solar'][i]

        # National IRENA totals for the two panels
        installed = dat_IC_plt[technology].loc[2017]

        # 2050 potentials based on TYNDP
        potential = dat_IC_plt[technology].loc[2050]

        # Concentric circles: Hu potential and IRENA installed capacity
        for code, (lon, lat) in positions.items():
            capacity_i = float(installed.loc[code])
            potential_i = float(potential.loc[code])

            ax.scatter(
                lon, lat, s=capacity_i * circle_scale,
                facecolor='#b05aa9', edgecolor='0.15',
                linewidth=0.7, alpha=0.85,
                transform=PC, zorder=5
            )

            ax.scatter(
                lon, lat, s=potential_i * circle_scale,
                facecolor='none', edgecolor='#702568',
                linewidth=1.3, transform=PC, zorder=6
            )

            ax.annotate(
                code, xy=(lon, lat), xycoords=PC._as_mpl_transform(ax),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold', zorder=7,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')]
            )
        gl = ax.gridlines(
            crs=PC, draw_labels=True, linewidth=0.35,
            color='0.6', alpha=0.5, linestyle=':',
            xlocs=[-10, 0, 10], ylocs=[40, 45, 50, 55, 60]
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (i == 0)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        ax.set_title(
            ['(a)  Onshore and offshore wind', '(b)  Solar PV'][i],
            loc='left', fontsize=11, fontweight='bold', pad=10
        )
        ax.spines['geo'].set_edgecolor('0.55')

    # Separate, explicitly labelled linear colourbars
    bars = [
        ('pot_on',  [0.07, 0.21, 0.38, 0.015], 'Onshore wind'),
        ('pot_off', [0.07, 0.13, 0.38, 0.015], 'Offshore wind'),
        ('pot_pv',  [0.56, 0.21, 0.38, 0.015], 'Solar PV')
    ]
    for var, rect, title in bars:
        cbar = fig.colorbar(
            meshes[var], cax=fig.add_axes(rect), orientation='horizontal'
        )
        cbar.set_label(f'Hu {title}: maximum capacity (GW/grid cell)')

    handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=8,
               markerfacecolor='#b05aa9', markeredgecolor='0.15',
               label=f'IRENA installed ({SCENARIO_YEARS[0]})'),
        Line2D([], [], marker='o', linestyle='none', markersize=10,
               markerfacecolor='none', markeredgecolor='#702568',
               markeredgewidth=1.3, label=f'TYNDP projected {SCENARIO_YEARS[-1]} capacity')
    ]

    # Example sizes using the same scale as the map
    for gw in [10, 100, 500]:
        handles.append(axes[0].scatter(
            [], [], s=gw * 1000 * circle_scale,
            facecolor='none', edgecolor='0.4', label=f'{gw} GW'
        ))

    handles.append(Line2D(
        [], [], color='#456779', linestyle='--', linewidth=0.8,
        label='Offshore mask'
    ))

    fig.legend(
        handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005),
        ncol=3, frameon=False, columnspacing=2, labelspacing=1.2,
        title='National capacity totals; circle area proportional to cap'
    )
    fig.suptitle('Renewable deployment potential and installed capacity',
                 fontsize=14, y=0.97)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(
            FIG_DIR / f'Study_region_IRENA_TYNDP_linear.{ext}',
            dpi=300, bbox_inches='tight', facecolor='white'
        )

    plt.show()
#%% Installed capacities by technology and scenario year
# Installed capacities by technology and scenario year

from matplotlib.lines import Line2D

scenario_years = [2017, 2024, 2030, 2040, 2050]
country_order = ['FR', 'BE', 'NL', 'DE', 'DK', 'UK', 'IE']

technology_settings = {
    'onshore':  ('Onshore wind',  '#3366AA', 'o'),
    'offshore': ('Offshore wind', '#009E73', 's'),
    'solar':    ('Solar PV',      '#E69F00', '^'),
}

# IC has a MultiIndex: scenario_year, country
capacity_data = IC.reset_index().copy()
capacity_data['scenario_year'] = (
    capacity_data['scenario_year'].astype(int)
)

fig, axes = plt.subplots(
    2, 4,
    figsize=(15, 8),
    sharex=True,
    sharey=False,
    layout='constrained'
)

axes = axes.ravel()

for ax, country in zip(axes, country_order):

    country_data = (
        capacity_data.loc[capacity_data['country'].eq(country)]
        .set_index('scenario_year')
        .reindex(scenario_years)
    )

    if country_data[list(technology_settings)].isna().any().any():
        raise ValueError(f'Missing installed capacities for {country}.')

    for technology, (label, colour, marker) in technology_settings.items():

        ax.plot(
            scenario_years,
            country_data[technology] / 1000,
            label=label,
            color=colour,
            marker=marker,
            markersize=6,
            linewidth=2
        )

    ax.set_title(country)
    ax.set_xlabel('Installed-capacity scenario year')
    ax.set_ylabel('Installed capacity (GW)')
    ax.set_xticks(scenario_years)
    ax.tick_params(axis='x', labelbottom=True, rotation=45)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

# Shared legend in the unused eighth panel
axes[-1].axis('off')

handles = [
    Line2D(
        [], [],
        color=colour,
        marker=marker,
        linewidth=2,
        label=label
    )
    for label, colour, marker in technology_settings.values()
]

axes[-1].legend(
    handles=handles,
    title='Technology',
    loc='upper left',
    frameon=False
)

axes[-1].text(
    0, 0.25,
    '2017 and 2024: IRENA\n'
    '2030, 2040 and 2050: TYNDP\n\n'
    'Y-axis scales differ by country.\n'
    'Lines connect capacity snapshots.',
    transform=axes[-1].transAxes,
    fontsize=10
)

fig.suptitle(
    'Installed wind and solar capacities by country',
    fontsize=15
)

# Optional export:
# fig.savefig(
#     FIG_DIR / 'Installed_capacity_scenarios_by_technology.pdf',
#     bbox_inches='tight'
# )

plt.show()
#%%
# Hu-weighted country CFs per technology

# tech names and cf data
tech_input = {
    'onshore':('pot_on',CF_wind_dly),
    'offshore':('pot_off',CF_wind_dly),
    'solar':('pot_pv',CF_solar_dly),
}

# dict to store technology-level CFs
technology_cf = {
    tech: pd.DataFrame(
        index=times,
        columns=list(COUNTRIES.values()),
        dtype=float
    )
    for tech in tech_input.keys()
}
# dict to store Hu-weighted country CFs per technology
hu_potentials = pd.DataFrame(
    index=list(COUNTRIES.values()),
    columns=list(tech_input),
    dtype=float,
)
hu_potentials.index.name = 'country'

# loop through countries and technologies to calculate technology-level CFs
for country in COUNTRIES.values():
    for tech, (pot_var, field) in tech_input.items():

        potential = pot_grid[pot_var].fillna(0.0).where(masks_country[country],0.0)
        potential_total = float(potential.sum())

        hu_potentials.loc[country, tech] = potential_total

        weights = potential / potential_total

        supported_cf = field.where(potential > 0, 0.0)

        country_cf = (supported_cf * weights).sum(["latitude","longitude"], skipna=False).compute()

        technology_cf[tech][country] = country_cf

#%%
# Generation and fleet CF for the five capacity scenarios

GEN_tracks = {}
CF_tracks = {}
track_capacities = {}

capacities = IC.drop(columns=['total'])
total_capacities = IC['total']

for year in SCENARIO_YEARS:
    track = str(year)

    national_capacity = capacities.xs(year, level='scenario_year')
    total_capacity = total_capacities.xs(year, level='scenario_year')

    generation = sum(
        technology_cf[tech].mul(national_capacity[tech], axis='columns')
        for tech in tech_input.keys()
    )

    fleet_cf = generation.div(total_capacity, axis='columns')

    GEN_tracks[track] = generation
    CF_tracks[track] = fleet_cf
    track_capacities[track] = total_capacity

#%%
# Save daily generation and fleet CF for every capacity scenario

OUT_DIR.mkdir(parents=True, exist_ok=True)

for track in GEN_tracks:
    generation_out = GEN_tracks[track].copy()
    generation_out.index.name = 'date'
    generation_out.to_parquet(
        OUT_DIR / f'generation_{track}.parquet'
    )

    cf_out = CF_tracks[track].copy()
    cf_out.index.name = 'date'
    cf_out.to_parquet(
        OUT_DIR / f'fleet_CF_{track}.parquet'
    )

# Save the installed capacities used in these calculations
IC.to_csv(OUT_DIR / 'installed_capacities_by_scenario.csv')

#%%
# FUNCTIONS: Event Detection and characterization 
# =================================

def merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """
    Merge consecutive below-threshold runs separated by at most max_gap days,
    then remove merged events shorter than min_duration days.

    Gap days are included in the final event mask and event duration.
    
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
    Detect below threshold events in a daily time series
    The threshold and std are calculated using valid winter days only
    """

    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype = float)
    mask_valid = np.asarray(mask_valid, dtype = bool)


    # checks
    # Confirm that the series and mask align
    if len(x) != len(mask_valid):
        raise ValueError("x and mask_valid must have the same length")

    # Confirm that there are valid winter days
    if not np.any(mask_valid):
        return np.zeros_like(mask_valid, dtype=bool), np.nan, np.nan

    # Missing values must not be treated as ordinary gap days
    if np.any(mask_valid & ~np.isfinite(x)):
        raise ValueError(
            "Missing or non-finite values found within valid winter days"
        )

    # Extract valid winter values
    wv = x[mask_valid]

    # Calculate the reference-period standard deviation
    sigma = np.std(wv)

    # Calculate percentile or absolute threshold
    if abs_thr is not None:
        thr = float(abs_thr)

    elif thr_pct is not None:
        thr = float(np.percentile(wv, thr_pct))

    else:
        raise ValueError(
            "Either thr_pct or abs_thr must be provided"
        )

    # Flag days strictly below the threshold
    raw_flag = (x < thr) & mask_valid

    # Merge nearby runs and remove short events
    evt_mask = merge_and_filter_runs_1d(
        raw_flag,
        max_gap,
        min_duration
    )

    return evt_mask, thr, sigma

def characterize_events_1d(evt_mask, x, thr, sigma):
    """
    Extract severity metrics for each pooled event.
    Enforces the 'Biewald variant': gap days (where seg > thr) contribute 0 to Severity.
    """
    rows = []

    x = np.asarray(x, dtype=float)
    evt_mask = np.asarray(evt_mask, dtype=bool)

    # Label each continuous event
    lab, num_features = label(evt_mask)

    for k in range(1, num_features + 1):

        # Indices belonging to the event
        idx = np.where(lab == k)[0]

        # Complete event segment, including merged gap days
        seg = x[idx[0]:idx[-1] + 1]

        # Only values below the threshold contribute to deficit
        shortfall = np.clip(thr - seg, 0, None)

        # Number of genuinely low days and merged gap days
        n_low_days = int(np.sum(seg < thr))
        n_gap_days = int(np.sum(seg >= thr))

        # Avoid dividing by zero for constant time series
        if np.isfinite(sigma) and sigma > 0:
            severity = float(np.sum(shortfall) / sigma)
        else:
            severity = np.nan

        rows.append(dict(
            start_idx=int(idx[0]),
            end_idx=int(idx[-1]),
            duration=int(len(seg)),
            n_low_days=n_low_days,
            n_gap_days=n_gap_days,
            min_val=float(np.min(seg)),
            mean_val=float(np.mean(seg)),
            severity_S=severity,
            deficit_sum=float(np.sum(shortfall))
        ))

    return rows


# Wrapper for event detection
def run_detection(df_series, track_name, source_name,
        threshold_type, thr_val,
        country_capacities=None,
        value_units=None):
    """
    Run event detection and characterization for every country.

    For generation data, an absolute CF threshold is converted to MW
    using the capacity supplied for each country.
    """

    all_events = []

    # Use the dataframe's own time index
    series_times = pd.DatetimeIndex(df_series.index)

    # Detection assumes one regularly spaced value per day
    if not series_times.is_monotonic_increasing:
        raise ValueError("Dataframe index must be in chronological order")

    if series_times.has_duplicates:
        raise ValueError("Dataframe index contains duplicate dates")

    if len(series_times) > 1:
        time_steps = series_times.to_series().diff().dropna()

        if not (time_steps == pd.Timedelta(days=1)).all():
            raise ValueError(
                "Event detection requires a continuous daily time series"
            )

    # Recalculate winter labels from this dataframe's index
    series_months = series_times.month.values
    series_years = series_times.year.values

    series_winter_id = np.where(
        series_months >= 11,
        series_years + 1,
        series_years
    )

    series_winter_ok = (
        np.isin(series_months, WINTER_MONTHS)
        & np.isin(series_winter_id, valid_winters)
    )

    # Loop through countries
    for code in COUNTRIES.values():

        if code not in df_series.columns:
            raise ValueError(
                f"{code} is missing from {track_name} {source_name} dataframe"
            )

        x = df_series[code].to_numpy(dtype=float)

        # Percentile-based detection
        if threshold_type == 'percentile':

            evt_mask, thr, sigma = detect_events_1d(
                x,
                series_winter_ok,
                thr_pct=thr_val,
                max_gap=MAX_GAP,
                min_duration=MIN_DURATION
            )

        # Absolute-CF-threshold detection
        elif threshold_type == 'absolute':

            if source_name == 'generation':

                if country_capacities is None:
                    raise ValueError(
                        "Generation detection requires country capacities "
                        "to convert the CF threshold to MW"
                    )

                # Works with either a pandas Series or dictionary
                if isinstance(country_capacities, pd.Series):
                    country_capacity = country_capacities.loc[code]
                else:
                    country_capacity = country_capacities[code]

                # Convert CF threshold to daily mean generation in MW
                absolute_threshold = (
                    float(thr_val) * float(country_capacity)
                )

            else:
                # CF input is already dimensionless
                absolute_threshold = float(thr_val)

            evt_mask, thr, sigma = detect_events_1d(
                x,
                series_winter_ok,
                abs_thr=absolute_threshold,
                max_gap=MAX_GAP,
                min_duration=MIN_DURATION
            )

        else:
            raise ValueError(
                "threshold_type must be 'percentile' or 'absolute'"
            )

        # Characterize every detected event
        event_rows = characterize_events_1d(
            evt_mask,
            x,
            thr,
            sigma
        )

        for row in event_rows:

            start_idx = row.pop('start_idx')
            end_idx = row.pop('end_idx')

            all_events.append(dict(
                country=code,
                track=track_name,
                source=source_name,
                threshold_type=threshold_type,
                threshold_value=float(thr_val),
                units=value_units,
                winter=int(series_winter_id[start_idx]),
                start=series_times[start_idx],
                end=series_times[end_idx],
                threshold=float(thr),
                **row
            ))

    return pd.DataFrame(
        all_events,
        columns=EVENT_COLUMNS
    )

# Build annual supply-side event metrics

def build_annual_supply_metrics(
        events_df, countries, tracks,
        thresholds, winters, source_name='CF'):
    """
    Calculate annual event metrics for every country, track,
    threshold setting and winter.

    Winters without events are retained with zero counts and deficits.
    Mean and maximum event duration remain undefined (NaN).
    """

    # Columns required from events_supply
    required_columns = [
        'country',
        'track',
        'source',
        'threshold_type',
        'threshold_value',
        'winter',
        'duration',
        'n_low_days',
        'severity_S',
        'deficit_sum',
        'deficit_MWh'
    ]
    # Check that all required columns are available
    missing_columns = [
        column for column in required_columns
        if column not in events_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required event columns: {missing_columns}"
        )

    # Container for annual result rows
    annual_rows = []

    # Loop over every possible analysis combination
    for code in countries:
        for track_name in tracks:
            for threshold_type, threshold_values in thresholds.items():
                for threshold_value in threshold_values:
                    for winter in winters:

                        # Select events belonging to this combination
                        selected = events_df[
                            (events_df['country'] == code)
                            & (events_df['track'] == track_name)
                            & (events_df['source'] == source_name)
                            & (
                                events_df['threshold_type']
                                == threshold_type
                            )
                            & np.isclose(
                                events_df['threshold_value'],
                                float(threshold_value)
                            )
                            & (events_df['winter'] == winter)
                        ]

                        # Winter without any detected events
                        if selected.empty:

                            annual_rows.append(dict(
                                country=code,
                                track=track_name,
                                source=source_name,
                                threshold_type=threshold_type,
                                threshold_value=float(threshold_value),
                                winter=int(winter),
                                n_events=0,
                                event_days=0,
                                low_days=0,
                                mean_duration=np.nan,
                                max_duration=np.nan,
                                total_severity_S=0.0,
                                total_deficit_MWh=0.0,
                                max_event_deficit_MWh=np.nan,
                                total_deficit_CF_days=0.0,
                                max_event_deficit_CF_days=np.nan,
                            ))

                        # Winter containing one or more events
                        else:

                            annual_rows.append(dict(
                                country=code,
                                track=track_name,
                                source=source_name,
                                threshold_type=threshold_type,
                                threshold_value=float(threshold_value),
                                winter=int(winter),
                                n_events=int(len(selected)),
                                event_days=int(
                                    selected['duration'].sum()
                                ),
                                low_days=int(
                                    selected['n_low_days'].sum()
                                ),
                                mean_duration=float(
                                    selected['duration'].mean()
                                ),
                                max_duration=int(
                                    selected['duration'].max()
                                ),
                                total_severity_S=float(
                                    selected['severity_S'].sum()
                                ),
                                total_deficit_MWh=float(
                                    selected['deficit_MWh'].sum()
                                ),
                                max_event_deficit_MWh=float(
                                    selected['deficit_MWh'].max()
                                ),
                                total_deficit_CF_days=float(
                                    selected['deficit_sum'].sum()
                                ),
                                max_event_deficit_CF_days=float(
                                    selected['deficit_sum'].max()
                                ),
                            ))

    # Convert result rows into one dataframe
    annual_df = pd.DataFrame(annual_rows)

    # Check that every expected combination was created
    n_threshold_settings = sum(
        len(values) for values in thresholds.values()
    )

    expected_rows = (
        len(countries)
        * len(tracks)
        * n_threshold_settings
        * len(winters)
    )

    assert len(annual_df) == expected_rows, \
        "Annual output does not contain every expected combination"

    return annual_df
#%%
# run the event detection and characterization for 3 tracks
EVENT_COLUMNS = [
    'country',
    'track',
    'source',
    'threshold_type',
    'threshold_value',
    'units',
    'winter',
    'start',
    'end',
    'duration',
    'n_low_days',
    'n_gap_days',
    'threshold',
    'min_val',
    'mean_val',
    'severity_S',
    'deficit_sum'
]

# values for CF thresholds and percentiles for relative events
ABS_CF_THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2]
PERCENTILE_THRESHOLDS = [1, 5, 10]

# Detect and characterize events for all capacity scenarios

thresholds = {
    'absolute': ABS_CF_THRESHOLDS,
    'percentile': PERCENTILE_THRESHOLDS,
}

event_tables = []

for track_name, df in CF_tracks.items():
    for threshold_type, threshold_values in thresholds.items():
        for threshold_value in threshold_values:

            detected = run_detection(
                df,
                track_name=track_name,
                source_name='CF',
                threshold_type=threshold_type,
                thr_val=threshold_value,
                value_units='fraction',
            )

            if not detected.empty:
                event_tables.append(detected)

events_supply = (
    pd.concat(event_tables, ignore_index=True)
    if event_tables
    else pd.DataFrame(columns=EVENT_COLUMNS)
)

#%%
# Add capacity and generation-unit characteristics

events_supply['capacity_MW'] = pd.Series(
    [
        float(track_capacities[track].loc[country])
        for track, country in zip(
            events_supply['track'],
            events_supply['country'],
        )
    ],
    index=events_supply.index,
    dtype=float,
)

for cf_column, mw_column in {
    'threshold': 'threshold_MW',
    'min_val': 'min_generation_MW',
    'mean_val': 'mean_generation_MW',
}.items():
    events_supply[mw_column] = (
        events_supply[cf_column].astype(float)
        * events_supply['capacity_MW']
    )

events_supply['deficit_MWh'] = (
    events_supply['deficit_sum'].astype(float)
    * events_supply['capacity_MW']
    * 24
)

events_supply['scenario_year'] = (
    events_supply['track'].astype(int)
)

events_supply.to_csv(
    OUT_DIR / 'DF_supply_events_all.csv',
    index=False,
)
#%%
# Annual metrics for every country, scenario, threshold and winter

annual_supply = build_annual_supply_metrics(
    events_df=events_supply,
    countries=list(COUNTRIES.values()),
    tracks=list(CF_tracks),
    thresholds=thresholds,
    winters=valid_winters,
    source_name='CF',
)

annual_supply['scenario_year'] = (
    annual_supply['track'].astype(int)
)

annual_supply.to_csv(
    OUT_DIR / 'DF_supply_annual_all.csv',
    index=False,
)

expected_rows = (
    len(COUNTRIES)
    * len(CF_tracks)
    * sum(len(values) for values in thresholds.values())
    * len(valid_winters)
)

print(f'Annual rows: {len(annual_supply)}; expected: {expected_rows}')
print(
    annual_supply.groupby('track')[
        ['n_events', 'event_days', 'total_deficit_MWh']
    ].mean()
)

#%%
# Country × scenario heatmaps: event days and normalized shortfall

import numpy as np
import matplotlib.pyplot as plt

threshold_type = 'absolute'
threshold_value = 0.15

scenario_order = ['2017', '2024', '2030', '2040', '2050']
country_order = ['FR', 'BE', 'NL', 'DE', 'DK', 'UK', 'IE']

plot_data = annual_supply.loc[
    annual_supply['source'].eq('CF')
    & annual_supply['threshold_type'].eq(threshold_type)
    & np.isclose(
        annual_supply['threshold_value'],
        threshold_value
    )
].copy()

if plot_data.empty:
    raise ValueError('No results match the selected threshold.')

plot_data['track'] = plot_data['track'].astype(str)

summary = (
    plot_data
    .groupby(['country', 'track'])[
        ['event_days', 'total_deficit_CF_days']
    ]
    .mean()
)

panel_settings = [
    ('event_days', 'Mean event days per winter', '.1f'),
    (
        'total_deficit_CF_days',
        'Mean CF shortfall per winter (CF-days)',
        '.2f'
    ),
]

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 5),
    layout='constrained'
)

for ax, (metric, title, number_format) in zip(axes, panel_settings):

    table = (
        summary[metric]
        .unstack('track')
        .reindex(index=country_order, columns=scenario_order)
    )

    if table.isna().any().any():
        raise ValueError(f'Missing country–scenario means for {metric}.')

    values = table.to_numpy()
    vmax = max(float(values.max()), 1e-12)

    image = ax.imshow(
        values,
        cmap='YlOrRd',
        vmin=0,
        vmax=vmax,
        aspect='auto'
    )

    for row, column in np.ndindex(values.shape):
        value = values[row, column]

        ax.text(
            column, row,
            format(value, number_format),
            ha='center',
            va='center',
            fontsize=10,
            color='white' if value > 0.65 * vmax else '#222222'
        )

    ax.set(
        xticks=range(len(scenario_order)),
        xticklabels=scenario_order,
        yticks=range(len(country_order)),
        yticklabels=country_order,
        xlabel='Installed-capacity scenario year',
        title=title
    )

    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)

axes[0].set_ylabel('Country')

threshold_label = (
    f'Fleet CF threshold: {threshold_value:.0%}'
    if threshold_type == 'absolute'
    else f'Winter CF percentile: {threshold_value:g}'
)

fig.suptitle(
    f'Supply-drought characteristics | {threshold_label}',
    fontsize=14
)

# Optional export:
# fig.savefig(
#     FIG_DIR / 'DF_supply_scenario_heatmaps.pdf',
#     bbox_inches='tight'
# )

plt.show()

#%%
# Event-duration sensitivity to CF threshold, by country and scenario

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

threshold_values = [0.01, 0.05, 0.10, 0.15, 0.20]
scenario_order = ['2017', '2024', '2030', '2040', '2050']
country_order = ['FR', 'BE', 'NL', 'DE', 'DK', 'GB', 'IE']

scenario_colours = {
    '2017': '#3366AA',
    '2024': '#009E73',
    '2030': '#E69F00',
    '2040': '#D55E00',
    '2050': '#8A508F',
}

scenario_markers = {
    '2017': 'o',
    '2024': 's',
    '2030': '^',
    '2040': 'D',
    '2050': 'P',
}

plot_data = annual_supply.loc[
    annual_supply['source'].eq('CF')
    & annual_supply['threshold_type'].eq('absolute')
].copy()

plot_data['track'] = plot_data['track'].astype(str)

fig, axes = plt.subplots(
    2, 4,
    figsize=(15, 8),
    sharex=True,
    sharey=True,
    layout='constrained'
)

axes = axes.ravel()

for ax, country in zip(axes, country_order):

    for scenario in scenario_order:

        means, lower, upper = [], [], []

        for threshold in threshold_values:

            selected = plot_data.loc[
                plot_data['country'].eq(country)
                & plot_data['track'].eq(scenario)
                & np.isclose(
                    plot_data['threshold_value'],
                    threshold
                ),
                'event_days'
            ]

            if selected.empty:
                raise ValueError(
                    f'Missing annual rows: {country}, '
                    f'{scenario}, threshold {threshold}.'
                )


            # Include every winter, including winters with zero event days.
            means.append(selected.mean())
            lower.append(selected.quantile(0.10))
            upper.append(selected.quantile(0.90))

        means = np.asarray(means, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Thresholds with no events in any winter remain NaN.
        ax.fill_between(
            threshold_values,
            lower,
            upper,
            color=scenario_colours[scenario],
            alpha=0.12,
            linewidth=0
        )

        ax.plot(
            threshold_values,
            means,
            color=scenario_colours[scenario],
            marker=scenario_markers[scenario],
            markersize=5,
            linewidth=2
        )

    ax.set_title(country)
    ax.set_xlabel('Fleet CF threshold')
    ax.set_ylabel('Event days per winter')
    ax.set_xticks(threshold_values)
    ax.xaxis.set_major_formatter(
        PercentFormatter(xmax=1, decimals=0)
    )
    ax.tick_params(axis='x', labelbottom=True)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

# Shared legend in the unused eighth panel
axes[-1].axis('off')

handles = [
    Line2D(
        [], [],
        color=scenario_colours[scenario],
        marker=scenario_markers[scenario],
        linewidth=2,
        label=scenario
    )
    for scenario in scenario_order
]

axes[-1].legend(
    handles=handles,
    title='Capacity scenario',
    loc='upper left',
    frameon=False
)

axes[-1].text(
    0, 0.08,
    'Lines: mean event days per winter\n'
    'Shading: 10th–90th percentiles\n'
    'across all historical winters\n\n'
    'Zero-event winters included;\n'
    'event days include pooled gap days.',
    transform=axes[-1].transAxes,
    fontsize=10
)

fig.suptitle(
    'Mean event days per winter: sensitivity to CF threshold',
    fontsize=15
)
# Optional export:
# fig.savefig(
#     FIG_DIR / 'DF_supply_duration_CF_threshold_sensitivity.pdf',
#     bbox_inches='tight'
# )

plt.show()

#%%
# Probability of at least one event exceeding a duration per winter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# -------------------------------------------------------------
# Settings
# -------------------------------------------------------------
country_order = ['FR', 'BE', 'NL', 'DE', 'DK', 'GB', 'IE']
scenario_order = ['2017', '2024', '2030', '2040', '2050']

cf_thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]
duration_thresholds = np.array([3, 5, 7, 10, 14, 21])

baseline = '2017'
comparison_scenarios = ['2030', '2050']

# Use a subset here for a smaller initial figure, e.g. ['DK', 'FR', 'DE']
plot_countries = country_order

# -------------------------------------------------------------
# Prepare the annual results
# -------------------------------------------------------------
data = annual_supply.loc[
    annual_supply['source'].eq('CF')
    & annual_supply['threshold_type'].eq('absolute')
].copy()

data['track'] = data['track'].astype(str)

# Common historical winters: all scenarios must use the same sample
winter_order = np.sort(data['winter'].unique())
n_winters = len(winter_order)

if n_winters == 0:
    raise ValueError('No winter results were found.')

# -------------------------------------------------------------
# Calculate empirical probabilities
# -------------------------------------------------------------
probability_rows = []

for country in country_order:
    for scenario in scenario_order:
        for threshold in cf_thresholds:

            selected = data.loc[
                data['country'].eq(country)
                & data['track'].eq(scenario)
                & np.isclose(data['threshold_value'], threshold)
            ].copy()

            if selected['winter'].duplicated().any():
                raise ValueError(
                    f'Duplicate winters: {country}, {scenario}, {threshold}'
                )

            selected = (
                selected
                .set_index('winter')
                .reindex(winter_order)
            )

            # Missing annual rows must not become zero-event winters
            if selected['n_events'].isna().any():
                raise ValueError(
                    f'Missing winter results: '
                    f'{country}, {scenario}, {threshold}'
                )

            has_events = selected['n_events'].gt(0)

            if selected.loc[has_events, 'max_duration'].isna().any():
                raise ValueError(
                    f'Missing duration for a winter containing events: '
                    f'{country}, {scenario}, {threshold}'
                )

            # Zero is used only as an indicator that no event occurred.
            # It does not replace undefined mean event durations.
            longest_event = selected['max_duration'].where(
                has_events, 0
            ).to_numpy(dtype=float)

            for duration in duration_thresholds:

                qualifying_winters = int(
                    np.count_nonzero(longest_event >= duration)
                )

                probability_rows.append({
                    'country': country,
                    'track': scenario,
                    'CF_threshold': threshold,
                    'minimum_duration': int(duration),
                    'qualifying_winters': qualifying_winters,
                    'n_winters': n_winters,
                    'probability': qualifying_winters / n_winters,
                })

duration_probabilities = pd.DataFrame(probability_rows)

# -------------------------------------------------------------
# Helper: turn one country/scenario into a plotting matrix
# -------------------------------------------------------------
def probability_matrix(country, scenario):

    return (
        duration_probabilities.loc[
            duration_probabilities['country'].eq(country)
            & duration_probabilities['track'].eq(scenario)
        ]
        .pivot(
            index='CF_threshold',
            columns='minimum_duration',
            values='probability'
        )
        .reindex(
            index=cf_thresholds,
            columns=duration_thresholds
        )
        .to_numpy()
        * 100
    )

baseline_matrices = {
    country: probability_matrix(country, baseline)
    for country in plot_countries
}

change_matrices = {
    (country, scenario):
        probability_matrix(country, scenario)
        - baseline_matrices[country]
    for country in plot_countries
    for scenario in comparison_scenarios
}

# Common scales across all countries
probability_norm = Normalize(vmin=0, vmax=100)

change_limit = max(
    1.0,
    max(np.abs(matrix).max() for matrix in change_matrices.values())
)

change_norm = TwoSlopeNorm(
    vmin=-change_limit,
    vcenter=0,
    vmax=change_limit
)

# -------------------------------------------------------------
# Plot baseline probabilities and scenario changes
# -------------------------------------------------------------
n_columns = 1 + len(comparison_scenarios)

fig, axes = plt.subplots(
    len(plot_countries),
    n_columns,
    figsize=(4.2 * n_columns, 2.35 * len(plot_countries)),
    squeeze=False,
    layout='constrained'
)

for row, country in enumerate(plot_countries):

    matrices = [baseline_matrices[country]] + [
        change_matrices[(country, scenario)]
        for scenario in comparison_scenarios
    ]

    for column, matrix in enumerate(matrices):

        ax = axes[row, column]
        is_baseline = column == 0

        image = ax.imshow(
            matrix,
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            cmap='YlOrRd' if is_baseline else 'RdBu_r',
            norm=probability_norm if is_baseline else change_norm
        )

        if is_baseline:
            probability_image = image
        else:
            change_image = image

        ax.set_xticks(range(len(duration_thresholds)))
        ax.set_xticklabels(duration_thresholds)

        ax.set_yticks(range(len(cf_thresholds)))
        ax.set_yticklabels([f'{q:.0%}' for q in cf_thresholds])

        ax.set_title(country, loc='left', fontsize=11)

        if column == 0:
            ax.set_ylabel('Fleet CF threshold')

        if row == len(plot_countries) - 1:
            ax.set_xlabel('Minimum event duration (days)')

        if row == 0:
            heading = (
                f'{baseline}: probability'
                if is_baseline
                else f'{comparison_scenarios[column - 1]} − {baseline}'
            )
            ax.set_title(heading, loc='center', fontsize=11)

fig.colorbar(
    probability_image,
    ax=list(axes[:, 0]),
    label='Winters with ≥1 qualifying event (%)',
    shrink=0.85,
    pad=0.02
)

fig.colorbar(
    change_image,
    ax=list(axes[:, 1:].ravel()),
    label='Change in probability (percentage points)',
    shrink=0.85,
    pad=0.02
)

fig.suptitle(
    'Winter probability of prolonged renewable-production droughts\n'
    f'{n_winters} historical winters; identical weather across capacity scenarios',
    fontsize=14
)

# Optional exports:
# duration_probabilities.to_csv(
#     OUT_DIR / 'DF_supply_duration_probabilities.csv',
#     index=False
# )
# fig.savefig(
#     FIG_DIR / 'DF_supply_duration_threshold_probabilities.pdf',
#     bbox_inches='tight'
# )

plt.show()
#%%
# Maximum winter event duration: exceedance curves by country

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator

capacity_scenario = '2024'
threshold_values = [0.05, 0.10, 0.15]

country_order = ['FR', 'BE', 'NL', 'DE', 'DK', 'GB', 'IE']

country_colours = {
    'FR': '#0072B2',
    'BE': '#E69F00',
    'NL': '#009E73',
    'DE': '#D55E00',
    'DK': '#CC79A7',
    'GB': '#222222',
    'IE': '#56B4E9',
}

country_styles = {
    'FR': '-', 'BE': '--', 'NL': '-.',
    'DE': '-', 'DK': '--', 'GB': '-', 'IE': ':',
}

data = annual_supply.loc[
    annual_supply['source'].eq('CF')
    & annual_supply['threshold_type'].eq('absolute')
].copy()

data['track'] = data['track'].astype(str)
winter_order = np.sort(np.asarray(valid_winters))
n_winters = len(winter_order)

if n_winters == 0:
    raise ValueError('No valid winters were found.')

# Gather maximum durations, preserving zero-event winters
maximum_durations = {}

for country in country_order:
    for threshold in threshold_values:

        selected = data.loc[
            data['country'].eq(country)
            & data['track'].eq(capacity_scenario)
            & np.isclose(data['threshold_value'], threshold)
        ]

        if selected['winter'].duplicated().any():
            raise ValueError(
                f'Duplicate winters: {country}, {threshold}'
            )

        selected = (
            selected.set_index('winter')
            .reindex(winter_order)
        )

        if selected['n_events'].isna().any():
            raise ValueError(
                f'Missing annual results: {country}, {threshold}'
            )

        has_events = selected['n_events'].gt(0)

        # Zero here means no event occurred in that winter.
        longest = selected['max_duration'].where(
            has_events, 0
        ).to_numpy(dtype=float)

        if not np.isfinite(longest).all():
            raise ValueError(
                f'Missing event duration: {country}, {threshold}'
            )

        maximum_durations[(country, threshold)] = longest

# Start at your minimum retained event duration.
# Extend one day beyond the observed maximum to show the zero endpoint.
largest_event = max(
    values.max() for values in maximum_durations.values()
)

duration_grid = np.arange(
    MIN_DURATION,
    max(MIN_DURATION + 1, int(np.ceil(largest_event)) + 1) + 1
)

fig, axes = plt.subplots(
    1, len(threshold_values),
    figsize=(14, 4.8),
    sharex=True,
    sharey=True
)

axes = np.atleast_1d(axes)

for ax, threshold in zip(axes, threshold_values):

    for country in country_order:

        longest = maximum_durations[(country, threshold)]

        probability = (
            longest[:, None] >= duration_grid[None, :]
        ).mean(axis=0)

        ax.step(
            duration_grid,
            probability,
            where='pre',
            color=country_colours[country],
            linestyle=country_styles[country],
            linewidth=2,
            label=country
        )

    ax.set_title(f'Fleet CF threshold: {threshold:.0%}')
    ax.set_xlabel('Minimum event duration (days)')
    ax.set_xlim(duration_grid[0], duration_grid[-1])
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)

axes[0].set_ylabel(
    'Winters with at least one event\n'
    'of this duration or longer'
)

handles = [
    Line2D(
        [], [],
        color=country_colours[country],
        linestyle=country_styles[country],
        linewidth=2,
        label=country
    )
    for country in country_order
]

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.045),
    ncol=len(country_order),
    frameon=False
)

fig.suptitle(
    f'Winter drought-duration exceedance | {capacity_scenario} capacity scenario',
    fontsize=14
)

fig.text(
    0.5, 0.01,
    f'{n_winters} historical winters; no-event winters included. '
    'Durations include pooled gap days.',
    ha='center',
    fontsize=10
)

fig.tight_layout(rect=[0, 0.15, 1, 0.92])

# Optional export:
# fig.savefig(
#     FIG_DIR / f'DF_duration_exceedance_{capacity_scenario}.pdf',
#     bbox_inches='tight'
# )

plt.show()