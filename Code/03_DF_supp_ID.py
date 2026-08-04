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
# load installed capacity data (IRENA and Hu)
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
# Load daily capacity factors and mask for winter
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
# Build Country Spatial Masks (Onshore + Coastal Offshore Buffer)

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
# visualize masks 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import xarray as xr
import numpy as np

def plot_country_masks(masks_on, masks_off):
    """Plots onshore and offshore xarray masks on a Cartopy map."""
    
    # 1. Get a template array to initialize the combined map
    first_code = list(masks_on.keys())[0]
    template = masks_on[first_code]
    
    # Create empty arrays filled with NaNs (transparent background)
    combined_on = xr.full_like(template, fill_value=np.nan, dtype=float)
    combined_off = xr.full_like(template, fill_value=np.nan, dtype=float)
    
    # 2. Assign a unique integer to each country so they get distinct colors
    country_codes = list(masks_on.keys())
    for i, code in enumerate(country_codes, start=1):
        combined_on = xr.where(masks_on[code], i, combined_on)
        combined_off = xr.where(masks_off[code], i, combined_off)

    # 3. Set up the Map Projection (PlateCarree is standard for lat/lon data)
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add real-world map features under the data
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='darkgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')

    # 4. Plot the Data
    # Use a discrete colormap ('tab20' is good for categorical distinction)
    cmap = plt.get_cmap('tab20', len(country_codes))
    
    # Plot Offshore (semi-transparent)
    combined_off.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0.5, vmax=len(country_codes) + 0.5,
        add_colorbar=False,
        alpha=0.4  # Transparency distinguishes the offshore buffer
    )

    # Plot Onshore (fully opaque)
    combined_on.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0.5, vmax=len(country_codes) + 0.5,
        add_colorbar=False,
        alpha=0.9
    )

    # 5. Create a Custom Legend
    legend_patches = []
    for i, code in enumerate(country_codes):
        # Extract the exact color used for this country from the colormap
        color = cmap(i / max(1, len(country_codes) - 1))
        
        # Add a patch for both the solid onshore and transparent offshore
        legend_patches.append(mpatches.Patch(color=color, label=f'{code} Onshore'))
        legend_patches.append(mpatches.Patch(color=color, alpha=0.4, label=f'{code} Offshore'))

    # Place legend outside the main plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=9, title="Regions", frameon=True)

    # 6. Final Formatting
    ax.set_title("Country Energy Spatial Masks (Onshore & Offshore)", fontsize=15, pad=15)
    
    # Zoom the map strictly to the bounds of our data grid
    ax.set_extent([
        template.longitude.min().item(), template.longitude.max().item(),
        template.latitude.min().item(), template.latitude.max().item()
    ], crs=ccrs.PlateCarree())

    # Add Latitude / Longitude gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.show()

# Execute the visualization using the dictionaries created in your script
plot_country_masks(masks_on, masks_off)
#%%
#TODO
# Detection functions ──────────────────────────────────────────────────────

def merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """Pool runs of True separated by <= max_gap days; drop runs < min_duration.

    Operates on the FULL daily index (with winter masking already applied to
    `flag`), so runs cannot bridge the Mar -> Nov gap: intervening non-winter
    days are False and the ~7-month separation exceeds any sensible max_gap.
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


def detect_events_1d(
        x,
        winter_ok,
        thr_pct=None,
        severe_pct=None,
        max_gap=1,
        min_duration=3,
        absolute_threshold=None
    ):
    """
    Lower-tail event detection on a 1-D daily series.

    Thresholds and sigma come from the within-NDJFM empirical distribution of
    the series itself. Returns the pooled event mask plus the per-series
    threshold, severe threshold and sigma.
    """
    wv = x[winter_ok]

    if np.all(np.isnan(wv)):
        return None

    sigma = np.nanstd(wv)

    if absolute_threshold is not None:

        # Absolute energy scarcity
        thr = thr_sev = absolute_threshold

    else:

        # Relative percentile threshold
        thr = np.nanpercentile(wv, thr_pct)
        thr_sev = np.nanpercentile(wv, severe_pct)

    raw = (x < thr) & winter_ok

    evt = merge_and_filter_runs_1d(
        raw,
        max_gap,
        min_duration
    )

    return evt, thr, thr_sev, sigma

def characterize_events_1d(evt, x, thr, thr_sev, sigma):
    """
    Per-event metrics for a 1-D series with pooled event mask `evt`.

    severity_S      : Otero'22 summed standardized shortfall, signed —
                      pooled gap days that sit above threshold contribute
                      negatively. (Biewald's variant clips at zero; switch
                      np.sum -> np.sum(np.clip(..., 0, None)) if preferred.)
    deficit_cf_days : sum of positive shortfall in CF·days. Capacity-free
                      energy analog: deficit_GWh = deficit_cf_days
                      * IC_total_MW * 24 / 1000 once capacities exist.
    """
    rows = []
    lab, n = label(evt)
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        seg = x[idx[0]: idx[-1] + 1]
        shortfall = np.clip(thr - seg, 0, None)           
        rows.append(dict(
            start_idx       = int(idx[0]),
            end_idx         = int(idx[-1]),
            duration        = int(len(seg)),
            min_cf          = float(np.nanmin(seg)),
            mean_cf         = float(np.nanmean(seg)),
            severity_S = np.sum(shortfall)/sigma,
            deficit_cf_days = float(np.nansum(np.clip(shortfall, 0, None))),
            is_severe       = bool(np.nanmin(seg) < thr_sev),
        ))
    return rows


#%%
# Track A: per-cell detection → spatial climatology ────────────────────────

print("Per-cell detection ...")
cf_np = CF_sys.values                      # (time, lat, lon)

# Relative meteorological hazard
freq_arr      = np.full((n_lat, n_lon), np.nan)
mean_dur_arr  = np.full((n_lat, n_lon), np.nan)
mean_S_arr    = np.full((n_lat, n_lon), np.nan)
tot_S_yr_arr  = np.full((n_lat, n_lon), np.nan)
thr_arr       = np.full((n_lat, n_lon), np.nan)


# Absolute energy scarcity
abs_freq_arr      = np.full((n_lat, n_lon), np.nan)
abs_mean_dur_arr  = np.full((n_lat, n_lon), np.nan)
abs_thr_arr       = np.full((n_lat, n_lon), ABS_CF_THRESHOLD)


for i in range(n_lat):
    for j in range(n_lon):

        # ------------------------------------------------
        # Relative meteorological hazard detection
        # ------------------------------------------------

        res = detect_events_1d(
            cf_np[:, i, j],
            winter_ok,
            thr_pct=THRESHOLD_PCT,
            severe_pct=SEVERE_PCT,
            max_gap=MAX_GAP,
            min_duration=MIN_DURATION
        )
        if res is None:
            continue

        evt, thr, thr_sev, sigma = res

        thr_arr[i,j] = thr

        ev_rows = characterize_events_1d(
            evt,
            cf_np[:, i,j],
            thr,
            thr_sev,
            sigma
        )

        freq_arr[i,j] = len(ev_rows)/n_winters

        if ev_rows:
            Ss = [r['severity_S'] for r in ev_rows]

            mean_dur_arr[i,j] = np.mean([r["duration"] for r in ev_rows])

            mean_S_arr[i,j] = np.mean(Ss)

            tot_S_yr_arr[i,j] = np.sum(Ss) / n_winters



        # ------------------------------------------------
        # Absolute scarcity detection
        # ------------------------------------------------

        abs_res = detect_events_1d(
            cf_np[:,i,j],
            winter_ok,
            max_gap=MAX_GAP,
            min_duration=MIN_DURATION,
            absolute_threshold=ABS_CF_THRESHOLD
        )


        abs_evt, abs_thr, _, abs_sigma = abs_res


        abs_rows = characterize_events_1d(
            abs_evt,
            cf_np[:,i,j],
            abs_thr,
            abs_thr,
            abs_sigma
        )


        abs_freq_arr[i,j] = len(abs_rows)/n_winters

        if abs_rows:
            abs_mean_dur_arr[i,j] = np.mean(
                [r["duration"] for r in abs_rows]
            )
        
coords_2d = {'latitude': CF_sys.latitude, 'longitude': CF_sys.longitude}

cell_stats = xr.Dataset(
    {
        'freq'     : (('latitude', 'longitude'), freq_arr,),
        'mean_dur' : (('latitude', 'longitude'), mean_dur_arr),
        'mean_S'   : (('latitude', 'longitude'), mean_S_arr),
        'tot_S_yr' : (('latitude', 'longitude'), tot_S_yr_arr),
        'threshold': (('latitude', 'longitude'), thr_arr),
                'abs_freq':
            (('latitude','longitude'), abs_freq_arr),

        'abs_mean_dur':
            (('latitude','longitude'), abs_mean_dur_arr),

        'abs_threshold':
            (('latitude','longitude'), abs_thr_arr),
    },
    coords=coords_2d,
    attrs={
        'description'      : 'Per-cell supply-side dunkelflaute climatology (NDJFM)',
        'mix_w_wind'       : W_WIND,
        'mix_w_solar'      : W_SOLAR,
        'threshold_pct'    : THRESHOLD_PCT,
        'severe_pct'       : SEVERE_PCT,
        'max_gap_days'     : MAX_GAP,
        'min_duration_days': MIN_DURATION,
        'n_winters'        : n_winters,
    },
)
cell_stats['freq'].attrs.update(long_name='Mean events per winter')
cell_stats['mean_dur'].attrs.update(long_name='Mean event duration', units='days')
cell_stats['mean_S'].attrs.update(long_name='Mean event severity S (Otero 2022)')
cell_stats['tot_S_yr'].attrs.update(long_name='Total severity per winter')
cell_stats['threshold'].attrs.update(long_name=f'{THRESHOLD_PCT}th pct NDJFM CF_sys')
cell_stats['abs_freq'].attrs.update(
    long_name='Absolute energy scarcity events per winter',
    description=f'CF_sys < {ABS_CF_THRESHOLD}'
)

cell_stats['abs_threshold'].attrs.update(
    long_name='Absolute CF_sys scarcity threshold'
)
cell_stats.to_netcdf(OUT_DIR / 'df_supply_cell_stats.nc')
print(f"Per-cell stats saved → {OUT_DIR / 'df_supply_cell_stats.nc'}")
print(f"  freq     : {np.nanmin(freq_arr):.2f} – {np.nanmax(freq_arr):.2f} events/winter")
print(f"  mean_dur : {np.nanmin(mean_dur_arr):.1f} – {np.nanmax(mean_dur_arr):.1f} days")


#%%
# ─── Map: absolute-scarcity track (with relative track for contrast) ──────────
# Top row: absolute scarcity (CF_sys < ABS_CF_THRESHOLD) — frequency + duration.
# Bottom row: relative frequency, shown only to contrast the self-flattening
# percentile definition against the gradient absolute one.

proj = ccrs.LambertAzimuthalEqualArea(central_longitude=5, central_latitude=52)

panels = [
    (cell_stats['abs_freq'],     'Absolute scarcity — frequency',  'events / winter', 'YlOrRd'),
    (cell_stats['abs_mean_dur'], 'Absolute scarcity — duration',   'days / event',    'YlOrRd'),
    (cell_stats['freq'],         'Relative (10th pct) — frequency','events / winter', 'YlOrRd'),
    (cell_stats['tot_S_yr'],     'Relative — total severity',      'S / winter',      'YlOrRd'),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 13),
                         subplot_kw={'projection': proj})

for ax, (da, ttl, cbar_lbl, cmap) in zip(axes.ravel(), panels):
    ax.set_extent([-14, 18, 42, 62], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#C6E2F5', zorder=0)
    im = da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                          cmap=cmap, levels=12,
                          add_colorbar=False, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
                   linestyle=':', zorder=3)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label(cbar_lbl, fontsize=10)
    ax.set_title(ttl, fontsize=11, fontweight='bold')

fig.suptitle(
    f'Supply-side dunkelflaute — absolute (CF_sys < {ABS_CF_THRESHOLD}) vs '
    f'relative ({THRESHOLD_PCT}th pct)  |  NDJFM, min {MIN_DURATION} d, '
    f'gap ≤ {MAX_GAP} d  (wind {W_WIND:.2f} / solar {W_SOLAR:.2f})',
    fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'df_supply_cell_maps.svg', bbox_inches='tight')
plt.show()


#%%
# ─── Country masks with offshore buffer ───────────────────────────────────────
# Land + coastal buffer; buffered ocean cells are assigned to their NEAREST
# country so North Sea cells are not double-counted. Distance is computed in
# degrees (anisotropic in longitude at these latitudes — acceptable v1;
# upgrade to a projected CRS if precision at the buffer edge ever matters).

def build_country_masks(template, shp_path, buffer_deg):
    gdf = gpd.read_file(str(shp_path))
    lons, lats = np.meshgrid(template.longitude.values, template.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())

    dists, codes = [], []
    for name, code in COUNTRIES.items():
        geom = gdf.loc[gdf['ADMIN'] == name].union_all()
        if code == 'FR':   # exclude Corsica — consistent with ag side / ENTSO-E FR
            geom = geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
        d = shapely.distance(geom, pts).reshape(lats.shape)   # 0 inside polygon
        dists.append(d)
        codes.append(code)

    dist_stack = np.stack(dists)                 # (n_countries, lat, lon)
    nearest    = np.argmin(dist_stack, axis=0)   # shared borders: first match wins

    def _mk(m):
        return xr.DataArray(
            m, coords={'latitude': template.latitude, 'longitude': template.longitude},
            dims=['latitude', 'longitude'])

    masks_on, masks_off = {}, {}
    for k, code in enumerate(codes):
        assigned = (dist_stack[k] <= buffer_deg) & (nearest == k)
        masks_on[code]  = _mk(assigned & (dist_stack[k] == 0.0))
        masks_off[code] = _mk(assigned & (dist_stack[k] >  0.0))
    return masks_on, masks_off


masks_on, masks_off = build_country_masks(CF_sys.isel(time=0), COUNTRIES_SHP, BUFFER_DEG)
print("Cells per country (onshore / offshore):")
for code in COUNTRIES.values():
    print(f"  {code}: {int(masks_on[code].sum())} / {int(masks_off[code].sum())}")



#%%
# ─── Per-country system CF from three capacity-weighted components ─────────────
# CF_sys_country = (w_on*CF_wind_onshore + w_off*CF_wind_offshore + w_sol*CF_solar)
# with per-country IRENA-2024 shares. Offshore wind uses the same V90@100m curve
# as onshore (single-turbine assumption -> understates offshore CF; stated caveat).
# Solar averaged over onshore cells only (offshore PV negligible).

cf_country = {}
for code in COUNTRIES.values():
    cf_on  = CF_wind_dly.where(masks_on[code]).mean(['latitude', 'longitude'])
    cf_off = CF_wind_dly.where(masks_off[code]).mean(['latitude', 'longitude'])
    cf_sol = CF_solar_dly.where(masks_on[code]).mean(['latitude', 'longitude'])

    won, woff, wsol = W.loc[code, ['won', 'woff', 'wsol']]
    cf_sys_c = won * cf_on + woff * cf_off.fillna(0.0) + wsol * cf_sol
    cf_country[code] = cf_sys_c.values

cf_country = pd.DataFrame(cf_country, index=times)

print("\nPer-country CF_sys means (NDJFM):")
print(cf_country[winter_ok].mean().round(3).to_string())


#%%
# ─── Annual aggregates (zero-filled over complete winters, per track) ─────────
# GWh translation is meaningful for the absolute track (shortfall = below a fixed
# fraction of nameplate); for the relative track deficit_cf_days is vs an arbitrary
# percentile, so GWh is left as a relative diagnostic only.

def aggregate_annual(df_track):
    # determine how to aggregate n_severe
    severe_agg = ('is severe', 'sum') if 'is_severe' in df_track.columns else ('duration', 'size')
 
    a = (df_track.groupby(['country', 'winter'])
         .agg(n_events   =('duration',        'size'),
              n_severe   = severe_agg,
              total_dur  =('duration',        'sum'),
              mean_dur   =('duration',        'mean'),
              max_dur    =('duration',        'max'),
              total_S    =('severity_S',      'sum'),
              total_cf   =('deficit_cf_days', 'sum'))
         .reset_index())
    full = pd.MultiIndex.from_product([country_codes, valid_winters],
                                      names=['country', 'winter'])
    a = a.set_index(['country', 'winter']).reindex(full, fill_value=0).reset_index()
    a.loc[a['n_events'] == 0, 'mean_dur'] = np.nan   # undefined, not zero
    a.loc[a['n_events'] == 0, 'max_dur']  = np.nan
    # Energy shortfall vs threshold [GWh] = CF·days × total VRE MW × 24 h / 1000
    a['deficit_GWh'] = a['total_cf'] * a['country'].map(IC_CAPACITY_TOTAL) * 24 / 1000
    return a

annual_rel = aggregate_annual(ev_df[ev_df.track == 'relative'])
annual_abs = aggregate_annual(ev_df[ev_df.track == 'absolute'])
annual_rel['track'] = 'relative'
annual_abs['track'] = 'absolute'
annual = pd.concat([annual_rel, annual_abs], ignore_index=True)

# Provenance sidecar so parameter sweeps don't overwrite silently
run_cfg = dict(cap_vintage=CAP_VINTAGE, threshold_pct=THRESHOLD_PCT,
               severe_pct=SEVERE_PCT, abs_cf=ABS_CF_THRESHOLD,
               max_gap=MAX_GAP, min_duration=MIN_DURATION,
               winter_months=WINTER_MONTHS, buffer_deg=BUFFER_DEG,
               n_winters=n_winters)

# ev_df.to_parquet(OUT_DIR / 'df_supply_events.parquet')
# # annual.to_parquet(OUT_DIR / 'df_supply_annual.parquet')
# pd.Series(run_cfg).to_json(OUT_DIR / 'df_supply_runconfig.json')
# print(f"Saved → {OUT_DIR / 'df_supply_events.parquet'}")
# print(f"Saved → {OUT_DIR / 'df_supply_annual.parquet'}")
# print(f"Saved → {OUT_DIR / 'df_supply_runconfig.json'}")

#%%
# ─── Concurrence: fraction of countries in a shared event day ─────────────────
# Energy-side analog of the CDHW concurrence ratio. Supports the "one driver"
# thesis: continental-scale blocking should co-activate many countries at once.

evt_matrix = pd.DataFrame(event_masks, index=times)[winter_ok]   # winter days × country
n_active   = evt_matrix.sum(axis=1)                              # countries in event per day
concurrence = (n_active[n_active > 0]
               .value_counts().sort_index() / (n_active > 0).sum())

print("\nConcurrence (share of active-event days by # countries co-active):")
print(concurrence.round(3).to_string())
print(f"All-{len(country_codes)}-country co-event days: "
      f"{int((n_active == len(country_codes)).sum())}")


#%%
# ─── Diagnostic 1: annual frequency + severity trends (MK tau) ────────────────
# Two distinct questions, labelled distinctly (as on the CDHW side):
#   frequency (zero-filled)  -> occurrence-rate trend
#   total severity per winter -> intensity trend (the more informative one,
#   since a fixed percentile pins the long-run count)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (metric, ttl) in zip(
        axes, [('n_events', 'frequency (zero-filled)'),
               ('total_S',  'total severity per winter')]):
    for code in country_codes:
        s = annual_rel[annual_rel.country == code].sort_values('winter')
        tau, p = kendalltau(s['winter'], s[metric])
        ax.plot(s['winter'], s[metric], marker='o', ms=3, lw=1.1,
                label=f'{code} (τ={tau:+.2f}, p={p:.2f})')
    ax.set_xlabel('Winter (year of Jan–Mar)')
    ax.set_ylabel(metric)
    ax.set_title(f'Supply-side DF — {ttl}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
# fig.savefig(OUT_DIR / 'df_supply_annual_trends.svg', bbox_inches='tight')
plt.show()


#%%
# ─── Diagnostic 2: monthly distribution of event days ─────────────────────────
# Checks whether the within-NDJFM pooled threshold concentrates events in
# midwinter through the residual solar cycle (motivates a DOY moving-window
# threshold as a sensitivity variant if the pile-up is strong).

month_order = [11, 12, 1, 2, 3]
month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
counts = pd.DataFrame(
    {code: [int(((month == m) & event_masks[code]).sum()) for m in month_order]
     for code in country_codes},
    index=month_names)

fig, ax = plt.subplots(figsize=(10, 5))
counts.plot(kind='bar', ax=ax, width=0.8, edgecolor='k', linewidth=0.3)
ax.set_ylabel('Event days')
ax.set_xlabel('Month')
ax.set_title('Event days by calendar month — within-season threshold diagnostic',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, ncol=4)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
# fig.savefig(OUT_DIR / 'df_supply_monthly_diag.svg', bbox_inches='tight')
plt.show()


# %%
#%%

# ─── Diagnostic: sample country-mean CF_sys time series (2015–2020) ───────────
# Daily CF_sys for NL and DK, with NDJFM relative-event days shaded and the
# absolute scarcity floor drawn in. Panel titles show the IRENA mix shares.

sample_codes = ['NL', 'DK']
t0, t1 = pd.Timestamp('2015-01-01'), pd.Timestamp('2020-12-31')
win = (times >= t0) & (times <= t1)
t_win = times[win]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for ax, code in zip(axes, sample_codes):
    y = cf_country[code].values[win]

    ax.plot(t_win, y, color='#333333', lw=0.7, label='daily CF$_{sys}$')

    # Relative-event days (NDJFM 10th-pct track), shaded
    evt_win = event_masks[code][win]
    ax.fill_between(t_win, 0, y, where=evt_win,
                    color='#d62728', alpha=0.35, linewidth=0,
                    label='relative event day')

    # Relative threshold (per-country, within-NDJFM) and absolute floor
    ax.axhline(thresholds[code], color='#d62728', lw=0.8, ls='--',
               label=f'10th-pct thr = {thresholds[code]:.2f}')
    ax.axhline(ABS_CF_THRESHOLD, color='#1f77b4', lw=0.8, ls=':',
               label=f'abs floor = {ABS_CF_THRESHOLD}')

    # Light winter shading (NDJFM) for orientation
    ax.fill_between(t_win, 0, 1, where=is_ndjfm[win],
                    color='#b0c4de', alpha=0.12, linewidth=0, zorder=0)

    won, woff, wsol = W.loc[code, ['won', 'woff', 'wsol']]
    ax.set_ylim(0, max(0.6, float(np.nanmax(y)) * 1.05))
    ax.set_ylabel('CF$_{sys}$')
    ax.set_title(f'{code}  (onshore {won:.2f} / offshore {woff:.2f} / solar {wsol:.2f})',
                 fontsize=10, fontweight='bold', loc='left')
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=7, ncol=4, loc='upper right')

axes[-1].set_xlabel('Date')
fig.suptitle('Country-mean system capacity factor — daily, 2015–2020\n'
             f'(IRENA {CAP_VINTAGE} mix; blue band = NDJFM)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
# fig.savefig(OUT_DIR / 'df_supply_sample_timeseries.svg', bbox_inches='tight')
plt.show()
# %%
#%%
# ─── Map: onshore / offshore mask footprint colored by per-country frequency ──
# Per-country relative event frequency (events/winter) painted onto that
# country's mask cells. Onshore = solid fill; offshore = same color, hatched,
# so the split is visible while color encodes frequency.

# Per-country relative-track frequency (events per winter)
metric_by_country = (ev_df[ev_df.track == 'relative']
                     .groupby('country')['severity_S'].sum()
                     .reindex(country_codes, fill_value=0.0) / n_winters)
print("Total severity S per winter by country:")
print(metric_by_country.round(2).to_string())


# Paint frequency onto onshore and offshore cell grids separately
lat, lon = CF_sys.latitude.values, CF_sys.longitude.values
onshore_val  = np.full((lat.size, lon.size), np.nan)
offshore_val = np.full((lat.size, lon.size), np.nan)

for code in country_codes:
    v = metric_by_country[code]
    onshore_val [masks_on [code].values] = v
    offshore_val[masks_off[code].values] = v

vmax = float(metric_by_country.max())
vmin = float(metric_by_country.min())

proj = ccrs.LambertAzimuthalEqualArea(central_longitude=5, central_latitude=52)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})
ax.set_extent([-14, 18, 42, 62], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#C6E2F5', zorder=0)
ax.add_feature(cfeature.LAND.with_scale('10m'),  facecolor='#f0f0f0', zorder=0)

# Onshore — solid fill
im = ax.pcolormesh(lon, lat, onshore_val, transform=ccrs.PlateCarree(),
                   cmap='YlOrRd', vmin=vmin, vmax=vmax, shading='nearest', zorder=1)
# Offshore — same color scale, hatched to mark the sea zone
ax.pcolormesh(lon, lat, offshore_val, transform=ccrs.PlateCarree(),
              cmap='YlOrRd', vmin=vmin, vmax=vmax, shading='nearest',
              alpha=0.85, zorder=1, hatch='///')

ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
               linestyle=':', zorder=3)

cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
cbar.set_label('Relative events per winter', fontsize=11)

# Country-code labels at mask centroids (onshore cells)
for code in country_codes:
    m = masks_on[code].values
    if m.any():
        la = lat[np.where(m.any(axis=1))[0]].mean()
        lo = lon[np.where(m.any(axis=0))[0]].mean()
        ax.text(lo, la, code, transform=ccrs.PlateCarree(),
                fontsize=10, fontweight='bold', ha='center', va='center',
                zorder=4, bbox=dict(boxstyle='round,pad=0.15',
                                    fc='white', ec='none', alpha=0.6))

ax.set_title(
    f'Per-country supply-side DF frequency — onshore (solid) vs offshore (hatched)\n'
    f'NDJFM {THRESHOLD_PCT}th pct, min {MIN_DURATION} d, gap ≤ {MAX_GAP} d  '
    f'(IRENA {CAP_VINTAGE} mix)',
    fontsize=11, fontweight='bold', pad=8)

plt.tight_layout()
# fig.savefig(OUT_DIR / 'df_supply_country_freq_map.svg', bbox_inches='tight')
plt.show()
# %%
#%%
# ─── Load maximum installable capacity potentials ─────────────────────────────
# Source: Advances in Applied Energy (2023), doi:10.1016/j.adapen.2023.100134
# Land-eligibility x maximum installable density, per 0.25 deg cell, in MW.
#
# The file's own CF_* columns are static per-cell annual means from the source
# study's siting analysis -- they are DROPPED. All temporal variability comes
# from our ERA5-derived daily CFs; only the potential_* (MW) columns are used.
#
# Grid check: potentials are on an exact 0.25 deg grid, pixel-aligned with ERA5,
# so a reindex onto the CF grid is sufficient -- no interpolation/regridding.

POT_PATH = Path('./../Data/CF_info_grid.csv')

pot_df = pd.read_csv(POT_PATH)[
    ['lon', 'lat', 'potential_onshore', 'potential_offshore', 'potential_PV']
]

def potentials_to_grid(df, template):
    """Pivot the long-format potential table onto the ERA5 CF grid (MW per cell)."""
    ds = (df.set_index(['lat', 'lon'])
            .to_xarray()
            .rename({'lat': 'latitude', 'lon': 'longitude'}))
    # Align to the CF grid; cells absent from the source file have no potential
    ds = ds.reindex(latitude=template.latitude,
                    longitude=template.longitude,
                    fill_value=0.0)
    return ds.rename({'potential_onshore':  'pot_on',
                      'potential_offshore': 'pot_off',
                      'potential_PV':       'pot_pv'})

pot = potentials_to_grid(pot_df, CF_sys.isel(time=0))

print("Installable potential on CF grid [GW]:")
for v, lbl in [('pot_on', 'onshore'), ('pot_off', 'offshore'), ('pot_pv', 'solar PV')]:
    print(f"  {lbl:9s}: {float(pot[v].sum()) / 1e3:8.1f} GW  "
          f"({int((pot[v] > 0).sum()):5d} cells)")


#%%
# ─── Track B1 (headline): potential-weighted country CF_sys ───────────────────
# Per cell, the three sources are weighted by their installable capacity, then
# summed over the country and normalised by that country's total potential:
#
#   CF_sys_country(t) = sum_cells [ CF_wind(t)*pot_on + CF_wind(t)*pot_off
#                                 + CF_solar(t)*pot_pv ]
#                       / sum_cells [ pot_on + pot_off + pot_pv ]
#
# This is dimensionless (a fleet-average capacity factor) and carries the real
# WITHIN-country spatial mix -- no national-average mix constant, no build-out
# magnitude claim. Offshore uses the same wind CF field as onshore (single
# turbine assumption; understates offshore CF -- stated caveat).
#
# Country assignment reuses the buffered masks: a cell counts toward a country
# if it falls in that country's onshore OR offshore mask. Which SOURCE a cell
# contributes is decided by the data (pot_on / pot_off / pot_pv), not by the
# land/sea split -- an improvement over the buffer approximation.

masks_any = {c: (masks_on[c] | masks_off[c]) for c in COUNTRIES.values()}

cf_country_pot = {}
mix_pot        = {}      # per-country potential shares, for reporting
tot_pot        = {}      # per-country total installable MW (supplementary)

for code in COUNTRIES.values():
    m = masks_any[code]
    p_on  = float(pot['pot_on'].where(m).sum())
    p_off = float(pot['pot_off'].where(m).sum())
    p_pv  = float(pot['pot_pv'].where(m).sum())
    p_tot = p_on + p_off + p_pv

    # Capacity-weighted generation summed over the country, then normalised
    gen = ((CF_wind_dly  * pot['pot_on'].where(m, 0.0)).sum(['latitude', 'longitude'])
         + (CF_wind_dly  * pot['pot_off'].where(m, 0.0)).sum(['latitude', 'longitude'])
         + (CF_solar_dly * pot['pot_pv'].where(m, 0.0)).sum(['latitude', 'longitude']))

    cf_country_pot[code] = (gen / p_tot).values
    mix_pot[code] = dict(won=p_on / p_tot, woff=p_off / p_tot, wsol=p_pv / p_tot)
    tot_pot[code] = p_tot

cf_country_pot = pd.DataFrame(cf_country_pot, index=times)
mix_pot        = pd.DataFrame(mix_pot).T
tot_pot        = pd.Series(tot_pot, name='total_potential_MW')

print("\nPotential-based mix shares (onshore / offshore / solar):")
print(mix_pot.round(3).to_string())
print("\nTotal installable potential per country [GW]:")
print((tot_pot / 1e3).round(1).to_string())
print("\nMean NDJFM CF_sys (potential-weighted):")
print(cf_country_pot[winter_ok].mean().round(3).to_string())


#%%
# ─── Track B2 (comparison): IRENA-2024 current-mix country CF_sys ─────────────
# Spatially aggregate CFs first, then combine with the country-level IRENA mix.
# Coarser by construction (one national mix, no within-country texture) -- kept
# as the present-day comparison against the potential-weighted headline.

cf_country_irena = {}
for code in COUNTRIES.values():
    cf_on  = CF_wind_dly.where(masks_on[code]).mean(['latitude', 'longitude'])
    cf_off = CF_wind_dly.where(masks_off[code]).mean(['latitude', 'longitude'])
    cf_sol = CF_solar_dly.where(masks_on[code]).mean(['latitude', 'longitude'])

    won, woff, wsol = W.loc[code, ['won', 'woff', 'wsol']]
    cf_country_irena[code] = (won  * cf_on
                            + woff * cf_off.fillna(0.0)
                            + wsol * cf_sol).values

cf_country_irena = pd.DataFrame(cf_country_irena, index=times)

print("Mean NDJFM CF_sys (IRENA 2024 mix):")
print(cf_country_irena[winter_ok].mean().round(3).to_string())

# Mix comparison: how different are the two portfolios?
mix_cmp = pd.concat([mix_pot.add_suffix('_pot'), W.add_suffix('_irena')], axis=1)
print("\nMix comparison (potential vs IRENA 2024):")
print(mix_cmp[['won_pot', 'won_irena', 'woff_pot', 'woff_irena',
               'wsol_pot', 'wsol_irena']].round(3).to_string())


#%%
# ─── Run detection on both mixes ──────────────────────────────────────────────
# Same grammar for both, so the only difference is the portfolio weighting.

def run_country_detection(cf_df, mix_label):
    """Relative + absolute detection for every country in cf_df."""
    rows, thr_d, evt_d, abs_evt_d = [], {}, {}, {}

    for code in cf_df.columns:
        x = cf_df[code].values

        # Relative (within-NDJFM percentile)
        evt, thr, thr_sev, sigma = detect_events_1d(
            x, winter_ok, thr_pct=THRESHOLD_PCT, severe_pct=SEVERE_PCT,
            max_gap=MAX_GAP, min_duration=MIN_DURATION)
        thr_d[code], evt_d[code] = thr, evt

        for r in characterize_events_1d(evt, x, thr, thr_sev, sigma):
            s_idx, e_idx = r.pop('start_idx'), r.pop('end_idx')
            rows.append(dict(country=code, mix=mix_label, track='relative',
                             start_date=times[s_idx], end_date=times[e_idx],
                             winter=int(winter_id[s_idx]), thr_cf=float(thr), **r))

        # Absolute (fixed CF floor)
        a_evt, a_thr, _, a_sigma = detect_events_1d(
            x, winter_ok, max_gap=MAX_GAP, min_duration=MIN_DURATION,
            absolute_threshold=ABS_CF_THRESHOLD)
        abs_evt_d[code] = a_evt

        for r in characterize_events_1d(a_evt, x, a_thr, a_thr, a_sigma):
            r.pop('is_severe')
            s_idx, e_idx = r.pop('start_idx'), r.pop('end_idx')
            rows.append(dict(country=code, mix=mix_label, track='absolute',
                             start_date=times[s_idx], end_date=times[e_idx],
                             winter=int(winter_id[s_idx]),
                             thr_cf=float(ABS_CF_THRESHOLD), **r))

    return pd.DataFrame(rows), thr_d, evt_d, abs_evt_d


ev_pot,   thr_pot,   evt_pot,   abs_evt_pot   = run_country_detection(cf_country_pot,   'potential')
ev_irena, thr_irena, evt_irena, abs_evt_irena = run_country_detection(cf_country_irena, 'irena2024')

ev_df = pd.concat([ev_pot, ev_irena], ignore_index=True)

print("Events detected:")
print(ev_df.groupby(['mix', 'track']).size().to_string())
print("\nRelative thresholds (potential mix):")
print(", ".join(f"{c}: {v:.3f}" for c, v in thr_pot.items()))
print("Relative thresholds (IRENA mix):")
print(", ".join(f"{c}: {v:.3f}" for c, v in thr_irena.items()))


#%%
# ─── Event-set agreement between the two mixes ────────────────────────────────
# How much does the portfolio weighting change WHICH days are flagged?
# High overlap -> detection is robust to the mix assumption (a useful result).

print("Relative-track event-day overlap (potential vs IRENA):")
for code in COUNTRIES.values():
    a, b = evt_pot[code], evt_irena[code]
    inter, union = (a & b).sum(), (a | b).sum()
    print(f"  {code}: Jaccard = {inter / union:.2f}   "
          f"({int(inter)} shared of {int(union)} union event days)")


#%%
# ─── Map: installable potential and its spatial mix ───────────────────────────
# Headline spatial figure: where the renewable resource can physically go.
# Panels 1-3: absolute MW per cell (supplementary framing).
# Panel 4: solar share of total potential -- the spatial mix that drives the
#          headline detection.

pot_total  = pot['pot_on'] + pot['pot_off'] + pot['pot_pv']
solar_frac = (pot['pot_pv'] / pot_total).where(pot_total > 0)

panels = [
    (pot['pot_on'],  'Onshore wind potential',  'MW / cell',        'Greens'),
    (pot['pot_off'], 'Offshore wind potential', 'MW / cell',        'Blues'),
    (pot['pot_pv'],  'Solar PV potential',      'MW / cell',        'Oranges'),
    (solar_frac,     'Solar share of potential', 'fraction',        'RdYlBu_r'),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 13),
                         subplot_kw={'projection': proj})

for ax, (da, ttl, cbar_lbl, cmap) in zip(axes.ravel(), panels):
    ax.set_extent([-14, 18, 42, 62], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#C6E2F5', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),  facecolor='#f0f0f0', zorder=0)
    im = ax.pcolormesh(da.longitude, da.latitude, da.where(da > 0),
                       transform=ccrs.PlateCarree(), cmap=cmap,
                       shading='nearest', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
                   linestyle=':', zorder=3)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label(cbar_lbl, fontsize=10)
    ax.set_title(ttl, fontsize=11, fontweight='bold')

fig.suptitle('Maximum installable renewable capacity (Adv. Appl. Energy 2023)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'df_supply_potential_maps.svg', bbox_inches='tight')
plt.show()


#%%
# ─── Save supply-side outputs for the residual-load stage ─────────────────────
# cf_country_*.parquet feed 06_DF_RL.py: multiply by capacity (IRENA MW for the
# present-day RL arm) and difference against demand_weather.parquet.

annual_all = []
for mix_label, df_mix in ev_df.groupby('mix'):
    for track, df_t in df_mix.groupby('track'):
        a = (df_t.groupby(['country', 'winter'])
             .agg(n_events =('duration',        'size'),
                  total_dur=('duration',        'sum'),
                  mean_dur =('duration',        'mean'),
                  max_dur  =('duration',        'max'),
                  total_S  =('severity_S',      'sum'),
                  total_cf =('deficit_cf_days', 'sum'))
             .reset_index())
        full = pd.MultiIndex.from_product(
            [list(COUNTRIES.values()), valid_winters], names=['country', 'winter'])
        a = a.set_index(['country', 'winter']).reindex(full, fill_value=0).reset_index()
        a.loc[a['n_events'] == 0, ['mean_dur', 'max_dur']] = np.nan
        a['mix'], a['track'] = mix_label, track
        annual_all.append(a)

annual = pd.concat(annual_all, ignore_index=True)

cf_country_pot.to_parquet(OUT_DIR   / 'cf_country_potential.parquet')
cf_country_irena.to_parquet(OUT_DIR / 'cf_country_irena.parquet')
mix_pot.to_csv(OUT_DIR / 'mix_shares_potential.csv')
tot_pot.to_csv(OUT_DIR / 'total_potential_MW.csv')
ev_df.to_parquet(OUT_DIR  / 'df_supply_events.parquet')
annual.to_parquet(OUT_DIR / 'df_supply_annual.parquet')

run_cfg = dict(potential_source='doi:10.1016/j.adapen.2023.100134',
               cap_vintage=CAP_VINTAGE, threshold_pct=THRESHOLD_PCT,
               severe_pct=SEVERE_PCT, abs_cf=ABS_CF_THRESHOLD,
               max_gap=MAX_GAP, min_duration=MIN_DURATION,
               winter_months=WINTER_MONTHS, buffer_deg=BUFFER_DEG,
               n_winters=n_winters)
pd.Series(run_cfg).to_json(OUT_DIR / 'df_supply_runconfig.json')

print(f"Saved → {OUT_DIR / 'cf_country_potential.parquet'}  (headline)")
print(f"Saved → {OUT_DIR / 'cf_country_irena.parquet'}     (comparison)")
print(f"Saved → {OUT_DIR / 'df_supply_events.parquet'}")
print(f"Saved → {OUT_DIR / 'df_supply_annual.parquet'}")

# %%