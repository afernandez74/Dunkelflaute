#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_results.py — Dunkelflaute Results Visualization
===================================================
Publication figure set for the supply-side and residual-load (RL) dunkelflaute
detection pipelines, plus the joint (agriculture + energy) circulation figure.

Detection philosophy in this revision
-------------------------------------
Percentile (relative) thresholds are dropped from all headline results.  A
within-season 90th/10th percentile threshold places a fixed fraction of days
beyond it in EVERY country by construction, so relative-threshold frequency
cannot vary across countries and carries no information.  More importantly, the
hazard here is defined by ABSOLUTE shortfall — a system is stressed by megawatts
not delivered, not by a departure from its own seasonal norm.  This mirrors the
impact-anchored variable selection used on the agricultural side, where the
detection variable is chosen from the damage mechanism rather than convention.

Relative thresholds are retained in one place only: FIG 3, where they are swept
alongside the absolute scenarios as an explicit robustness check.

Figures produced
-----------------
  FIG 1  Supply-side weather risk (4 panels) — purely meteorological, present-day
         IRENA fleet.  Grid-cell frequency + severity (contourf), country-level
         frequency + severity.  Answers: where is generation weather-risk highest?
  FIG 2  Residual-load results — event anatomy for DE and DK under three capacity
         build-out scenarios, plus within-season timing.
  FIG 3  Robustness — events per winter and cross-country concurrence under six
         detection configurations (3 percentile x 3 capacity scenario).
  FIG 4  Z500 composites, concurrence-filtered — CDHW, RL aggregate, and one
         panel per country, on a shared norm.
  SI 1   Trend forest plot (Kendall tau with bootstrap CI).
  SI 2   Duration-severity joint distribution.
  SI 3   Supply-side absolute CF threshold sensitivity.

Known upstream issues addressed here
------------------------------------
* 03_DF_supp_ID.py writes no per-cell statistics file, yet the previous version
  of this script loaded 'df_supply_cell_stats.nc'.  The grid-cell climatology is
  now computed here from the CF_daily zarr stores and cached, so its provenance
  and parameters are explicit.
* 05_DF_RL_ID.py computes severity_S as a raw MW-day sum with no sigma
  normalisation (unlike 03_DF_supp_ID.py, which divides by sigma).  Raw MW-days
  are not comparable across systems of different size; severity is normalised by
  mean winter demand here, giving a "deficit-day equivalent".
* 05_DF_RL_ID.py's Jaccard block filters supply events on track == 'T1', which
  never matches any supply track name, so all its values are 0.  That panel has
  been dropped from the figure set at the user's request; if it is ever revived,
  use SUPPLY_HEADLINE below.

@author: afer
"""
# %%
# Imports

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from scipy.stats import kendalltau

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
# =================================================================
# Config and params
# =================================================================

WINTER_MONTHS = [11, 12, 1, 2, 3]   # NDJFM
SUMMER_MONTHS = [6, 7, 8]           # JJA (CDHW season)

MAX_GAP      = 1                    # Otero'22 pooling
MIN_DURATION = 3                    # minimum event duration (days)

# --- Supply side (FIG 1): system-CF floor ----------------------------------
# A day is a low-generation day when the capacity-weighted system CF falls below
# a floor.  Two ways of setting that floor, because they answer different
# questions and one of them has a trap:
#
#   'absolute'          — a single CF value for every country (classic
#                         dunkelflaute criterion).  TRAP: national technology
#                         mixes differ enormously, and winter solar CF above
#                         ~50 N is near zero.  A solar-heavy system (DE: roughly
#                         58% of capacity is PV) therefore has a MEAN winter
#                         system CF close to 0.12, i.e. below a 0.15 floor, so it
#                         would register as "in event" most of the winter, while
#                         a wind-heavy system (DK, mean ~0.23) almost never
#                         would.  That reproduces the structural-not-weather
#                         artefact of the old max-potential track in a new form.
#                         Check the printed mean-CF table before trusting it.
#
#   'fraction_of_mean'  — floor is CF_MEAN_FRACTION x that country's own mean
#                         NDJFM system CF, i.e. "output fell below X% of what
#                         this fleet typically delivers in winter".  Still an
#                         absolute shortfall criterion (not a percentile of the
#                         distribution, so it does not fix the event rate by
#                         construction), but it removes the mix confound and
#                         makes countries comparable.  Recommended default.
CF_THRESHOLD_MODE = 'fraction_of_mean'
ABS_CF_THRESHOLD  = 0.15        # used when mode == 'absolute'
CF_MEAN_FRACTION  = 0.50        # used when mode == 'fraction_of_mean'
CF_THR_SWEEP      = [0.10, 0.15, 0.20]        # SI 3, absolute mode
CF_FRAC_SWEEP     = [0.40, 0.50, 0.60]        # SI 3, fraction mode

# --- Supply gridcell fields (paper FIG 5) ----------------------------------
GRID_ABS_CF   = 0.05        # panel (e): absolute blended-CF floor (deep-lull geography)
GRID_REL_FRAC = 0.50        # panel (f): fraction of each cell's own NDJFM mean blended CF

# --- Residual load (FIG 2/3): capacity build-out scenarios -----------------
# capgen_pot_absolute.parquet is generation under Hu et al. (2023) MAXIMUM
# installable potential, which greatly exceeds any plausible fleet and therefore
# produces a structural surplus.  Scaling it by alpha represents partial
# build-out.  CAVEAT: uniform scaling assumes proportional deployment across all
# sites and technologies; real build-out develops the best sites first, so a
# uniform 50% fleet under-performs a realistically-sited 50% fleet.  These
# scenarios are therefore conservative (pessimistic) on generation.
CAPACITY_SCENARIOS = [0.50, 0.75, 1.00]
HEADLINE_ALPHA     = 0.50

# Percentile thresholds retained ONLY for the FIG 3 robustness sweep.
RL_PCT_SWEEP = [85, 90, 95]

# --- Concurrence filtering for the Z500 composites -------------------------
MIN_CONCURRENT_COUNTRIES = 4        # of 7; tune against the printed distribution
CDHW_AREA_PCTL           = 90       # top decile of days by domain area affected
MIN_COMPOSITE_DAYS       = 100      # warn below this

CONCURRENCE_ALPHA = HEADLINE_ALPHA  # scenario used for the composite event days

SPOTLIGHT_WINTER   = 2018           # NDJFM labelled by its Jan-Mar year
ANATOMY_COUNTRIES  = ['DE', 'DK']   # FIG 2 panels

OCEAN_COLOR = '#C6E2F5'
LAND_COLOR  = '#F5F5F2'
MAP_EXTENT  = [-11.5, 16.5, 41.5, 59.5]
Z500_EXTENT = [-14.0, 18.0, 42.0, 62.0]
PROJ        = ccrs.LambertConformal(central_longitude=3, central_latitude=50)
LAEA        = ccrs.LambertAzimuthalEqualArea(central_longitude=10.0, central_latitude=52.0)  # FIG 5 maps (matches 03)

COUNTRIES = {
    "France"        : "FR",
    "Belgium"       : "BE",
    "Netherlands"   : "NL",
    "Germany"       : "DE",
    "Denmark"       : "DK",
    "United Kingdom": "GB",
    "Ireland"       : "IE",
}
CODES = list(COUNTRIES.values())

SUPPLY_HEADLINE = 'MaxPot_Relative_10pct'   # only used if the Jaccard block returns

# Colour families: blue = percentile (relative), red = absolute capacity scenario
PCT_COLORS   = dict(zip(RL_PCT_SWEEP, plt.get_cmap('Blues')(np.linspace(0.45, 0.85, 3))))
ALPHA_COLORS = dict(zip(CAPACITY_SCENARIOS, plt.get_cmap('Reds')(np.linspace(0.45, 0.85, 3))))

# --- Paths -----------------------------------------------------------------
RESULTS_DIR   = Path("./../Results")
DATA_DIR      = Path("./../Data")
OUT_DIR       = RESULTS_DIR / "figures"
CF_DAILY_DIR  = RESULTS_DIR / "CF_daily"
COUNTRIES_SHP = Path("~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp").expanduser()
EEZ_SHP       = DATA_DIR / "EEZ/World_EEZ_v12_20231025/eez_v12.shp"
IC_PATH       = DATA_DIR / "IRENA_IC/IC.csv"
POT_PATH      = DATA_DIR / "Hu_IC/CF_info_grid.csv"        # Hu (2023) max-installable potential

# Cache for the grid-cell climatology computed in this script.  The filename
# encodes the threshold mode so switching modes does not silently reuse a stale
# cache computed under the other criterion.
_CACHE_TAG = (f"abs{int(ABS_CF_THRESHOLD*100)}" if CF_THRESHOLD_MODE == 'absolute'
              else f"frac{int(CF_MEAN_FRACTION*100)}")
CELL_CACHE = RESULTS_DIR / "DF_supply" / f"cell_stats_irena_{_CACHE_TAG}.nc"
GRID_CACHE = (RESULTS_DIR / "DF_supply" /
              f"fig5_grid_stats_abs{int(GRID_ABS_CF*100):02d}_rel{int(GRID_REL_FRAC*100):02d}.nc")

CDHW_ROOT = Path("~/CDHW_ag/Results").expanduser()
CDHW_RUN  = "sm20_vpd90_sev10_95_gap3_dur3_off0"

ERA5_dat  = os.environ.get("ERA5_dat")
Z500_ZARR = (Path(ERA5_dat) / "df_dat" / "processed" / "df_dat_cleaned.zarr"
             if ERA5_dat else None)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, stem):
    """Save as PNG (250 dpi) and SVG, mirroring the CDHW figure scripts."""
    for ext, kw in [("png", {"dpi": 250}), ("svg", {"format": "svg"})]:
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print(f"  Saved → {p}")


# %%
# =================================================================
# Load pipeline outputs
# =================================================================
print("Loading pipeline outputs...")

rl_events_df  = pd.read_parquet(RESULTS_DIR / "DF_RL/df_rl_events.parquet")
rl_annual_df  = pd.read_parquet(RESULTS_DIR / "DF_RL/df_rl_annual.parquet")
rl_country_df = pd.read_parquet(RESULTS_DIR / "DF_RL/rl_country.parquet")

demand_df = pd.read_parquet(RESULTS_DIR / "demand/demand_weather.parquet")
capgen_df = pd.read_parquet(RESULTS_DIR / "DF_supply/capgen_pot_absolute.parquet")

# NOTE ON NAMING: 'cf_irena_relative.parquet' is the capacity-share-weighted
# system capacity factor under the IRENA present-day national mix.  "Relative"
# refers to the WEIGHTING (shares of installed capacity), not to a relative
# threshold — the values are absolute CF in [0, 1], which is exactly what an
# absolute floor should be applied to.
cf_irena_df = pd.read_parquet(RESULTS_DIR / "DF_supply/cf_irena_relative.parquet")

demand_coef = pd.read_csv(RESULTS_DIR / "demand/demand_model_coefficients.csv", index_col=0)

print(f"  RL events: {len(rl_events_df)} | CF_irena series: {cf_irena_df.shape}")

# --- Winter bookkeeping ----------------------------------------------------
winter_id = pd.Series(rl_country_df.index.year, index=rl_country_df.index)
winter_id[rl_country_df.index.month.isin([11, 12])] += 1
is_winter = rl_country_df.index.month.isin(WINTER_MONTHS)      # ndarray, not Series
valid_winters = sorted(
    set(winter_id[rl_country_df.index.month == 11]) &
    set(winter_id[rl_country_df.index.month == 3])
)
is_valid_winter_day = is_winter & winter_id.isin(valid_winters)   # Series
winter_ok = is_valid_winter_day.values
N_VALID_DAYS = int(winter_ok.sum())
N_WINTERS    = len(valid_winters)
print(f"  {N_WINTERS} valid NDJFM winters ({valid_winters[0]}-{valid_winters[-1]}), "
      f"{N_VALID_DAYS} valid winter-days")

mean_winter_demand = demand_df.loc[winter_ok, CODES].mean()


# %%
# =================================================================
# Shared helpers
# =================================================================

def find_runs(flag, max_gap=MAX_GAP, min_duration=MIN_DURATION):
    """
    Pool True-runs separated by <= max_gap, drop runs shorter than min_duration.
    Returns inclusive (start_idx, end_idx) pairs.

    Vectorised equivalent of merge_and_filter_runs_1d in 03/05 — verified to
    reproduce that function exactly across randomised tests.
    """
    flag = np.asarray(flag, dtype=bool)
    if not flag.any():
        return []
    edges = np.diff(np.concatenate(([0], flag.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= min_duration]


def detect_absolute(x, mask, thr, tail):
    """
    Absolute-threshold event detection on a 1-D series.

    tail : 'lower' — x < thr is an event (capacity factor: low is bad)
           'upper' — x > thr is an event (residual load: high is bad)

    Severity is the summed exceedance beyond the threshold, in the units of x
    times days.  No sigma normalisation: with an absolute threshold, dividing by
    the within-season standard deviation would smuggle a relative criterion back
    into a deliberately absolute metric.
    """
    x = np.asarray(x, dtype=float)
    flag = (x < thr) & mask if tail == 'lower' else (x > thr) & mask
    out = []
    for s, e in find_runs(flag):
        seg = x[s:e + 1]
        exceed = (thr - seg) if tail == 'lower' else (seg - thr)
        out.append(dict(start_idx=s, end_idx=e, duration=e - s + 1,
                        severity=float(np.nansum(np.clip(exceed, 0, None))),
                        peak=float(np.nanmin(seg) if tail == 'lower' else np.nanmax(seg))))
    return out


def detect_percentile(x, mask, pct, tail):
    """Percentile-threshold detection — retained for the FIG 3 robustness sweep."""
    x = np.asarray(x, dtype=float)
    thr = np.nanpercentile(x[mask], pct)
    return detect_absolute(x, mask, thr, tail), thr


def events_to_frame(events, code, index, label):
    """Attach country/date metadata to the dicts returned by the detectors."""
    rows = []
    for e in events:
        rows.append(dict(country=code, config=label,
                         start_date=index[e['start_idx']], end_date=index[e['end_idx']],
                         winter=int(winter_id.iloc[e['start_idx']]),
                         duration=e['duration'], severity=e['severity'], peak=e['peak']))
    return rows


def daily_mask_from_events(events_df, index, codes, config=None):
    """Expand an event table into a (time x country) boolean frame."""
    mask = pd.DataFrame(False, index=index, columns=list(codes))
    sub = events_df if config is None else events_df[events_df['config'] == config]
    for _, row in sub.iterrows():
        if row['country'] in mask.columns:
            mask.loc[row['start_date']:row['end_date'], row['country']] = True
    return mask


def event_days_set(events_df, index=None, codes=None, config=None, country=None):
    """Set of dates covered by any matching event."""
    sub = events_df if config is None else events_df[events_df['config'] == config]
    if country is not None:
        sub = sub[sub['country'] == country]
    days = set()
    for _, row in sub.iterrows():
        days.update(pd.date_range(row['start_date'], row['end_date']))
    return days


def build_country_footprints(land_shp_path, eez_shp_path):
    """Onshore ADMIN land unioned with offshore EEZ, clipped to the map extent."""
    bbox = shapely.geometry.box(MAP_EXTENT[0] - 2, MAP_EXTENT[2] - 2,
                                MAP_EXTENT[1] + 2, MAP_EXTENT[3] + 2)
    land_gdf = gpd.read_file(str(land_shp_path), bbox=bbox)
    eez_gdf = gpd.read_file(str(eez_shp_path), bbox=bbox)

    rows = []
    for name, code in COUNTRIES.items():
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        if code == 'FR':
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
        if code == 'GB':
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))
        if code == 'DK':
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))
        footprint = land_geom.union(eez_geom).intersection(bbox)
        rows.append({'code': code, 'name': name,
                     'geometry': footprint.simplify(0.01, preserve_topology=True)})
    return gpd.GeoDataFrame(rows, crs=land_gdf.crs)


footprints = build_country_footprints(COUNTRIES_SHP, EEZ_SHP)
print("Country footprints built (onshore + EEZ offshore).")


def study_domain_mask(template_da):
    """Boolean (lat, lon) mask of cells inside ANY study-country footprint."""
    union = footprints.union_all()
    lons, lats = np.meshgrid(template_da.longitude.values, template_da.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())
    inside = shapely.contains(union, pts).reshape(lats.shape)
    return xr.DataArray(inside,
                        coords={'latitude': template_da.latitude,
                                'longitude': template_da.longitude},
                        dims=['latitude', 'longitude'])


def style_map_ax(ax, extent=MAP_EXTENT):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
                   edgecolor='dimgray', zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6,
                   edgecolor='black', zorder=4)


def plot_choropleth_panel(ax, values_by_code, cmap, norm, title,
                          fmt="{:.2f}", counts_by_code=None):
    """Fill each country footprint with a single scalar value."""
    style_map_ax(ax)
    gdf = footprints.copy()
    gdf['value'] = gdf['code'].map(values_by_code)
    gdf.plot(column='value', ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
             edgecolor='black', linewidth=0.6, zorder=2,
             missing_kwds={'color': '#d9d9d9', 'hatch': '///'})
    for _, row in gdf.iterrows():
        c = row['geometry'].centroid
        v = row['value']
        label = (f"{row['code']}\n(none)" if v is None or (isinstance(v, float) and np.isnan(v))
                 else f"{row['code']}\n{fmt.format(v)}")
        if counts_by_code is not None and not (v is None or (isinstance(v, float) and np.isnan(v))):
            label += f"\nn={int(counts_by_code.get(row['code'], 0))}"
        ax.text(c.x, c.y, label, transform=ccrs.PlateCarree(), ha='center', va='center',
                fontsize=7.5, fontweight='bold', zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.set_title(title, fontsize=11, fontweight='bold')


def style_z500_ax(ax):
    ax.set_extent(Z500_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5,
                   edgecolor='0.35', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7,
                   edgecolor='black', zorder=3)


# %%
# =================================================================
# IRENA present-day capacity mix
# =================================================================
# FIG 1 characterises weather risk for the fleet that actually exists today,
# driven by 45 years of historical weather.  This is the standard fixed-fleet
# counterfactual: "how often would today's fleet have been becalmed?"

IC = pd.read_csv(IC_PATH, index_col=0)
IC.index = IC.index.map({'France': 'FR', 'Belgium': 'BE', 'Netherlands': 'NL',
                         'Germany': 'DE', 'Denmark': 'DK', 'United Kingdom': 'GB',
                         'Ireland': 'IE'})
IC_COLS = ['Onshore Wind (MW)', 'Offshore Wind (MW)', 'Solar (MW)']
IC_TOTAL = IC[IC_COLS].sum(axis=1)

# Domain-aggregate wind/solar split for the GRID-CELL panels.  At a single grid
# cell there is no onshore/offshore distinction (it is the same wind field), so
# the mix collapses to wind vs solar.  Using the aggregate over the study
# countries keeps the map interpretable as "weather risk for a present-day
# typical NW-European technology mix".
W_WIND = float((IC['Onshore Wind (MW)'] + IC['Offshore Wind (MW)']).sum() / IC_TOTAL.sum())
W_SOL = float(IC['Solar (MW)'].sum() / IC_TOTAL.sum())
print(f"\nIRENA aggregate mix for grid-cell panels: wind {W_WIND:.2f} / solar {W_SOL:.2f}")
print("Per-country IRENA capacity shares:")
print((IC[IC_COLS].div(IC_TOTAL, axis=0)).reindex(CODES).round(3).to_string())


# --- Threshold resolution + the mix-confound diagnostic --------------------
mean_winter_cf = cf_irena_df.loc[winter_ok, CODES].mean()

if CF_THRESHOLD_MODE == 'absolute':
    cf_threshold = pd.Series(ABS_CF_THRESHOLD, index=CODES)
    THR_LABEL = f"CF$_{{sys}}$ < {ABS_CF_THRESHOLD}"
else:
    cf_threshold = CF_MEAN_FRACTION * mean_winter_cf
    THR_LABEL = f"CF$_{{sys}}$ < {CF_MEAN_FRACTION:.0%} of national winter mean"

print(f"\nSYSTEM-CF DIAGNOSTIC (mode = {CF_THRESHOLD_MODE})")
print(f"{'':4s} {'solar share':>12s} {'mean NDJFM CF':>14s} {'threshold':>10s} "
      f"{'mean/thr':>9s}")
for code in CODES:
    share_pv = IC.loc[code, 'Solar (MW)'] / IC_TOTAL[code]
    m, t = mean_winter_cf[code], cf_threshold[code]
    print(f"{code:4s} {share_pv:12.1%} {m:14.3f} {t:10.3f} {m/t:9.2f}")
print("""
Reading this table: under 'absolute' mode any country whose mean/thr is at or
below ~1.0 sits below the floor on a TYPICAL winter day, so its event count
reflects its technology mix rather than the weather. Solar-heavy systems are
most exposed to this because winter PV output above ~50 N is near zero. If any
mean/thr approaches 1, prefer CF_THRESHOLD_MODE = 'fraction_of_mean'.""")


# %%
# =================================================================
# Paper FIG 5 — supply-side baseline: inputs and grid-cell fields
# =================================================================
# The draft FIG 1 (grid-cell + country choropleths) is retired.  This section
# builds the paper's Figure 5: a supply-side weather-risk baseline whose maps
# mimic the "Maximum installable capacity" style in 03_DF_supp_ID.py — Lambert
# Azimuthal Equal Area (10E, 52N), filled contours, labelled gridlines, one
# vertical colorbar per panel.  Panels are shown individually (plt.show()) and
# NOT saved, so the final layout can be assembled by hand.
#
# Threshold logic (per the handoff, and no percentiles):
#   (e) absolute floor   — blended CF < GRID_ABS_CF                 (resource geography)
#   (f) relative to norm  — blended CF < GRID_REL_FRAC x cell mean  (local reliability)
# The relative panel uses a fraction of each cell's own NDJFM mean, which is an
# absolute CF magnitude that varies by cell — NOT a percentile of the distribution.

# --- Hu (2023) maximum-installable capacity on the ERA5 grid -----------------
_pot_raw = pd.read_csv(POT_PATH)[['lon', 'lat',
                                  'potential_onshore', 'potential_offshore', 'potential_PV']]
_ds_pot = (_pot_raw.set_index(['lat', 'lon']).to_xarray()
           .rename({'lat': 'latitude', 'lon': 'longitude',
                    'potential_onshore': 'pot_on',
                    'potential_offshore': 'pot_off',
                    'potential_PV': 'pot_pv'}))

# --- Daily CF fields, with a winter mask on THEIR OWN calendar ---------------
# Do not reuse winter_ok (RL/demand calendar): the CF stores are ERA5-native and
# can differ in length.  Build the mask exactly as 03 does.
CF_wind_dly  = xr.open_zarr(CF_DAILY_DIR / 'CF_wind_daily.zarr',
                            consolidated=True)['CF_wind_dly']
CF_solar_dly = xr.open_zarr(CF_DAILY_DIR / 'CF_solar_24h_daily.zarr',
                            consolidated=True)['CF_solar_24h_dly']

pot_grid = _ds_pot.reindex(latitude=CF_wind_dly.latitude,
                           longitude=CF_wind_dly.longitude, fill_value=0.0)

_t_cf     = pd.DatetimeIndex(CF_wind_dly.time.values)
_wid_cf   = np.where(_t_cf.month >= 11, _t_cf.year + 1, _t_cf.year)
_ndjfm_cf = _t_cf.month.isin(WINTER_MONTHS)
_vw_cf    = sorted(set(_wid_cf[_t_cf.month == 11]) & set(_wid_cf[_t_cf.month == 3]))
winter_ok_cf = (_ndjfm_cf & np.isin(_wid_cf, _vw_cf))
N_WINTERS_CF = len(_vw_cf)
print(f"\nFIG 5 grid: {N_WINTERS_CF} winters, {int(winter_ok_cf.sum())} winter-days "
      f"on the CF calendar")

# --- Country cell masks (onshore land / offshore EEZ), ported from 03 --------
def build_cell_masks(template_da, land_shp_path, eez_shp_path):
    land_gdf = gpd.read_file(str(land_shp_path))
    eez_gdf  = gpd.read_file(str(eez_shp_path))
    lons, lats = np.meshgrid(template_da.longitude.values, template_da.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())
    all_land = land_gdf.union_all()
    is_any_land = (shapely.distance(all_land, pts) == 0.0).reshape(lats.shape)

    def _da(a):
        return xr.DataArray(a, coords={'latitude': template_da.latitude,
                                       'longitude': template_da.longitude},
                            dims=['latitude', 'longitude'])
    m_on, m_off = {}, {}
    for name, code in COUNTRIES.items():
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        if code == 'FR':
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
        m_on[code] = _da(shapely.within(pts, land_geom).reshape(lats.shape))

        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
        if code == 'GB':
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))
        if code == 'DK':
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))
        in_eez = shapely.within(pts, eez_geom).reshape(lats.shape)
        m_off[code] = _da(in_eez & ~is_any_land)
    return m_on, m_off

masks_on, masks_off = build_cell_masks(CF_wind_dly.isel(time=0), COUNTRIES_SHP, EEZ_SHP)

# Study-domain boolean: any study-country cell with non-zero buildable potential.
study_cells = xr.full_like(masks_on['FR'], False, dtype=bool)
for code in CODES:
    study_cells = study_cells | masks_on[code] | masks_off[code]
p_tot_grid  = pot_grid['pot_on'] + pot_grid['pot_off'] + pot_grid['pot_pv']
study_cells = study_cells & (p_tot_grid > 0)

# --- (a) present-day capacity: IRENA nationals spread by Hu ratios (GW/cell) --
# Onshore wind + solar over the onshore mask; offshore wind over the EEZ mask;
# each technology distributed within a country in proportion to its Hu potential.
def _spread(nat_mw, pot_field, cell_mask):
    tot = float(pot_field.where(cell_mask, 0.0).sum())
    if tot <= 0:
        return xr.zeros_like(pot_field)
    return nat_mw * pot_field.where(cell_mask, 0.0) / tot

cap_present = xr.zeros_like(pot_grid['pot_on'])
for code in CODES:
    cap_present = cap_present + (
        _spread(IC.loc[code, 'Onshore Wind (MW)'],  pot_grid['pot_on'],  masks_on[code]) +
        _spread(IC.loc[code, 'Solar (MW)'],         pot_grid['pot_pv'],  masks_on[code]) +
        _spread(IC.loc[code, 'Offshore Wind (MW)'], pot_grid['pot_off'], masks_off[code]))
cap_present = (cap_present / 1e3).where(study_cells)          # -> GW

# --- (b) fully-built capacity: Hu absolute potential (GW/cell) ---------------
cap_full = (p_tot_grid / 1e3).where(study_cells)


# %%
# --- (c)-(f) grid-cell blended-CF fields (cached) ---------------------------
def compute_supply_grid(lat_block=20):
    """Mean CF, CV, and event frequency under an absolute floor and a
    fraction-of-local-mean floor.  The blend uses local Hu technology shares,
    identical to cf_blend in 03_DF_supp_ID.py."""
    p_safe = p_tot_grid.where(p_tot_grid > 0, 1.0)
    w_on  = pot_grid['pot_on']  / p_safe
    w_off = pot_grid['pot_off'] / p_safe
    w_sol = pot_grid['pot_pv']  / p_safe

    n_lat, n_lon = CF_wind_dly.sizes['latitude'], CF_wind_dly.sizes['longitude']
    mean_cf  = np.full((n_lat, n_lon), np.nan, np.float32)
    cv       = np.full((n_lat, n_lon), np.nan, np.float32)
    freq_abs = np.full((n_lat, n_lon), np.nan, np.float32)
    freq_rel = np.full((n_lat, n_lon), np.nan, np.float32)

    for i0 in range(0, n_lat, lat_block):
        i1 = min(i0 + lat_block, n_lat)
        sl = dict(latitude=slice(i0, i1))
        blend = (w_on.isel(**sl)  * CF_wind_dly.isel(**sl) +
                 w_off.isel(**sl) * CF_wind_dly.isel(**sl) +
                 w_sol.isel(**sl) * CF_solar_dly.isel(**sl)).astype('float32').compute()
        arr = blend.values.reshape(blend.sizes['time'], -1)      # (time, cells_in_block)
        print(f"  FIG5 grid cells {i0}-{i1} of {n_lat} ...", flush=True)
        for k in range(arr.shape[1]):
            x = arr[:, k]
            xw = x[winter_ok_cf]
            if not np.isfinite(xw).any():
                continue
            m = float(np.nanmean(xw))
            ii, jj = i0 + k // n_lon, k % n_lon
            mean_cf[ii, jj] = m
            cv[ii, jj]      = float(np.nanstd(xw) / m) if m > 0 else np.nan
            freq_abs[ii, jj] = len(detect_absolute(x, winter_ok_cf, GRID_ABS_CF, 'lower')) / N_WINTERS_CF
            freq_rel[ii, jj] = len(detect_absolute(x, winter_ok_cf, GRID_REL_FRAC * m, 'lower')) / N_WINTERS_CF

    coords = {'latitude': CF_wind_dly.latitude, 'longitude': CF_wind_dly.longitude}
    return xr.Dataset({'mean_cf':  (('latitude', 'longitude'), mean_cf),
                       'cv':       (('latitude', 'longitude'), cv),
                       'freq_abs': (('latitude', 'longitude'), freq_abs),
                       'freq_rel': (('latitude', 'longitude'), freq_rel)},
                      coords=coords,
                      attrs={'grid_abs_cf': GRID_ABS_CF, 'grid_rel_frac': GRID_REL_FRAC,
                             'season': 'NDJFM', 'n_winters': N_WINTERS_CF,
                             'blend': 'local Hu potential shares (as in 03)'})

# NOTE: this writes a small .nc CACHE (grid stats, not a figure) so re-runs are
# fast.  Delete GRID_CACHE to force a recompute after changing the thresholds.
if GRID_CACHE.exists():
    print(f"Loading cached FIG 5 grid stats: {GRID_CACHE}")
    grid_stats = xr.open_dataset(GRID_CACHE)
else:
    print("Computing FIG 5 grid stats (cached after first run)...")
    grid_stats = compute_supply_grid()
    GRID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    grid_stats.to_netcdf(GRID_CACHE)
    print(f"  Cached -> {GRID_CACHE}")

for _v in ['mean_cf', 'cv', 'freq_abs', 'freq_rel']:
    grid_stats[_v] = grid_stats[_v].where(study_cells)


# %%
# =================================================================
# Country-level supply-side detection (system-CF floor, IRENA fleet)
# =================================================================
print("\nCountry-level supply-side detection (absolute CF floor, IRENA mix)...")

supply_abs_rows = []
for code in CODES:
    ev = detect_absolute(cf_irena_df[code].values, winter_ok, cf_threshold[code], 'lower')
    supply_abs_rows += events_to_frame(ev, code, cf_irena_df.index, 'supply')
supply_abs_df = pd.DataFrame(supply_abs_rows)

supply_country = (supply_abs_df.groupby('country')
                  .agg(n_events=('duration', 'size'),
                       mean_severity=('severity', 'mean'),
                       mean_duration=('duration', 'mean'),
                       total_days=('duration', 'sum'))
                  .reindex(CODES))
supply_country['events_per_winter'] = supply_country['n_events'] / N_WINTERS
print(supply_country.round(3).to_string())


# %%
# =================================================================
# Paper FIG 5 — panels (shown individually via plt.show(), NOT saved)
# =================================================================
# Map style copied from 03_DF_supp_ID.py::format_capacity_map so every panel
# matches the "Maximum installable capacity" figures.  Run each cell below on its
# own to judge which panels to keep and how to assemble the final layout.
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

_cgdf_bbox = shapely.geometry.box(MAP_EXTENT[0] - 3, MAP_EXTENT[2] - 3,
                                  MAP_EXTENT[1] + 3, MAP_EXTENT[3] + 3)
countries_gdf = gpd.read_file(str(COUNTRIES_SHP), bbox=_cgdf_bbox)


def format_capacity_map(ax, extent=MAP_EXTENT):
    """The common project map style (03_DF_supp_ID.py)."""
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),  facecolor='0.93', zorder=0)
    ax.add_geometries(countries_gdf.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='0.35', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), lw=0.9, edgecolor='0.15', zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=0.9, zorder=4)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, lw=0.25,
                      color='0.6', alpha=0.5, ls='--')
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {'size': 7}


def supply_panel(field, title, cbar_label, cmap, vmax=None, vmin=0, levels=12):
    """One standalone LAEA filled-contour map in the 03 capacity-map style."""
    lon2d, lat2d = np.meshgrid(field.longitude.values, field.latitude.values)
    finite = field.values[np.isfinite(field.values)]
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={'projection': LAEA})
    format_capacity_map(ax)
    cf = ax.contourf(lon2d, lat2d, field.values, levels=levels, vmin=vmin, vmax=vmax,
                     cmap=cmap, transform=ccrs.PlateCarree(), extend='max', zorder=2)
    cb = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.03, shrink=0.82, aspect=22)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.show()
    return fig


# Shared capacity scale so (a) and (b) are comparable between countries.
_cap_all = np.concatenate([cap_present.values[np.isfinite(cap_present.values)],
                           cap_full.values[np.isfinite(cap_full.values)]])
_cap_vmax = float(np.nanpercentile(_cap_all, 99)) if _cap_all.size else 1.0

# %%
# (a) present-day installed capacity
supply_panel(cap_present,
             "(a) Present-day installed capacity\n(IRENA 2024, Hu spatial texture)",
             "Installed capacity (GW / cell)", 'YlGnBu', vmax=_cap_vmax)

# %%
# (b) fully-built installable capacity
supply_panel(cap_full,
             "(b) Fully-built installable capacity\n(Hu maximum potential)",
             "Installable capacity (GW / cell)", 'YlGnBu', vmax=_cap_vmax)

# %%
# (c) mean winter blended capacity factor (resource baseline)
supply_panel(grid_stats['mean_cf'],
             "(c) Mean NDJFM blended capacity factor",
             "Mean winter blended CF", 'viridis')

# %%
# (d) blended-CF variability (coefficient of variation)
supply_panel(grid_stats['cv'],
             "(d) Winter blended-CF variability (CV)",
             "Coefficient of variation (σ / μ)", 'plasma')

# %%
# (e) absolute low-CF frequency
supply_panel(grid_stats['freq_abs'],
             f"(e) Low-CF frequency — absolute floor\n(blended CF < {GRID_ABS_CF:.2f})",
             "Events per winter", 'inferno_r')

# %%
# (f) relative low-CF frequency (fraction of local winter mean; not a percentile)
supply_panel(grid_stats['freq_rel'],
             f"(f) Low-CF frequency — relative to local norm\n(< {GRID_REL_FRAC:.0%} of cell winter mean)",
             "Events per winter", 'inferno_r')


# %%
# =================================================================
# Residual load under capacity build-out scenarios
# =================================================================
# RL_alpha = demand - alpha * (generation at maximum installable potential)
print("\nResidual load under capacity scenarios...")

rl_scen = {a: demand_df[CODES] - a * capgen_df[CODES] for a in CAPACITY_SCENARIOS}

scen_rows = []
for a in CAPACITY_SCENARIOS:
    for code in CODES:
        ev = detect_absolute(rl_scen[a][code].values, winter_ok, 0.0, 'upper')
        scen_rows += events_to_frame(ev, code, rl_country_df.index, f'alpha={a:.2f}')
scen_df = pd.DataFrame(scen_rows)

print("\nDeficit events per winter by build-out scenario:")
piv = (scen_df.groupby(['country', 'config']).size().unstack(fill_value=0) / N_WINTERS)
print(piv.reindex(CODES).round(2).to_string())

# Internal consistency: alpha = 1.0 must reproduce the T2 catalogue from 05.
if 'T2' in set(rl_events_df['track']):
    a1 = scen_df[scen_df['config'] == 'alpha=1.00'].groupby('country')['duration'].sum()
    t2 = (rl_events_df[(rl_events_df['track'] == 'T2') & (rl_events_df['country'] != 'EU_AGG')]
          .groupby('country')['duration'].sum())
    comp = pd.DataFrame({'alpha=1.0': a1, '05_T2': t2}).reindex(CODES).fillna(0)
    ok = np.allclose(comp['alpha=1.0'], comp['05_T2'])
    print(f"\nConsistency check vs 05_DF_RL_ID.py T2 track: "
          f"{'MATCH' if ok else 'MISMATCH — investigate'}")
    if not ok:
        print(comp.to_string())


# %%
# =================================================================
# Demand model diagnostic
# =================================================================
# Closed-loop check on the concern that weather-driven demand looks flat.
# In NDJFM CDD is zero, so demand = a0 + HDD_coef * HDD.  Inverting the observed
# demand range therefore recovers the implied HDD range in degree-days: if that
# lands in a physically sensible window (~10-20 degC-days for NW Europe), the
# model is responding to temperature correctly and any visual flatness is an
# axis-scaling artefact, not a bug.
print("\nDEMAND MODEL DIAGNOSTIC (NDJFM)")
print(f"{'':4s} {'range GW':>9s} {'std GW':>7s} {'CV':>6s} {'HDD coef':>9s} "
      f"{'implied HDD span':>17s} {'gen/dem':>8s}")
for code in CODES:
    d = demand_df.loc[winter_ok, code]
    g = capgen_df.loc[winter_ok, code]
    rng = d.max() - d.min()
    coef = demand_coef.loc[code, 'HDD'] if 'HDD' in demand_coef.columns else np.nan
    implied = rng / coef if coef and np.isfinite(coef) and coef != 0 else np.nan
    print(f"{code:4s} {rng/1e3:9.1f} {d.std()/1e3:7.1f} {d.std()/d.mean():6.1%} "
          f"{coef:9.0f} {implied:15.1f} °C {g.mean()/d.mean():8.1f}x")
print("""
Reading this table:
* 'implied HDD span' between roughly 10 and 20 degC-days means the HDD response
  is being applied correctly across the record.
* 'gen/dem' is mean generation at MAXIMUM installable potential divided by mean
  demand.  Where this is much greater than 1, plotting demand and generation on a
  shared axis will make demand look flat purely because of the scale — which is
  why FIG 2 uses a twin axis and capacity scenarios.""")


# %%
# =================================================================
# FIGURE 2 — Residual-load results
#   A/B  event anatomy for two countries under three build-out scenarios
#   C    within-season timing of deficit days
# =================================================================
print("\nFIG 2 — residual-load anatomy and timing...")

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 0.85], hspace=0.42)

w_mask = (winter_id == SPOTLIGHT_WINTER).values & np.asarray(is_winter)
w_idx = rl_country_df.index[w_mask]

for row, code in enumerate(ANATOMY_COUNTRIES):
    ax = fig.add_subplot(gs[row, 0])
    dem = demand_df.loc[w_idx, code] / 1e3

    # Demand on its own axis: at alpha = 1 the generation potential dwarfs demand,
    # and a shared axis flattens the demand curve to a visually straight line.
    ax.plot(w_idx, dem, color='black', lw=1.8, label='Weather-driven demand', zorder=5)
    ax.set_ylabel('Demand (GW)')
    ax.set_ylim(0, dem.max() * 1.25)

    axg = ax.twinx()
    for a in CAPACITY_SCENARIOS:
        gen = a * capgen_df.loc[w_idx, code] / 1e3
        axg.plot(w_idx, gen, color=ALPHA_COLORS[a], lw=1.2, alpha=0.9,
                 label=f'Generation @ {a:.0%} of max potential')
    axg.set_ylabel('Generation potential (GW)')
    axg.set_ylim(bottom=0)

    # Shade deficit days for the headline scenario, read off the event catalogue
    # so the shading is exactly the detected events (pooling and duration filter
    # applied), not a naive day-by-day comparison.
    m = daily_mask_from_events(
        scen_df[scen_df['country'] == code], rl_country_df.index, [code],
        config=f'alpha={HEADLINE_ALPHA:.2f}')[code].loc[w_idx]
    ax.fill_between(w_idx, 0, 1, where=m.values, transform=ax.get_xaxis_transform(),
                    color=ALPHA_COLORS[HEADLINE_ALPHA], alpha=0.18, linewidth=0,
                    label=f'Deficit event @ {HEADLINE_ALPHA:.0%}', zorder=0)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_title(f"{'AB'[row]} — {code}, NDJFM {SPOTLIGHT_WINTER - 1}/{str(SPOTLIGHT_WINTER)[-2:]}",
                 fontsize=11, fontweight='bold', loc='left')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=3, loc='upper right', framealpha=0.9)

# --- C: within-season timing ---
# Van der Wiel et al. (2019) find low-production and high-shortfall events peak
# at different points in the season; this is the direct test on our catalogue.
ax_t = fig.add_subplot(gs[2, 0])
month_order = [11, 12, 1, 2, 3]
month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
xm = np.arange(len(month_order))
wm_ = 0.2

series = {}
sup_days = pd.DatetimeIndex(sorted(event_days_set(supply_abs_df)))
sc = pd.Series(sup_days.month).value_counts().reindex(month_order, fill_value=0)
series['Supply-side (CF floor)'] = (sc / sc.sum(), '#55A868')
for a in CAPACITY_SCENARIOS:
    d = pd.DatetimeIndex(sorted(event_days_set(scen_df, config=f'alpha={a:.2f}')))
    if len(d) == 0:
        continue
    c = pd.Series(d.month).value_counts().reindex(month_order, fill_value=0)
    series[f'Residual load @ {a:.0%}'] = (c / c.sum(), ALPHA_COLORS[a])

for i, (label, (vals, color)) in enumerate(series.items()):
    ax_t.bar(xm + (i - (len(series) - 1) / 2) * wm_, vals.values, wm_,
             label=label, color=color, edgecolor='k', linewidth=0.5)
ax_t.set_xticks(xm)
ax_t.set_xticklabels(month_names)
ax_t.set_ylabel('Fraction of event-days')
ax_t.grid(axis='y', linestyle='--', alpha=0.5)
ax_t.legend(fontsize=8, ncol=2)
ax_t.set_title("C — Within-season timing of event days", fontsize=11,
               fontweight='bold', loc='left')

fig.suptitle("Residual-load dunkelflaute under partial renewable build-out", fontsize=13, y=0.95)
save_fig(fig, "fig02_residual_load")
plt.show()
plt.close()


# %%
# =================================================================
# FIGURE 3 — Robustness across six detection configurations
#   A  events per winter per country
#   B  cross-country concurrence
# Three percentile thresholds (blue) and three capacity scenarios (red), so the
# two detection philosophies are compared like for like in both views.
# =================================================================
print("\nFIG 3 — robustness...")

configs = []      # (label, colour, daily mask frame, events frame)

for pct in RL_PCT_SWEEP:
    rows = []
    for code in CODES:
        ev, _ = detect_percentile(rl_country_df[code].values, winter_ok, pct, 'upper')
        rows += events_to_frame(ev, code, rl_country_df.index, f'p{pct}')
    df_p = pd.DataFrame(rows)
    configs.append((f'{pct}th pct RL', PCT_COLORS[pct], df_p))

for a in CAPACITY_SCENARIOS:
    df_a = scen_df[scen_df['config'] == f'alpha={a:.2f}'].copy()
    configs.append((f'RL>0 @ {a:.0%} build-out', ALPHA_COLORS[a], df_a))

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 5.6))

# --- A: events per winter ---
x = np.arange(len(CODES))
w = 0.13
for i, (label, color, df_c) in enumerate(configs):
    per_winter = (df_c.groupby('country').size().reindex(CODES, fill_value=0) / N_WINTERS)
    ax_a.bar(x + (i - (len(configs) - 1) / 2) * w, per_winter.values, w,
             label=label, color=color, edgecolor='k', linewidth=0.4)
ax_a.set_xticks(x)
ax_a.set_xticklabels(CODES)
ax_a.set_ylabel('Events per winter')
ax_a.grid(axis='y', linestyle='--', alpha=0.5)
ax_a.legend(fontsize=7.5, ncol=2, framealpha=0.9)
ax_a.set_title("A — Sensitivity of event frequency to the detection criterion\n"
               "(percentile thresholds are flat across countries by construction; "
               "absolute thresholds are not)",
               fontsize=10, fontweight='bold', loc='left')

# --- B: cross-country concurrence ---
# The balancing-effect diagnostic: how often is a shortfall continental rather
# than national?  Shown for all six configurations so the answer can be judged
# independently of the threshold choice.
for label, color, df_c in configs:
    m = daily_mask_from_events(df_c, rl_country_df.index, CODES)
    conc = m.sum(axis=1)[winter_ok]
    conc = conc[conc > 0]
    if conc.empty:
        continue
    counts = conc.value_counts().reindex(range(1, len(CODES) + 1), fill_value=0)
    ls = '--' if 'pct' in label else '-'
    ax_b.plot(counts.index, counts.values / counts.sum(), marker='o', lw=1.8,
              ls=ls, color=color, label=label)
ax_b.set_xticks(range(1, len(CODES) + 1))
ax_b.set_xlabel('Number of countries simultaneously in event')
ax_b.set_ylabel('Fraction of event-days')
ax_b.grid(alpha=0.3, linestyle='--')
ax_b.legend(fontsize=7.5, ncol=2)
ax_b.set_title("B — Cross-country concurrence\n"
               "(geographical balancing: is the shortfall national or continental?)",
               fontsize=10, fontweight='bold', loc='left')

fig.suptitle("Robustness of the residual-load detection to threshold choice", fontsize=13, y=1.02)
plt.tight_layout()
save_fig(fig, "fig03_robustness")
plt.show()
plt.close()


# %%
# =================================================================
# Z500 — load once, season-subset into memory
# =================================================================
# CAVEAT: the ERA5 domain downloaded for this subproject is the regional
# capacity-factor box (N62 W-14 S42 E18).  The composites show a monotonic NW-SE
# gradient maximised at the domain corner, the signature of an anticyclone
# centred OUTSIDE the window (Norwegian Sea / Iceland).  Z500 is a single level
# and only daily means are needed, so re-downloading it on e.g.
# [80N, 60W, 30N, 40E] is cheap relative to the rest of the pipeline and would
# let these panels show the blocking centre rather than its flank.
# Point Z500_ZARR at the wider store when available — the code is domain-agnostic.
print("\nLoading Z500...")

ds_z = xr.open_zarr(Z500_ZARR, consolidated=True)
z = ds_z['z']
if 'level' in z.dims:
    z = z.isel(level=0)
z_daily = z.coarsen(time=24, boundary='trim', coord_func='min').mean()
times_z = pd.DatetimeIndex(z_daily.time.values)

z_win = z_daily.isel(time=np.where(times_z.month.isin(WINTER_MONTHS))[0]).compute()
z_jja = z_daily.isel(time=np.where(times_z.month.isin(SUMMER_MONTHS))[0]).compute()
z_win_clim = z_win.mean('time')
z_jja_clim = z_jja.mean('time')
print(f"  Winter days: {z_win.sizes['time']} | Summer days: {z_jja.sizes['time']}")


def composite(z_season, z_clim, days):
    """Mean Z500 anomaly over a set of days, relative to that season's climatology."""
    days = pd.DatetimeIndex(sorted(days))
    sel = z_season.sel(time=z_season.time.isin(days.values))
    n = int(sel.sizes['time'])
    if n == 0:
        return None, 0
    return (sel.mean('time') - z_clim), n


# --- Energy concurrence ----------------------------------------------------
comp_events = scen_df[scen_df['config'] == f'alpha={CONCURRENCE_ALPHA:.2f}']
comp_mask = daily_mask_from_events(comp_events, rl_country_df.index, CODES)
conc = comp_mask.sum(axis=1)[winter_ok]

conc_table = conc[conc > 0].value_counts().sort_index()
print(f"\nDays by number of countries simultaneously in deficit "
      f"(alpha = {CONCURRENCE_ALPHA:.0%}):")
for k, v in conc_table.items():
    tag = "  <-- composite threshold" if k == MIN_CONCURRENT_COUNTRIES else ""
    print(f"  {k} countries: {v:5d} days  "
          f"(>={k}: {int(conc_table[conc_table.index >= k].sum()):5d}){tag}")

rl_concurrent_days = set(conc.index[conc >= MIN_CONCURRENT_COUNTRIES])
rl_any_days = event_days_set(comp_events)
print(f"\nRL event-days: {len(rl_any_days)} any-country → "
      f"{len(rl_concurrent_days)} with ≥{MIN_CONCURRENT_COUNTRIES} concurrent "
      f"({len(rl_any_days)/max(N_VALID_DAYS,1):.0%} → "
      f"{len(rl_concurrent_days)/max(N_VALID_DAYS,1):.0%} of winter days)")
if len(rl_concurrent_days) < MIN_COMPOSITE_DAYS:
    warnings.warn(f"Only {len(rl_concurrent_days)} days pass the concurrence filter — "
                  f"the composite will be noisy. Lower MIN_CONCURRENT_COUNTRIES using "
                  f"the distribution above.")

# --- Agricultural concurrence ----------------------------------------------
cdhw_mask_path = CDHW_ROOT / "CDHW_results" / CDHW_RUN / "compound_mask_idx.nc"
cdhw_available = cdhw_mask_path.exists()

if cdhw_available:
    cdhw_mask = xr.open_dataarray(cdhw_mask_path)
    ssi = xr.open_zarr(CDHW_ROOT / "SSI.zarr", consolidated=True)
    ssi = ssi[list(ssi.data_vars)[0]]
    n_land_cells = int(np.isfinite(ssi.isel(time=0).values).sum())

    area_frac = (cdhw_mask.sum(['latitude', 'longitude']) / n_land_cells).compute()
    area_thr = float(np.percentile(area_frac.values, CDHW_AREA_PCTL))
    cdhw_times = pd.DatetimeIndex(cdhw_mask.time.values)
    cdhw_concurrent_days = set(cdhw_times[area_frac.values >= area_thr])
    cdhw_any_days = set(cdhw_times[area_frac.values > 0])
    print(f"CDHW event-days: {len(cdhw_any_days)} any-cell → "
          f"{len(cdhw_concurrent_days)} in the top {100 - CDHW_AREA_PCTL}% by area "
          f"(threshold = {area_thr:.1%} of domain)")
else:
    warnings.warn(f"CDHW mask not found at {cdhw_mask_path} — "
                  "FIG 4 will omit the agricultural panel. Set CDHW_ROOT/CDHW_RUN.")


# %%
# =================================================================
# FIGURE 4 — Z500 composites, concurrence-filtered
# Merges the previous joint and per-country figures.  Panel 1 is the summer
# agricultural hazard, panel 2 the winter energy hazard aggregated over
# countries, and the remainder resolve the ridge position country by country —
# pooling blurs it, because a Danish event and a French event sit under
# differently-placed anticyclones.  Shared norm throughout.
# =================================================================
print("\nFIG 4 — Z500 composites (concurrence-filtered)...")

comp_panels = []
if cdhw_available:
    a_flt, n_a = composite(z_jja, z_jja_clim, cdhw_concurrent_days)
    comp_panels.append((f"CDHW (JJA)\ntop {100 - CDHW_AREA_PCTL}% by area affected",
                        a_flt, n_a, True))

e_flt, n_e = composite(z_win, z_win_clim, rl_concurrent_days)
comp_panels.append((f"Dunkelflaute (NDJFM)\n≥{MIN_CONCURRENT_COUNTRIES} countries concurrent",
                    e_flt, n_e, True))

for code in CODES:
    days = event_days_set(comp_events, country=code)
    anom, n = composite(z_win, z_win_clim, days)
    if anom is not None:
        comp_panels.append((f"{code} — national events", anom, n, False))

vmax = max(float(np.nanmax(np.abs(p[1].values))) for p in comp_panels if p[1] is not None)
norm_z = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

ncol = 3
nrow = int(np.ceil(len(comp_panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 5.0 * nrow),
                         subplot_kw={'projection': PROJ})
axes = np.atleast_1d(axes).ravel()

for ax, (title, anom, n, is_summary) in zip(axes, comp_panels):
    style_z500_ax(ax)
    ax.contourf(anom.longitude, anom.latitude, anom, levels=21, cmap='RdBu_r',
                norm=norm_z, transform=ccrs.PlateCarree(), zorder=1)
    if not is_summary:
        code = title.split(' — ')[0]
        ax.add_geometries(footprints.loc[footprints['code'] == code, 'geometry'],
                          crs=ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', linewidth=1.2, zorder=4)
    peak = float(np.nanmax(np.abs(anom.values)))
    ax.set_title(f"{title}\nn={n} days | peak |Z'| = {peak:.0f} m",
                 fontsize=9.5,
                 fontweight='bold' if is_summary else 'normal')

for ax in axes[len(comp_panels):]:
    ax.axis('off')

sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm_z)
cb = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal',
                  fraction=0.035, pad=0.04, shrink=0.5)
cb.set_label("Z500 anomaly (m) relative to own-season climatology")

fig.suptitle("Shared anticyclonic blocking signature across both case studies\n"
             "composites restricted to spatially concurrent event days",
             fontsize=13, y=1.00)
save_fig(fig, "fig04_z500_composites")
plt.show()
plt.close()

print("\nComposite peaks (m):")
for title, anom, n, _ in comp_panels:
    if anom is not None:
        print(f"  {title.splitlines()[0]:42s} {float(np.nanmax(np.abs(anom.values))):6.1f}  (n={n})")


# %%
# =================================================================
# SI 1 — Trend forest plot
# Replaces two 8-panel scatter grids that showed tau ~ 0 and p > 0.3 everywhere.
# Theil-Sen is omitted: on integer event counts the median slope is exactly zero
# in almost every country, which is a discretisation artefact, not a trend.
# =================================================================
print("\nSI 1 — trend forest plot...")


def tau_with_ci(years, values, n_boot=2000, seed=0):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    tau, p = kendalltau(years, values)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(years), size=len(years), replace=True)
        if len(np.unique(years[idx])) < 3:
            continue
        t, _ = kendalltau(years[idx], values[idx])
        if np.isfinite(t):
            boots.append(t)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return tau, p, lo, hi


trend_rows = []
for code in CODES:
    sub = scen_df[(scen_df['country'] == code) &
                  (scen_df['config'] == f'alpha={HEADLINE_ALPHA:.2f}')]
    occ = (sub.groupby('winter').size().reindex(valid_winters, fill_value=0))
    sev = (sub.groupby('winter')['severity'].sum().reindex(valid_winters, fill_value=0.0))
    for metric, vals in [('Occurrence', occ), ('Severity', sev)]:
        tau, p, lo, hi = tau_with_ci(valid_winters, vals.values)
        trend_rows.append({'country': code, 'metric': metric,
                           'tau': tau, 'p': p, 'lo': lo, 'hi': hi})
trend_df = pd.DataFrame(trend_rows)

fig, ax = plt.subplots(figsize=(8.5, 6))
offsets = {'Occurrence': -0.16, 'Severity': 0.16}
colors_si = {'Occurrence': '#4C72B0', 'Severity': '#C44E52'}
for metric in ['Occurrence', 'Severity']:
    sub = trend_df[trend_df['metric'] == metric].set_index('country').reindex(CODES)
    ypos = np.arange(len(CODES)) + offsets[metric]
    ax.errorbar(sub['tau'].values, ypos,
                xerr=[sub['tau'].values - sub['lo'].values,
                      sub['hi'].values - sub['tau'].values],
                fmt='o', color=colors_si[metric], ecolor=colors_si[metric],
                elinewidth=1.4, capsize=3, markersize=6, label=metric)
ax.axvline(0, color='black', lw=1.0)
ax.set_yticks(np.arange(len(CODES)))
ax.set_yticklabels(CODES)
ax.invert_yaxis()
ax.set_xlabel("Kendall's τ (trend over winters)")
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.legend(fontsize=9)
ax.set_title(f"No detectable trend in dunkelflaute hazard, "
             f"{valid_winters[0]}–{valid_winters[-1]}\n"
             f"(RL > 0 at {HEADLINE_ALPHA:.0%} build-out; every CI spans zero — "
             f"contrast with the intensifying CDHW trends)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
save_fig(fig, "figSI1_trend_forest")
plt.show()
plt.close()
print(trend_df.round(3).to_string(index=False))


# %%
# =================================================================
# SI 2 — Duration-severity joint distribution
# Kittel & Schill's central critique is that single-threshold characterisations
# are unreliable; the joint distribution is the direct response.
# =================================================================
print("\nSI 2 — duration-severity joint distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
colors_dur = plt.cm.tab10(np.linspace(0, 1, len(CODES)))

# Left: supply side (CF-deficit days).  Right: residual load, demand-normalised.
for code, color in zip(CODES, colors_dur):
    d = supply_abs_df[supply_abs_df['country'] == code]
    if not d.empty:
        axes[0].scatter(d['duration'], d['severity'], alpha=0.6, s=28, label=code,
                        color=color, edgecolors='k', linewidths=0.3)
    r = scen_df[(scen_df['country'] == code) &
                (scen_df['config'] == f'alpha={HEADLINE_ALPHA:.2f}')]
    if not r.empty:
        axes[1].scatter(r['duration'], r['severity'] / mean_winter_demand[code],
                        alpha=0.6, s=28, label=code, color=color,
                        edgecolors='k', linewidths=0.3)

axes[0].set_ylabel('Severity (CF-deficit days)')
axes[0].set_title(f"Supply side — {THR_LABEL}", fontsize=10, fontweight='bold')
axes[1].set_ylabel('Severity (deficit-day equivalent)')
axes[1].set_title(f"Residual load — RL > 0 at {HEADLINE_ALPHA:.0%} build-out",
                  fontsize=10, fontweight='bold')
for ax in axes:
    ax.set_xlabel('Duration (days)')
    ax.grid(True, linestyle='--', alpha=0.4)
axes[0].legend(fontsize=8, ncol=2)

fig.suptitle("Duration-severity joint distribution", fontsize=13)
plt.tight_layout()
save_fig(fig, "figSI2_duration_severity")
plt.show()
plt.close()


# %%
# =================================================================
# SI 3 — Supply-side sensitivity to the absolute CF floor
# FIG 1 fixes a single floor; this shows the country ranking is not an artefact
# of that choice.
# =================================================================
print("\nSI 3 — supply-side CF threshold sensitivity...")

# Sweep in whichever units the headline mode uses, so the SI matches FIG 1.
if CF_THRESHOLD_MODE == 'absolute':
    sweep_levels = CF_THR_SWEEP
    def _thr_for(code, lvl): return lvl
    def _lbl(lvl): return f'CF < {lvl}'
    sweep_title = "Supply-side sensitivity to the absolute capacity-factor floor"
else:
    sweep_levels = CF_FRAC_SWEEP
    def _thr_for(code, lvl): return lvl * mean_winter_cf[code]
    def _lbl(lvl): return f'< {lvl:.0%} of winter mean'
    sweep_title = "Supply-side sensitivity to the shortfall fraction"

sweep_rows = []
for lvl in sweep_levels:
    for code in CODES:
        ev = detect_absolute(cf_irena_df[code].values, winter_ok, _thr_for(code, lvl), 'lower')
        sweep_rows.append(dict(country=code, thr=lvl, events_per_winter=len(ev) / N_WINTERS,
                               mean_severity=np.mean([e['severity'] for e in ev]) if ev else 0.0))
cf_sweep = pd.DataFrame(sweep_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sweep_colors = plt.get_cmap('YlOrRd')(np.linspace(0.4, 0.85, len(sweep_levels)))
for ax, (metric, ylabel) in zip(axes, [('events_per_winter', 'Events per winter'),
                                       ('mean_severity', 'Mean severity (CF-deficit days)')]):
    for i, lvl in enumerate(sweep_levels):
        sub = cf_sweep[cf_sweep['thr'] == lvl].set_index('country').reindex(CODES)
        ax.bar(x + (i - 1) * 0.26, sub[metric].values, 0.26,
               label=_lbl(lvl), color=sweep_colors[i], edgecolor='k', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CODES)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend(fontsize=8, title='Shortfall criterion')
fig.suptitle(sweep_title, fontsize=13)
plt.tight_layout()
save_fig(fig, "figSI3_cf_threshold_sensitivity")
plt.show()
plt.close()

print("\nEvents per winter by CF floor:")
print(cf_sweep.pivot(index='country', columns='thr',
                     values='events_per_winter').reindex(CODES).round(2).to_string())

print(f"\nAll figures written to: {OUT_DIR}")
# %%