#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_results.py — Dunkelflaute Results Visualization
===================================================
Publication figure set for the residual-load (RL) and supply-side dunkelflaute
detection pipelines, plus the joint (agriculture + energy) circulation figure.

Design notes on this revision
-----------------------------
Several panels in the previous version of this script were information-free by
construction and have been rebuilt:

1. RELATIVE-THRESHOLD TAUTOLOGY.  A within-season 90th-percentile threshold
   places 10% of days above it in *every* country by definition; gap-pooling and
   the 3-day minimum reduce that to ~5-6% everywhere.  T1 frequency therefore
   cannot vary meaningfully across countries, and the same applies to the
   copper-plate aggregate (which derives its own 90th percentile).  T1 is now
   used only for duration/severity; frequency panels use T2, and the balancing
   effect is shown via cross-country CONCURRENCE rather than a copper-plate
   aggregate.

2. COMPOSITE DILUTION.  The previous Z500 composites averaged over any day on
   which *any* cell/country was in event: 2637 of 4140 JJA days (64%) for CDHW
   and 1419 of ~6650 NDJFM days (21%) for RL.  Compositing two thirds of all
   summer days against the climatology of those same days necessarily returns
   ~zero, which is why the CDHW panel appeared blank.  Composites are now
   filtered on CONCURRENCE (spatial co-occurrence), not mere occurrence.

3. SEVERITY NORMALISATION.  05_DF_RL_ID.py computes severity_S as a raw sum of
   threshold exceedance in MW-days, with no sigma normalisation (unlike
   03_DF_supp_ID.py, which does divide by sigma).  Raw MW-days are not
   comparable across countries of different system size.  Severity is here
   additionally normalised by mean winter demand, giving an interpretable
   "deficit-day equivalent" unit.

4. SUPPLY-TRACK NAME BUG.  05_DF_RL_ID.py computes its Jaccard overlap against
   supply track 'T1', which does not exist (supply tracks are named
   'IRENA_Relative_10pct' etc.), so every Jaccard value there is 0 by
   construction.  Recomputed here against the correct headline track.

Figures produced
-----------------
  FIG A  Spatial climatology       — T2 frequency choropleth, mean duration,
                                     masked per-cell grid climatology
  FIG B  Event anatomy + validation— winter 2017/18 demand/generation/RL,
                                     RL-vs-supply-only overlap, within-season timing
  FIG C  Robustness                — capacity-assumption tracks (split axes),
                                     detection-threshold sensitivity,
                                     cross-country concurrence
  FIG D  Joint Z500 composite      — CDHW (JJA) vs RL (NDJFM), concurrence-filtered
  FIG E  Per-country Z500          — one composite per country, shared norm
  SI 1   Trend forest plot         — Kendall tau with bootstrap CI (replaces the
                                     16-panel scatter grids)
  SI 2   Duration-severity joint distribution
  SI 3   Severity choropleth       — demand-normalised, shared norm, n reported

Literature: Otero et al. 2022; Bloomfield 2021; Biewald et al. 2025;
Van der Wiel et al. 2019; Kittel & Schill 2024; Hu et al. 2023.

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
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
# =================================================================
# Config and params
# =================================================================

WINTER_MONTHS = [11, 12, 1, 2, 3]   # NDJFM
SUMMER_MONTHS = [6, 7, 8]           # JJA (CDHW season)

THRESHOLD_PCT = 90                  # RL Track 1 (relative) percentile
MAX_GAP       = 1                   # Otero'22 pooling
MIN_DURATION  = 3                   # minimum event duration (days)

# Threshold sensitivity sweep (SI / Fig C).  The headline value must be included.
RL_PCT_SWEEP     = [85, 90, 95]     # upper tail, residual load
SUPPLY_PCT_SWEEP = [10, 5]          # lower tail, supply-side CF (SEVERE_PCT = 5)

# --- Concurrence filtering for the Z500 composites -------------------------
# Energy: composite only days on which at least this many countries are
# simultaneously in event.  With 7 countries, 4 = simple majority.
MIN_CONCURRENT_COUNTRIES = 4
# Agriculture: composite only days in the top (1 - CDHW_AREA_PCTL/100) fraction
# of domain area affected.  90 => top decile of most spatially extensive days.
CDHW_AREA_PCTL = 90

# Spotlight winter for the event-anatomy panel (parallels the CDHW 2018 case).
SPOTLIGHT_WINTER = 2018             # NDJFM labelled by its Jan-Mar year

OCEAN_COLOR = '#C6E2F5'
LAND_COLOR  = '#F5F5F2'
MAP_EXTENT  = [-11.5, 16.5, 41.5, 59.5]      # [W, E, S, N] — country choropleths
# Z500 composites want the widest domain available; the regional ERA5 box is
# truncated at N62/W-14, which clips the blocking centre (see caveat below).
Z500_EXTENT = [-14.0, 18.0, 42.0, 62.0]
PROJ        = ccrs.LambertConformal(central_longitude=3, central_latitude=50)

COUNTRIES = {                       # Natural Earth ADMIN name -> code
    "France"        : "FR",
    "Belgium"       : "BE",
    "Netherlands"   : "NL",
    "Germany"       : "DE",
    "Denmark"       : "DK",
    "United Kingdom": "GB",
    "Ireland"       : "IE",
}
CODES = list(COUNTRIES.values())

RL_TRACKS      = ['T1', 'T2']
RL_TRACK_LABEL = {'T1': f'T1 — Relative ({THRESHOLD_PCT}th pct RL)',
                  'T2': 'T2 — Absolute (RL > 0, renewable supply deficit)'}
SUPPLY_TRACKS  = ['IRENA_Relative_10pct', 'MaxPot_Relative_10pct',
                  'MaxPot_Absolute_15pct_floor']
SUPPLY_TRACK_LABEL = {'IRENA_Relative_10pct': 'IRENA current mix',
                      'MaxPot_Relative_10pct': 'Hu max potential (ratio)',
                      'MaxPot_Absolute_15pct_floor': 'Hu max potential (abs. MW)'}
# Headline supply track — used for the RL-vs-supply-only overlap validation.
SUPPLY_HEADLINE = 'MaxPot_Relative_10pct'

# Consistent track colours across all figures
TRACK_COLORS  = {'T1': '#4C72B0', 'T2': '#C44E52'}
TRACK_PALETTE = {'IRENA_Relative_10pct': '#55A868',
                 'MaxPot_Relative_10pct': '#4C72B0',
                 'MaxPot_Absolute_15pct_floor': '#C44E52'}

# --- Paths -----------------------------------------------------------------
RESULTS_DIR   = Path("./../Results")
OUT_DIR       = RESULTS_DIR / "figures"
COUNTRIES_SHP = Path("~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp").expanduser()
EEZ_SHP       = Path("./../Data/EEZ/World_EEZ_v12_20231025/eez_v12.shp")

# Agricultural (CDHW) case-study outputs — needed for the joint Z500 figure.
# Set CDHW_ROOT to the CDHW_ag Results directory; the run label must match the
# 03_CDHW.py run being used in the paper (VPD branch, not the t2m variant).
CDHW_ROOT = Path("~/CDHW_ag/Results").expanduser()
CDHW_RUN  = "sm20_vpd90_sev10_95_gap3_dur3_off0"

# ERA5 store holding z500.  If a wider-domain Z500 download becomes available,
# point Z500_ZARR at it — the composite code is domain-agnostic.
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
# Load pipeline outputs (03, 04, 05)
# =================================================================
print("Loading pipeline outputs...")

rl_events_df  = pd.read_parquet(RESULTS_DIR / "DF_RL/df_rl_events.parquet")
rl_annual_df  = pd.read_parquet(RESULTS_DIR / "DF_RL/df_rl_annual.parquet")
rl_country_df = pd.read_parquet(RESULTS_DIR / "DF_RL/rl_country.parquet")

supply_events_df = pd.read_parquet(RESULTS_DIR / "DF_supply/df_supply_events.parquet")
supply_annual_df = pd.read_parquet(RESULTS_DIR / "DF_supply/df_supply_annual.parquet")

# Underlying series — needed for the event-anatomy panel, the severity
# normalisation, and the detection-threshold sensitivity sweep.
demand_df      = pd.read_parquet(RESULTS_DIR / "demand/demand_weather.parquet")
capgen_df      = pd.read_parquet(RESULTS_DIR / "DF_supply/capgen_pot_absolute.parquet")
cf_pot_rel_df  = pd.read_parquet(RESULTS_DIR / "DF_supply/cf_pot_relative.parquet")

# Per-cell supply-side climatology.  NOTE: 03_DF_supp_ID.py as committed does not
# write this file — if it is missing, FIG A panel C is skipped rather than
# failing the whole script.  Resolve the provenance before submission.
CELL_STATS_PATH = RESULTS_DIR / "DF_supply/df_supply_cell_stats.nc"
cell_stats = xr.open_dataset(CELL_STATS_PATH) if CELL_STATS_PATH.exists() else None
if cell_stats is None:
    warnings.warn(f"{CELL_STATS_PATH} not found — FIG A panel C will be skipped. "
                  "No upstream script writes this file; check provenance.")

print(f"  RL events: {len(rl_events_df)} | Supply events: {len(supply_events_df)}")

# --- Winter bookkeeping (mirrors 05_DF_RL_ID.py) ----------------------------
winter_id = pd.Series(rl_country_df.index.year, index=rl_country_df.index)
winter_id[rl_country_df.index.month.isin([11, 12])] += 1
is_winter = rl_country_df.index.month.isin(WINTER_MONTHS)
valid_winters = sorted(
    set(winter_id[rl_country_df.index.month == 11]) &
    set(winter_id[rl_country_df.index.month == 3])
)
is_valid_winter_day = is_winter & winter_id.isin(valid_winters)
N_VALID_DAYS = int(is_valid_winter_day.sum())
N_WINTERS    = len(valid_winters)
print(f"  {N_WINTERS} valid NDJFM winters ({valid_winters[0]}-{valid_winters[-1]}), "
      f"{N_VALID_DAYS} valid winter-days")

# Mean winter demand per country — the normalisation constant that makes
# severity (MW-days) comparable across systems of very different size.
mean_winter_demand = demand_df.loc[is_valid_winter_day.values, CODES].mean()
print("\nMean NDJFM demand (MW), used to normalise severity:")
print(mean_winter_demand.round(0).to_string())


# %%
# =================================================================
# Shared helpers
# =================================================================

def daily_event_mask(events_df, track, index, codes):
    """
    Reconstruct a (time x country) boolean DataFrame from an event catalogue.

    Event catalogues store only start/end dates; nearly every downstream
    diagnostic (concurrence, composites, overlap) needs the expanded daily form.
    """
    mask = pd.DataFrame(False, index=index, columns=list(codes))
    sub = events_df[events_df['track'] == track]
    for _, row in sub.iterrows():
        if row['country'] in mask.columns:
            mask.loc[row['start_date']:row['end_date'], row['country']] = True
    return mask


def event_days_set(events_df, track, country=None):
    """Set of dates belonging to any event of the given track (optionally per country)."""
    sub = events_df[events_df['track'] == track]
    if country is not None:
        sub = sub[sub['country'] == country]
    else:
        sub = sub[sub['country'] != 'EU_AGG']
    days = set()
    for _, row in sub.iterrows():
        days.update(pd.date_range(row['start_date'], row['end_date']))
    return days


def find_runs(flag, max_gap=MAX_GAP, min_duration=MIN_DURATION):
    """
    Pool True-runs separated by <= max_gap and drop runs shorter than
    min_duration.  Returns a list of (start_idx, end_idx) inclusive pairs.

    Vectorised equivalent of the loop in 03/05; used for the threshold sweep.
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


def sweep_detection(series_df, winter_mask, pct, tail):
    """
    Re-run detection at an arbitrary percentile threshold for the sensitivity
    analysis.  Returns a per-country summary (event count, total severity).

    tail : 'upper' (residual load — high is bad)
           'lower' (capacity factor — low is bad)
    """
    wm = np.asarray(winter_mask, dtype=bool)
    rows = []
    for code in series_df.columns:
        x = series_df[code].to_numpy(dtype=float)
        wv = x[wm]
        if np.all(np.isnan(wv)):
            continue
        thr = np.nanpercentile(wv, pct)
        sigma = np.nanstd(wv)
        flag = (x > thr) & wm if tail == 'upper' else (x < thr) & wm

        n_events, total_S, total_days = 0, 0.0, 0
        for s, e in find_runs(flag):
            seg = x[s:e + 1]
            shortfall = (np.clip(seg - thr, 0, None) if tail == 'upper'
                         else np.clip(thr - seg, 0, None))
            n_events += 1
            total_days += (e - s + 1)
            total_S += float(np.nansum(shortfall)) / (sigma if sigma > 0 else 1.0)

        rows.append(dict(country=code, pct=pct, n_events=n_events,
                         total_S=total_S, total_days=total_days,
                         events_per_winter=n_events / N_WINTERS))
    return pd.DataFrame(rows)


def build_country_footprints(land_shp_path, eez_shp_path):
    """Onshore ADMIN land unioned with offshore EEZ, clipped to the map extent."""
    bbox = shapely.geometry.box(MAP_EXTENT[0] - 2, MAP_EXTENT[2] - 2,
                                MAP_EXTENT[1] + 2, MAP_EXTENT[3] + 2)
    land_gdf = gpd.read_file(str(land_shp_path), bbox=bbox)
    eez_gdf = gpd.read_file(str(eez_shp_path), bbox=bbox)

    rows = []
    for name, code in COUNTRIES.items():
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        if code == 'FR':      # Exclude Corsica
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))

        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
        if code == 'GB':      # Exclude Rockall
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))
        if code == 'DK':      # Exclude Greenland / Faroe Islands
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))

        footprint = land_geom.union(eez_geom).intersection(bbox)
        footprint = footprint.simplify(0.01, preserve_topology=True)
        rows.append({'code': code, 'name': name, 'geometry': footprint})

    return gpd.GeoDataFrame(rows, crs=land_gdf.crs)


footprints = build_country_footprints(COUNTRIES_SHP, EEZ_SHP)
print("\nCountry footprints built (onshore + EEZ offshore, clipped to map extent).")


def study_domain_mask(template_da):
    """
    Boolean (lat, lon) mask of cells inside ANY study-country footprint.

    Used to clip the per-cell climatology: without it the colour scale is set by
    open-Atlantic cells where no capacity exists or could ever be installed, and
    the map shows data over countries (IT/AT/PL) that are in no country mask.
    """
    union = footprints.union_all()
    lons, lats = np.meshgrid(template_da.longitude.values, template_da.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())
    inside = shapely.contains(union, pts).reshape(lats.shape)
    return xr.DataArray(inside,
                        coords={'latitude': template_da.latitude,
                                'longitude': template_da.longitude},
                        dims=['latitude', 'longitude'])


def plot_choropleth_panel(ax, values_by_code, cmap, norm, title,
                          fmt="{:.3f}", counts_by_code=None):
    """Fill each country footprint with a single scalar value."""
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=LAND_COLOR, zorder=0)

    gdf = footprints.copy()
    gdf['value'] = gdf['code'].map(values_by_code)
    gdf.plot(column='value', ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
             edgecolor='black', linewidth=0.6, zorder=2,
             missing_kwds={'color': '#d9d9d9', 'hatch': '///'})

    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
                   edgecolor='dimgray', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6,
                   edgecolor='black', zorder=3)

    for _, row in gdf.iterrows():
        c = row['geometry'].centroid
        if row['value'] is None or (isinstance(row['value'], float) and np.isnan(row['value'])):
            label = f"{row['code']}\n(none)"
        else:
            label = f"{row['code']}\n{fmt.format(row['value'])}"
            if counts_by_code is not None:
                label += f"\nn={int(counts_by_code.get(row['code'], 0))}"
        ax.text(c.x, c.y, label, transform=ccrs.PlateCarree(), ha='center', va='center',
                fontsize=7.5, fontweight='bold', zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_title(title, fontsize=11, fontweight='bold')


def style_z500_ax(ax):
    ax.set_extent(Z500_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5,
                   edgecolor='0.35', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7,
                   edgecolor='black', zorder=3)


# --- Daily event masks (built once, reused everywhere) ----------------------
rl_daily_mask = {t: daily_event_mask(rl_events_df, t, rl_country_df.index, CODES)
                 for t in RL_TRACKS}
supply_daily_mask = {t: daily_event_mask(supply_events_df, t, rl_country_df.index, CODES)
                     for t in SUPPLY_TRACKS}

# Cross-country concurrence: how many countries are in event on each day.
concurrence = {t: rl_daily_mask[t].sum(axis=1) for t in RL_TRACKS}


# %%
# =================================================================
# FIGURE A — Spatial climatology
#   A1 T2 event frequency (T1 omitted: constant by construction)
#   A2 mean event duration (the weather-controlled metric)
#   A3 per-cell supply-side climatology, masked to the study domain
# =================================================================
print("\nFIG A — spatial climatology...")

rl_country_events = rl_events_df[rl_events_df['country'] != 'EU_AGG']

freq_t2 = ((rl_country_events[rl_country_events['track'] == 'T2']
            .groupby('country')['duration'].sum() / N_VALID_DAYS)
           .reindex(CODES, fill_value=0.0).to_dict())
n_t2 = (rl_country_events[rl_country_events['track'] == 'T2']
        .groupby('country').size().reindex(CODES, fill_value=0).to_dict())

fig = plt.figure(figsize=(16.5, 6.2))
ax1 = fig.add_subplot(1, 3, 1, projection=PROJ)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection=PROJ)

# --- A1: T2 frequency choropleth ---
norm_freq = Normalize(vmin=0, vmax=max(freq_t2.values()))
plot_choropleth_panel(ax1, freq_t2, plt.get_cmap('YlOrRd'), norm_freq,
                      "A — Supply-deficit frequency (T2)", counts_by_code=n_t2)
sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm_freq)
cb = fig.colorbar(sm, ax=ax1, orientation='horizontal', fraction=0.05, pad=0.04)
cb.set_label('Fraction of NDJFM days in deficit', fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)

# --- A2: mean duration, both tracks ---
# Duration is the metric that is genuinely weather-controlled: it is ~flat
# across countries even where T2 frequency spans two orders of magnitude.
x = np.arange(len(CODES))
width = 0.35
for i, track in enumerate(RL_TRACKS):
    sub = (rl_country_events[rl_country_events['track'] == track]
           .groupby('country')['duration'].mean().reindex(CODES))
    ax2.bar(x + (i - 0.5) * width, sub.values, width, label=RL_TRACK_LABEL[track],
            color=TRACK_COLORS[track], edgecolor='k', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(CODES)
ax2.set_ylabel('Mean event duration (days)')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.set_title("B — Mean event duration", fontsize=11, fontweight='bold')

# --- A3: per-cell climatology, masked to study footprints ---
if cell_stats is not None:
    var = 'freq' if 'freq' in cell_stats else list(cell_stats.data_vars)[0]
    da = cell_stats[var]
    da = da.where(study_domain_mask(da))     # clip to land + EEZ of study countries

    ax3.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax3.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=OCEAN_COLOR, zorder=0)
    ax3.add_feature(cfeature.LAND.with_scale('10m'), facecolor=LAND_COLOR, zorder=0)
    cf = ax3.contourf(da.longitude, da.latitude, da, levels=20, cmap='viridis',
                      transform=ccrs.PlateCarree(), zorder=1)
    ax3.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5,
                    edgecolor='white', zorder=3)
    ax3.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6,
                    edgecolor='black', zorder=3)
    cb3 = fig.colorbar(cf, ax=ax3, orientation='horizontal', fraction=0.05, pad=0.04)
    # NB: previous label read "relative frequency" but the values (1.4-2.5) are
    # events per winter — relabelled to match the quantity actually plotted.
    cb3.set_label('Supply-side events per winter (grid cell)', fontsize=8.5)
    cb3.ax.tick_params(labelsize=7.5)
    ax3.set_title("C — Grid-cell climatology", fontsize=11, fontweight='bold')
else:
    ax3.remove()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    ax3.text(0.5, 0.5, 'df_supply_cell_stats.nc\nnot found', ha='center', va='center',
             fontsize=10, color='0.4')

fig.suptitle(f"Residual-load dunkelflaute climatology, {valid_winters[0]}–{valid_winters[-1]}",
             fontsize=14, y=1.00)
plt.tight_layout()
save_fig(fig, "figA_climatology")
plt.show()
plt.close()


# %%
# =================================================================
# FIGURE B — Event anatomy and detection validation
#   B1 winter 2017/18: demand, generation, residual load with events shaded
#   B2 RL vs supply-only overlap (Jaccard) — corrected supply track name
#   B3 within-season timing of event days
# =================================================================
print("\nFIG B — event anatomy and validation...")

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.22)
ax_b1 = fig.add_subplot(gs[0, :])
ax_b2 = fig.add_subplot(gs[1, 0])
ax_b3 = fig.add_subplot(gs[1, 1])

# --- B1: the anatomy panel ---
# Nothing in the previous figure set showed what a dunkelflaute actually looks
# like; this is the direct analogue of the CDHW 2018 fingerprint panel.
ANATOMY_COUNTRY = 'DE'
# NB: is_winter is a plain ndarray (Index.isin returns ndarray), so no .values here.
w_mask = (winter_id == SPOTLIGHT_WINTER).values & np.asarray(is_winter)
w_idx = rl_country_df.index[w_mask]

dem = demand_df.loc[w_idx, ANATOMY_COUNTRY]
gen = capgen_df.loc[w_idx, ANATOMY_COUNTRY]
rl = rl_country_df.loc[w_idx, ANATOMY_COUNTRY]

ax_b1.plot(w_idx, dem / 1e3, color='#333333', lw=1.4, label='Weather-driven demand')
ax_b1.plot(w_idx, gen / 1e3, color='#2E8B57', lw=1.4, label='Wind + solar generation potential')
ax_b1.plot(w_idx, rl / 1e3, color='#C44E52', lw=1.6, label='Residual load (demand − generation)')
ax_b1.axhline(0, color='0.5', lw=0.8, ls='-')

thr_t1 = np.percentile(rl_country_df.loc[is_valid_winter_day.values, ANATOMY_COUNTRY],
                       THRESHOLD_PCT)
ax_b1.axhline(thr_t1 / 1e3, color=TRACK_COLORS['T1'], ls='--', lw=1.2,
              label=f'T1 threshold ({THRESHOLD_PCT}th pct)')

# Capture the limits BEFORE shading: fill_between would otherwise expand them
# on the first pass and the second band would be drawn to the wrong extent.
y_lo, y_hi = ax_b1.get_ylim()
for track in RL_TRACKS:
    m = rl_daily_mask[track].loc[w_idx, ANATOMY_COUNTRY]
    ax_b1.fill_between(w_idx, y_lo, y_hi, where=m.values,
                       color=TRACK_COLORS[track], alpha=0.15, linewidth=0,
                       label=f'{track} event')
ax_b1.set_ylim(y_lo, y_hi)

ax_b1.set_ylabel('GW')
ax_b1.set_title(f"A — Anatomy of a dunkelflaute winter: {ANATOMY_COUNTRY}, "
                f"NDJFM {SPOTLIGHT_WINTER - 1}/{str(SPOTLIGHT_WINTER)[-2:]}",
                fontsize=11, fontweight='bold', loc='left')
ax_b1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_b1.grid(alpha=0.3, linestyle='--')
handles, labels = ax_b1.get_legend_handles_labels()
ax_b1.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(),
             fontsize=8, ncol=3, loc='upper right')

# --- B2: RL vs supply-only overlap ---
# 05_DF_RL_ID.py filters supply_events_df on track == 'T1', which never matches
# (supply tracks are named differently), so its Jaccard values are all 0.
# Recomputed here against the headline supply track.
overlap = []
for code in CODES:
    sup_days = event_days_set(supply_events_df, SUPPLY_HEADLINE, code)
    for track in RL_TRACKS:
        rl_days = event_days_set(rl_events_df, track, code)
        inter = len(sup_days & rl_days)
        union = len(sup_days | rl_days)
        overlap.append({'country': code, 'track': track,
                        'jaccard': inter / union if union else np.nan,
                        'n_supply': len(sup_days), 'n_rl': len(rl_days)})
overlap_df = pd.DataFrame(overlap)

for i, track in enumerate(RL_TRACKS):
    sub = overlap_df[overlap_df['track'] == track].set_index('country').reindex(CODES)
    ax_b2.bar(x + (i - 0.5) * width, sub['jaccard'].values, width,
              label=f'{track} vs supply-only', color=TRACK_COLORS[track],
              edgecolor='k', linewidth=0.5)
ax_b2.set_xticks(x)
ax_b2.set_xticklabels(CODES)
ax_b2.set_ylabel('Jaccard index (event-day overlap)')
ax_b2.set_ylim(0, 1)
ax_b2.grid(axis='y', linestyle='--', alpha=0.5)
ax_b2.legend(fontsize=8)
ax_b2.set_title("B — Residual-load vs supply-only detection\n"
                "(low overlap ⇒ the two definitions find different events)",
                fontsize=10, fontweight='bold', loc='left')

print("\nRL vs supply-only overlap (Jaccard):")
print(overlap_df.pivot(index='country', columns='track', values='jaccard').round(3).to_string())

# --- B3: within-season timing ---
# Van der Wiel et al. (2019) find low-production and high-shortfall events peak
# at different times; this is the cheap direct test of that on our catalogue.
month_order = [11, 12, 1, 2, 3]
month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']

timing = {}
for track in RL_TRACKS:
    days = pd.DatetimeIndex(sorted(event_days_set(rl_events_df, track)))
    counts = pd.Series(days.month).value_counts().reindex(month_order, fill_value=0)
    timing[RL_TRACK_LABEL[track]] = counts / counts.sum()
sup_days_all = pd.DatetimeIndex(sorted(event_days_set(supply_events_df, SUPPLY_HEADLINE)))
sup_counts = pd.Series(sup_days_all.month).value_counts().reindex(month_order, fill_value=0)
timing['Supply-only (Hu ratio)'] = sup_counts / sup_counts.sum()

xm = np.arange(len(month_order))
wm_ = 0.26
colors_b3 = [TRACK_COLORS['T1'], TRACK_COLORS['T2'], '#55A868']
for i, (label, series) in enumerate(timing.items()):
    ax_b3.bar(xm + (i - 1) * wm_, series.values, wm_, label=label,
              color=colors_b3[i], edgecolor='k', linewidth=0.5)
ax_b3.set_xticks(xm)
ax_b3.set_xticklabels(month_names)
ax_b3.set_ylabel('Fraction of event-days')
ax_b3.grid(axis='y', linestyle='--', alpha=0.5)
ax_b3.legend(fontsize=7.5)
ax_b3.set_title("C — Within-season timing of event days", fontsize=10,
                fontweight='bold', loc='left')

save_fig(fig, "figB_anatomy_validation")
plt.show()
plt.close()


# %%
# =================================================================
# FIGURE C — Robustness
#   C1 capacity-assumption tracks (relative and absolute on SEPARATE axes)
#   C2 detection-threshold sensitivity
#   C3 cross-country concurrence (replaces the copper-plate comparison)
# =================================================================
print("\nFIG C — robustness...")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
ax_c1, ax_c2, ax_c3 = axes

# --- C1: capacity assumptions ---
# The two relative tracks carry sigma-normalised severity; the absolute track
# carries MW-based severity.  Plotting them on one axis (as previously) makes
# the absolute track look ~20x more severe when it is simply in other units.
supply_agg = (supply_annual_df.groupby(['country', 'track'])
              .agg(n_events=('n_events', 'sum')).reset_index())
supply_agg['events_per_winter'] = supply_agg['n_events'] / N_WINTERS

for i, track in enumerate(SUPPLY_TRACKS):
    sub = supply_agg[supply_agg['track'] == track].set_index('country').reindex(CODES)
    ax_c1.bar(x + (i - 1) * width * 0.7, sub['events_per_winter'].values, width * 0.7,
              label=SUPPLY_TRACK_LABEL[track], color=TRACK_PALETTE[track],
              edgecolor='k', linewidth=0.4)
ax_c1.set_xticks(x)
ax_c1.set_xticklabels(CODES)
ax_c1.set_ylabel('Events per winter')
ax_c1.grid(axis='y', linestyle='--', alpha=0.5)
ax_c1.legend(fontsize=7.5, framealpha=0.9)
ax_c1.set_title("A — Sensitivity to capacity assumptions\n"
                "(absolute track's event count is set by the 15% floor, not weather)",
                fontsize=10, fontweight='bold', loc='left')

# --- C2: detection-threshold sensitivity ---
# SEVERE_PCT = 5 is defined in 03_DF_supp_ID.py but never used; the RL side has
# no sensitivity check at all.  Both are swept here.
rl_sweep = pd.concat([sweep_detection(rl_country_df[CODES], is_valid_winter_day.values,
                                      pct, 'upper')
                      for pct in RL_PCT_SWEEP], ignore_index=True)
supply_sweep = pd.concat([sweep_detection(cf_pot_rel_df[CODES], is_valid_winter_day.values,
                                          pct, 'lower')
                          for pct in SUPPLY_PCT_SWEEP], ignore_index=True)

sweep_colors = plt.get_cmap('Blues')(np.linspace(0.45, 0.85, len(RL_PCT_SWEEP)))
for i, pct in enumerate(RL_PCT_SWEEP):
    sub = rl_sweep[rl_sweep['pct'] == pct].set_index('country').reindex(CODES)
    ax_c2.bar(x + (i - 1) * width * 0.7, sub['events_per_winter'].values, width * 0.7,
              label=f'RL {pct}th pct', color=sweep_colors[i], edgecolor='k', linewidth=0.4)
ax_c2.set_xticks(x)
ax_c2.set_xticklabels(CODES)
ax_c2.set_ylabel('Events per winter (RL track)')
ax_c2.grid(axis='y', linestyle='--', alpha=0.5)
ax_c2.legend(fontsize=7.5, title='Threshold', title_fontsize=8)
ax_c2.set_title("B — Sensitivity to detection threshold\n"
                "(does the headline ranking survive a stricter tail?)",
                fontsize=10, fontweight='bold', loc='left')

print("\nSupply-side threshold sweep (events per winter):")
print(supply_sweep.pivot(index='country', columns='pct',
                         values='events_per_winter').round(2).to_string())
print("\nRL threshold sweep (events per winter):")
print(rl_sweep.pivot(index='country', columns='pct',
                     values='events_per_winter').round(2).to_string())

# --- C3: cross-country concurrence ---
# This is the real balancing-effect diagnostic.  The copper-plate comparison it
# replaces was tautological: the EU aggregate derives its own 90th percentile,
# so its frequency is pinned near the national values by construction.
for track in RL_TRACKS:
    conc = concurrence[track][is_valid_winter_day.values]
    conc = conc[conc > 0]
    counts = conc.value_counts().reindex(range(1, len(CODES) + 1), fill_value=0)
    ax_c3.plot(counts.index, counts.values / counts.sum(), marker='o', lw=1.8,
               color=TRACK_COLORS[track], label=RL_TRACK_LABEL[track])
ax_c3.set_xticks(range(1, len(CODES) + 1))
ax_c3.set_xlabel('Number of countries simultaneously in event')
ax_c3.set_ylabel('Fraction of event-days')
ax_c3.grid(alpha=0.3, linestyle='--')
ax_c3.legend(fontsize=7.5)
ax_c3.set_title("C — Cross-country concurrence\n"
                "(geographical balancing: how often is the shortfall continental?)",
                fontsize=10, fontweight='bold', loc='left')

for track in RL_TRACKS:
    conc = concurrence[track][is_valid_winter_day.values]
    frac_all = float((conc == len(CODES)).sum()) / max(int((conc > 0).sum()), 1)
    print(f"  {track}: {frac_all:.1%} of event-days affect all {len(CODES)} countries")

fig.suptitle("Robustness of the dunkelflaute detection", fontsize=13, y=1.02)
plt.tight_layout()
save_fig(fig, "figC_robustness")
plt.show()
plt.close()


# %%
# =================================================================
# Z500 — load once, season-subset into memory
# =================================================================
# CAVEAT (unchanged, and now more visible): the ERA5 domain downloaded for this
# subproject is the regional capacity-factor box (N62 W-14 S42 E18).  The
# composites below show a monotonic NW-SE gradient maximised at the domain
# corner, which is the signature of an anticyclone centred OUTSIDE the window
# (Norwegian Sea / Iceland).  Z500 is a single level and only daily means are
# needed, so re-downloading it on e.g. [80N, 60W, 30N, 40E] is cheap relative to
# the rest of the pipeline and would let these panels show the blocking centre
# rather than its flank.  Point Z500_ZARR at the wider store when available.
print("\nLoading Z500 and subsetting by season...")

ds_z = xr.open_zarr(Z500_ZARR, consolidated=True)
z = ds_z['z']
if 'level' in z.dims:
    z = z.isel(level=0)

z_daily = z.coarsen(time=24, boundary='trim', coord_func='min').mean()
times_z = pd.DatetimeIndex(z_daily.time.values)

# Loading one season at a time keeps this in memory comfortably and makes the
# many subsequent composites (7 countries + joint panels) essentially free.
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


# --- Energy: concurrence-filtered event days -------------------------------
# There is a trade-off here: a stricter concurrence requirement sharpens the
# composite but shrinks n.  The full distribution is printed so the threshold
# can be tuned against the actual data rather than guessed.
conc_t1 = concurrence['T1'][is_valid_winter_day.values]
conc_table = conc_t1[conc_t1 > 0].value_counts().sort_index()
print("\nDays by number of countries simultaneously in T1 event:")
for k, v in conc_table.items():
    flag = "  <-- composite threshold" if k == MIN_CONCURRENT_COUNTRIES else ""
    print(f"  {k} countries: {v:5d} days"
          f"  (>={k}: {int(conc_table[conc_table.index >= k].sum()):5d}){flag}")

rl_concurrent_days = set(conc_t1.index[conc_t1 >= MIN_CONCURRENT_COUNTRIES])
rl_any_days = event_days_set(rl_events_df, 'T1')
print(f"\nRL T1 event-days: {len(rl_any_days)} any-country → "
      f"{len(rl_concurrent_days)} with ≥{MIN_CONCURRENT_COUNTRIES} countries concurrent")

MIN_COMPOSITE_DAYS = 100
if len(rl_concurrent_days) < MIN_COMPOSITE_DAYS:
    warnings.warn(
        f"Only {len(rl_concurrent_days)} days pass the ≥{MIN_CONCURRENT_COUNTRIES}-country "
        f"filter — the composite will be noisy. Lower MIN_CONCURRENT_COUNTRIES using the "
        f"distribution printed above (aim for >{MIN_COMPOSITE_DAYS} days while still "
        f"excluding the bulk of single-country days)."
    )

# --- Agriculture: area-fraction-filtered event days ------------------------
cdhw_mask_path = CDHW_ROOT / "CDHW_results" / CDHW_RUN / "compound_mask_idx.nc"
cdhw_available = cdhw_mask_path.exists()

if cdhw_available:
    cdhw_mask = xr.open_dataarray(cdhw_mask_path)
    # Land-cell count from the SSI grid (the compound mask is bool; ocean cells
    # are simply never True, so they cannot be counted from the mask itself).
    ssi = xr.open_zarr(CDHW_ROOT / "SSI.zarr", consolidated=True)
    ssi = ssi[list(ssi.data_vars)[0]]
    n_land_cells = int(np.isfinite(ssi.isel(time=0).values).sum())

    area_frac = (cdhw_mask.sum(['latitude', 'longitude']) / n_land_cells).compute()
    area_thr = float(np.percentile(area_frac.values, CDHW_AREA_PCTL))
    cdhw_times = pd.DatetimeIndex(cdhw_mask.time.values)
    cdhw_concurrent_days = set(cdhw_times[area_frac.values >= area_thr])
    cdhw_any_days = set(cdhw_times[area_frac.values > 0])

    print(f"CDHW event-days: {len(cdhw_any_days)} any-cell → "
          f"{len(cdhw_concurrent_days)} in top {100 - CDHW_AREA_PCTL}% by area "
          f"(threshold = {area_thr:.1%} of domain)")
else:
    warnings.warn(f"CDHW mask not found at {cdhw_mask_path} — "
                  "FIG D will show the energy panels only. Set CDHW_ROOT/CDHW_RUN.")


# %%
# =================================================================
# FIGURE D — Joint Z500 composite, concurrence-filtered
# The empirical test of the 'one driver, two receptors' claim.
# Unfiltered composites are shown alongside as a diagnostic: they demonstrate
# WHY the filter is necessary rather than leaving the reader to wonder.
# =================================================================
print("\nFIG D — joint Z500 composite (concurrence-filtered)...")

panels = []
if cdhw_available:
    a_unf, n_a_unf = composite(z_jja, z_jja_clim, cdhw_any_days)
    a_flt, n_a_flt = composite(z_jja, z_jja_clim, cdhw_concurrent_days)
    panels += [("CDHW — all event days (JJA)", a_unf, n_a_unf, len(z_jja.time)),
               (f"CDHW — top {100 - CDHW_AREA_PCTL}% by area affected", a_flt, n_a_flt,
                len(z_jja.time))]

e_unf, n_e_unf = composite(z_win, z_win_clim, rl_any_days)
e_flt, n_e_flt = composite(z_win, z_win_clim, rl_concurrent_days)
panels += [("Dunkelflaute RL — all event days (NDJFM)", e_unf, n_e_unf, len(z_win.time)),
           (f"Dunkelflaute RL — ≥{MIN_CONCURRENT_COUNTRIES} countries concurrent",
            e_flt, n_e_flt, len(z_win.time))]

# A shared norm makes the two hazards visually comparable; report the peak of
# each so the caption can state the ratio rather than relying on colour alone.
anom_max = max(float(np.nanmax(np.abs(p[1].values))) for p in panels if p[1] is not None)
norm_anom = TwoSlopeNorm(vmin=-anom_max, vcenter=0, vmax=anom_max)

ncol = 2
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 6.2 * nrow),
                         subplot_kw={'projection': PROJ})
axes = np.atleast_1d(axes).ravel()

for ax, (title, anom, n, n_season) in zip(axes, panels):
    style_z500_ax(ax)
    if anom is None:
        ax.set_title(f"{title}\n(no days)", fontsize=10)
        continue
    cf = ax.contourf(anom.longitude, anom.latitude, anom, levels=21, cmap='RdBu_r',
                     norm=norm_anom, transform=ccrs.PlateCarree(), zorder=1)
    peak = float(np.nanmax(np.abs(anom.values)))
    ax.set_title(f"{title}\nn={n} days ({n / n_season:.0%} of season) | peak |Z'| = {peak:.0f} m",
                 fontsize=10)

for ax in axes[len(panels):]:
    ax.axis('off')

sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm_anom)
cb = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal',
                  fraction=0.04, pad=0.05, shrink=0.6)
cb.set_label("Z500 anomaly (m) relative to own-season climatology")

fig.suptitle("Shared anticyclonic blocking signature: summer CDHW vs. winter dunkelflaute\n"
             "left column: all event days (diluted) — right column: concurrence-filtered",
             fontsize=13, y=1.01)
save_fig(fig, "figD_z500_joint_composite")
plt.show()
plt.close()

print("\nComposite peak |Z500 anomaly| (m):")
for title, anom, n, _ in panels:
    if anom is not None:
        print(f"  {title}: {float(np.nanmax(np.abs(anom.values))):.1f} (n={n})")


# %%
# =================================================================
# FIGURE E — Per-country Z500 composites
# Pooling across countries blurs the ridge: the blocking centre sits in a
# different place for a Danish event than for a French one.  One panel per
# country, shared norm, so the geographic shift is directly readable.
# =================================================================
print("\nFIG E — per-country Z500 composites...")

country_anoms = {}
for code in CODES:
    days = event_days_set(rl_events_df, 'T1', code)
    anom, n = composite(z_win, z_win_clim, days)
    if anom is not None:
        country_anoms[code] = (anom, n)

cmax = max(float(np.nanmax(np.abs(a.values))) for a, _ in country_anoms.values())
norm_c = TwoSlopeNorm(vmin=-cmax, vcenter=0, vmax=cmax)

ncol = 4
nrow = int(np.ceil(len(country_anoms) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.6 * nrow),
                         subplot_kw={'projection': PROJ})
axes = np.atleast_1d(axes).ravel()

for ax, (code, (anom, n)) in zip(axes, country_anoms.items()):
    style_z500_ax(ax)
    ax.contourf(anom.longitude, anom.latitude, anom, levels=21, cmap='RdBu_r',
                norm=norm_c, transform=ccrs.PlateCarree(), zorder=1)
    # Outline the country whose events are being composited, for orientation.
    geom = footprints.loc[footprints['code'] == code, 'geometry']
    ax.add_geometries(geom, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='black', linewidth=1.1, zorder=4)
    peak = float(np.nanmax(np.abs(anom.values)))
    ax.set_title(f"{code} — n={n} days | peak {peak:.0f} m", fontsize=10, fontweight='bold')

for ax in axes[len(country_anoms):]:
    ax.axis('off')

sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm_c)
cb = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal',
                  fraction=0.04, pad=0.04, shrink=0.5)
cb.set_label("Z500 anomaly (m) relative to NDJFM climatology")

fig.suptitle("Per-country Z500 composites over national dunkelflaute event days (T1)",
             fontsize=13, y=1.00)
save_fig(fig, "figE_z500_per_country")
plt.show()
plt.close()


# %%
# =================================================================
# SI 1 — Trend forest plot
# Replaces the two 8-panel scatter grids.  Those showed tau ~ 0 and p > 0.3
# everywhere across 16 panels; the same information fits in one.  Theil-Sen is
# dropped for occurrence: on integer counts the median slope is exactly 0 in
# almost every country, which is a discretisation artefact, not a trend.
# =================================================================
print("\nSI 1 — trend forest plot...")


def tau_with_ci(years, values, n_boot=2000, seed=0):
    """Kendall tau plus a bootstrap CI (resampling winters)."""
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
for metric, label in [('n_events', 'Occurrence'), ('total_S', 'Severity')]:
    for code in CODES:
        ts = (rl_annual_df[(rl_annual_df['country'] == code) &
                           (rl_annual_df['track'] == 'T1')].sort_values('winter'))
        tau, p, lo, hi = tau_with_ci(ts['winter'].values, ts[metric].values)
        trend_rows.append({'country': code, 'metric': label,
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
ax.set_xlabel("Kendall's τ (trend over winters, T1)")
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.legend(fontsize=9)
ax.set_title(f"No detectable trend in dunkelflaute hazard, "
             f"{valid_winters[0]}–{valid_winters[-1]}\n"
             "every CI spans zero — contrast with the intensifying CDHW trends",
             fontsize=11, fontweight='bold')
plt.tight_layout()
save_fig(fig, "figSI1_trend_forest")
plt.show()
plt.close()

print("\nTrend summary (T1):")
print(trend_df.round(3).to_string(index=False))


# %%
# =================================================================
# SI 2 — Duration-severity joint distribution
# Kittel & Schill's central critique is that single-threshold characterisations
# are unreliable; the joint distribution is the direct response.
# =================================================================
print("\nSI 2 — duration-severity joint distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
colors_dur = plt.cm.tab10(np.linspace(0, 1, len(CODES)))

for ax, track in zip(axes, RL_TRACKS):
    for code, color in zip(CODES, colors_dur):
        dat = rl_events_df[(rl_events_df['country'] == code) &
                           (rl_events_df['track'] == track)]
        if dat.empty:
            continue
        # Normalised by mean winter demand so systems of different size compare.
        sev_norm = dat['severity_S'] / mean_winter_demand[code]
        ax.scatter(dat['duration'], sev_norm, alpha=0.6, s=28, label=code,
                   color=color, edgecolors='k', linewidths=0.3)
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('Severity (deficit-day equivalent)')
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(RL_TRACK_LABEL[track], fontsize=10, fontweight='bold')

axes[0].legend(fontsize=8, ncol=2)
fig.suptitle("Duration-severity joint distribution of residual-load events", fontsize=13)
plt.tight_layout()
save_fig(fig, "figSI2_duration_severity")
plt.show()
plt.close()


# %%
# =================================================================
# SI 3 — Severity choropleth, demand-normalised
# Fixes on the previous version: (a) severity divided by mean winter demand so
# GB's larger system does not masquerade as greater severity; (b) a SHARED norm
# across panels, which the previous independent norms made impossible;
# (c) event counts printed, and countries with fewer than MIN_EVENTS_SHOWN
# events greyed out rather than plotted with the same visual weight.
# =================================================================
print("\nSI 3 — demand-normalised severity choropleth...")

MIN_EVENTS_SHOWN = 5

sev_by_track, n_by_track = {}, {}
for track in RL_TRACKS:
    sub = rl_country_events[rl_country_events['track'] == track]
    mean_sev = sub.groupby('country')['severity_S'].mean().reindex(CODES)
    counts = sub.groupby('country').size().reindex(CODES, fill_value=0)
    norm_sev = mean_sev / mean_winter_demand.reindex(CODES)
    norm_sev[counts < MIN_EVENTS_SHOWN] = np.nan     # too few events to be stable
    sev_by_track[track] = norm_sev.to_dict()
    n_by_track[track] = counts.to_dict()

all_vals = [v for d in sev_by_track.values() for v in d.values()
            if v is not None and np.isfinite(v)]
norm_shared = Normalize(vmin=0, vmax=max(all_vals) if all_vals else 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': PROJ})
for ax, track in zip(axes, RL_TRACKS):
    plot_choropleth_panel(ax, sev_by_track[track], 'YlOrRd', norm_shared,
                          RL_TRACK_LABEL[track], fmt="{:.2f}",
                          counts_by_code=n_by_track[track])

sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm_shared)
cb = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal',
                  fraction=0.05, pad=0.05, shrink=0.6)
cb.set_label('Mean event severity (deficit-day equivalent, normalised by mean winter demand)')

fig.suptitle("Mean event severity by country — residual-load pipeline\n"
             f"hatched = fewer than {MIN_EVENTS_SHOWN} events (estimate unstable)",
             fontsize=13, y=1.00)
save_fig(fig, "figSI3_severity_choropleth")
plt.show()
plt.close()

print(f"\nAll figures written to: {OUT_DIR}")
# %%