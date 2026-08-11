#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_results.py — Dunkelflaute Results Visualization
===================================================
Publication-oriented figures for the residual-load (RL) and supply-side
dunkelflaute detection pipelines (Otero et al. 2022 methodology).

Figures produced
-----------------
 1. Country choropleth — RL event frequency (T1 relative / T2 absolute)
 2. Country choropleth — RL mean event severity (T1 / T2)
 3. Per-country climatology bars — frequency, mean duration, mean severity (RL)
 4. Cross-track comparison — supply-side IRENA vs. Hu-ratio vs. Hu-MW
 5. Annual trends — occurrence & severity per country, Mann-Kendall tau +
    Theil-Sen slope (T1)
 6. Copper-plate (EU_AGG) vs. national aggregation — spatial smoothing effect
 7. Per-cell spatial climatology (supply-side 50/50 wind-solar mix)
 8. Z500 anomaly composite over RL event dates — shared anticyclonic driver

Literature: Otero et al. 2022; Bloomfield 2021; Biewald et al. 2025;
Van der Wiel et al. 2019; Hu et al. 2023.

@author: afer
"""
#%%
# Imports

from pathlib import Path
import os

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from scipy.stats import kendalltau, theilslopes

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#%%
# Config and params

WINTER_MONTHS = [11, 12, 1, 2, 3]   # NDJFM
THRESHOLD_PCT = 90                  # RL Track 1 (relative) percentile
OCEAN_COLOR   = '#C6E2F5'
LAND_COLOR    = '#F5F5F2'
MAP_EXTENT    = [-11.5, 16.5, 41.5, 59.5]   # [W, E, S, N]
PROJ          = ccrs.LambertConformal(central_longitude=3, central_latitude=50)

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

RL_TRACKS     = ['T1', 'T2']
RL_TRACK_LABEL = {'T1': f'T1 — Relative ({THRESHOLD_PCT}th pct RL)',
                   'T2': 'T2 — Absolute (RL > 0, unserved energy)'}
SUPPLY_TRACKS = ['IRENA_Relative_10pct', 'MaxPot_Relative_10pct', 'MaxPot_Absolute_15pct_floor']
SUPPLY_TRACK_LABEL = {'IRENA_Relative_10pct': 'IRENA current mix',
                       'MaxPot_Relative_10pct': 'Hu max potential (ratio)',
                       'MaxPot_Absolute_15pct_floor': 'Hu max potential (abs. MW)'}

# Paths
RESULTS_DIR   = Path("./../Results")
OUT_DIR       = RESULTS_DIR / "figures"
COUNTRIES_SHP = Path("~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp").expanduser()
EEZ_SHP       = Path("./../Data/EEZ/World_EEZ_v12_20231025/eez_v12.shp")

OUT_DIR.mkdir(parents=True, exist_ok=True)

#%%
# =================================================================
# Load results from the upstream pipeline (03, 04, 05)
# =================================================================
print("Loading pipeline outputs...")

rl_events_df   = pd.read_parquet(RESULTS_DIR / "DF_RL/df_rl_events.parquet")
rl_annual_df   = pd.read_parquet(RESULTS_DIR / "DF_RL/df_rl_annual.parquet")
rl_country_df  = pd.read_parquet(RESULTS_DIR / "DF_RL/rl_country.parquet")

supply_events_df = pd.read_parquet(RESULTS_DIR / "DF_supply/df_supply_events.parquet")
supply_annual_df = pd.read_parquet(RESULTS_DIR / "DF_supply/df_supply_annual.parquet")
cell_stats        = xr.open_dataset(RESULTS_DIR / "DF_supply/df_supply_cell_stats.nc")

print(f"  RL events: {len(rl_events_df)} | Supply events: {len(supply_events_df)}")

# Recompute the NDJFM valid-day count backing the RL pipeline (mirrors 05_DF_RL_ID.py)
# so that event-day counts can be normalised into a relative frequency.
winter_id = pd.Series(rl_country_df.index.year, index=rl_country_df.index)
winter_id[rl_country_df.index.month.isin([11, 12])] += 1
is_winter = rl_country_df.index.month.isin(WINTER_MONTHS)
valid_winters = sorted(
    set(winter_id[rl_country_df.index.month == 11]) &
    set(winter_id[rl_country_df.index.month == 3])
)
is_valid_winter_day = is_winter & winter_id.isin(valid_winters)
N_VALID_DAYS = int(is_valid_winter_day.sum())
N_WINTERS = len(valid_winters)
print(f"  {N_WINTERS} valid NDJFM winters ({valid_winters[0]}-{valid_winters[-1]}), "
      f"{N_VALID_DAYS} valid winter-days")

#%%
# =================================================================
# Country footprints (onshore ADMIN land + offshore EEZ), for choropleths
# Mirrors the mask logic in 03_DF_supp_ID.py so map extents match the
# domain actually used to derive the underlying generation time series.
# =================================================================

def build_country_footprints(land_shp_path, eez_shp_path):
    # SOVEREIGN1 in the EEZ dataset spans overseas territories scattered across
    # the globe (e.g. France -> French Polynesia); clip to the regional bbox
    # *before* filtering/unioning so shapely isn't unioning geometry oceans away.
    bbox = shapely.geometry.box(MAP_EXTENT[0] - 2, MAP_EXTENT[2] - 2, MAP_EXTENT[1] + 2, MAP_EXTENT[3] + 2)

    land_gdf = gpd.read_file(str(land_shp_path), bbox=bbox)
    eez_gdf = gpd.read_file(str(eez_shp_path), bbox=bbox)

    rows = []
    for name, code in COUNTRIES.items():
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        if code == 'FR':  # Exclude Corsica
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))

        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
        if code == 'GB':  # Exclude Rockall
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))
        if code == 'DK':  # Exclude Greenland / Faroe Islands
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))

        footprint = land_geom.union(eez_geom).intersection(bbox)
        # EEZ boundaries carry full coastline detail (~20k vertices for the UK) —
        # simplify for map rendering; negligible visual cost at country-choropleth scale.
        footprint = footprint.simplify(0.01, preserve_topology=True)
        rows.append({'code': code, 'name': name, 'geometry': footprint})

    return gpd.GeoDataFrame(rows, crs=land_gdf.crs)

footprints = build_country_footprints(COUNTRIES_SHP, EEZ_SHP)
print("Country footprints built (onshore + EEZ offshore, clipped to map extent).")

#%%
# =================================================================
# Choropleth plotting helper
# =================================================================

def plot_choropleth_panel(ax, values_by_code, cmap, norm, title):
    """Fills each country footprint with a single scalar value."""
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=LAND_COLOR, zorder=0)

    gdf = footprints.copy()
    gdf['value'] = gdf['code'].map(values_by_code)
    gdf.plot(column='value', ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
              edgecolor='black', linewidth=0.6, zorder=2, missing_kwds={'color': '#d9d9d9'})

    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, edgecolor='dimgray', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='black', zorder=3)

    for _, row in gdf.iterrows():
        c = row['geometry'].centroid
        label = f"{row['code']}\n{row['value']:.3f}" if not np.isnan(row['value']) else row['code']
        ax.text(c.x, c.y, label, transform=ccrs.PlateCarree(), ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_title(title, fontsize=12, fontweight='bold')

#%%
# =================================================================
# FIGURE 1 — Country choropleth: RL event frequency (T1 relative / T2 absolute)
# =================================================================
print("Figure 1 — RL event frequency choropleth...")

rl_country_events = rl_events_df[rl_events_df['country'] != 'EU_AGG']
freq_by_track = {
    track: (rl_country_events[rl_country_events['track'] == track]
            .groupby('country')['duration'].sum() / N_VALID_DAYS)
            .reindex(CODES, fill_value=0.0).to_dict()
    for track in RL_TRACKS
}

vmax_freq = max(max(d.values()) for d in freq_by_track.values())
norm_freq = Normalize(vmin=0, vmax=vmax_freq)
cmap_freq = plt.get_cmap('YlOrRd')

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': PROJ})
for ax, track in zip(axes, RL_TRACKS):
    plot_choropleth_panel(ax, freq_by_track[track], cmap_freq, norm_freq, RL_TRACK_LABEL[track])

sm = cm.ScalarMappable(cmap=cmap_freq, norm=norm_freq)
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, shrink=0.6)
cbar.set_label('Relative frequency of dunkelflaute event-days (fraction of NDJFM days)')

fig.suptitle(f"Residual-Load Dunkelflaute Frequency by Country ({valid_winters[0]}–{valid_winters[-1]})",
             fontsize=15, y=0.98)
plt.savefig(OUT_DIR / "fig01_rl_frequency_choropleth.png", dpi=250, bbox_inches='tight')
plt.show()
plt.close()

#%%
# =================================================================
# FIGURE 2 — Country choropleth: RL mean event severity (T1 / T2)
# =================================================================
print("Figure 2 — RL mean severity choropleth...")

severity_by_track = {
    track: (rl_country_events[rl_country_events['track'] == track]
            .groupby('country')['severity_S'].mean())
            .reindex(CODES).to_dict()
    for track in RL_TRACKS
}

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': PROJ})
for ax, track in zip(axes, RL_TRACKS):
    vals = severity_by_track[track]
    finite_vals = [v for v in vals.values() if not np.isnan(v)]
    norm_sev = Normalize(vmin=0, vmax=max(finite_vals) if finite_vals else 1)
    plot_choropleth_panel(ax, vals, 'inferno_r', norm_sev,
                           f"{RL_TRACK_LABEL[track]}")
    sm = cm.ScalarMappable(cmap='inferno_r', norm=norm_sev)
    fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.06,
                 label='Mean severity S (σ-normalised)' if track == 'T1' else 'Mean severity S')

fig.suptitle("Mean Event Severity by Country — Residual-Load Pipeline", fontsize=15, y=0.98)
plt.savefig(OUT_DIR / "fig02_rl_severity_choropleth.png", dpi=250, bbox_inches='tight')
plt.show()
plt.close()

#%%
# =================================================================
# FIGURE 3 — Per-country climatology: frequency, duration, severity (RL)
# =================================================================
print("Figure 3 — RL per-country climatology bars...")

full_idx = pd.MultiIndex.from_product([CODES, RL_TRACKS], names=['country', 'track'])
rl_metrics = (rl_country_events.groupby(['country', 'track'])
              .agg(event_days=('duration', 'sum'),
                   mean_duration=('duration', 'mean'),
                   mean_severity=('severity_S', 'mean'))
              .reindex(full_idx))
rl_metrics['freq'] = (rl_metrics['event_days'] / N_VALID_DAYS).fillna(0.0)
rl_metrics = rl_metrics.reset_index()

track_colors = {'T1': '#4C72B0', 'T2': '#C44E52'}
x = np.arange(len(CODES))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metric_specs = [('freq', 'Relative frequency\n(fraction of NDJFM days)'),
                 ('mean_duration', 'Mean event duration (days)'),
                 ('mean_severity', 'Mean severity S')]

for ax, (metric, ylabel) in zip(axes, metric_specs):
    for i, track in enumerate(RL_TRACKS):
        sub = rl_metrics[rl_metrics['track'] == track].set_index('country').reindex(CODES)
        ax.bar(x + (i - 0.5) * width, sub[metric].values, width,
               label=RL_TRACK_LABEL[track], color=track_colors[track], edgecolor='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CODES)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

axes[0].legend(loc='upper left', fontsize=8, framealpha=0.9)
fig.suptitle("Per-Country Dunkelflaute Climatology — Residual-Load Pipeline", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig03_rl_climatology_bars.svg", bbox_inches='tight')
plt.show()
plt.close()

#%%
# =================================================================
# FIGURE 4 — Cross-track comparison (supply-side): IRENA vs. Hu-ratio vs. Hu-MW
# Sensitivity of the purely-meteorological detection to capacity assumptions.
# =================================================================
print("Figure 4 — supply-side cross-track comparison...")

supply_agg = (supply_annual_df.groupby(['country', 'track'])
              .agg(n_events=('n_events', 'sum'), total_S=('total_S', 'sum'))
              .reset_index())
supply_agg['events_per_winter'] = supply_agg['n_events'] / N_WINTERS

track_palette = {'IRENA_Relative_10pct': '#55A868',
                  'MaxPot_Relative_10pct': '#4C72B0',
                  'MaxPot_Absolute_15pct_floor': '#C44E52'}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, (metric, ylabel) in zip(
        axes, [('events_per_winter', 'Events per winter'), ('total_S', 'Total severity S (all winters)')]):
    for i, track in enumerate(SUPPLY_TRACKS):
        sub = supply_agg[supply_agg['track'] == track].set_index('country').reindex(CODES)
        ax.bar(x + (i - 1) * width * 0.7, sub[metric].values, width * 0.7,
               label=SUPPLY_TRACK_LABEL[track], color=track_palette[track], edgecolor='k', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CODES)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
fig.suptitle("Supply-Side Detection Sensitivity to Capacity Assumptions\n"
             "(Hu-ratio track is the headline; IRENA / Hu-MW are robustness checks)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig04_supply_track_comparison.svg", bbox_inches='tight')
plt.show()
plt.close()

#%%
# =================================================================
# FIGURE 5 — Annual trends: occurrence & severity per country (T1)
# Mann-Kendall tau (proxy) + Theil-Sen slope, mirroring the CDHW trend figure.
# =================================================================
print("Figure 5 — annual trend small multiples (T1)...")

def trend_grid(metric_col, ylabel, title, fname):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes = axes.ravel()

    for ax, code in zip(axes, CODES):
        ts = (rl_annual_df[(rl_annual_df['country'] == code) & (rl_annual_df['track'] == 'T1')]
              .sort_values('winter'))
        yrs, vals = ts['winter'].values, ts[metric_col].values

        tau, p = kendalltau(yrs, vals)
        slope, intercept, lo, hi = theilslopes(vals, yrs)

        ax.scatter(yrs, vals, s=18, color='#4C72B0', alpha=0.8, zorder=3)
        ax.plot(yrs, intercept + slope * yrs, color='#C44E52', lw=1.8, zorder=4,
                label=f"Theil-Sen: {slope:.3f}/yr")
        ax.set_title(f"{code} — τ={tau:.2f} (p={p:.2f})", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=7, loc='upper left')

    for ax in axes[len(CODES):]:
        ax.axis('off')

    fig.supxlabel("Winter")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches='tight')
    plt.show()
    plt.close()

trend_grid('n_events', 'Events per winter',
           "Annual Trend in Dunkelflaute Occurrence — T1 (Relative)",
           "fig05a_trend_occurrence.svg")
trend_grid('total_S', 'Total severity S per winter',
           "Annual Trend in Dunkelflaute Severity — T1 (Relative)",
           "fig05b_trend_severity.svg")

#%%
# =================================================================
# FIGURE 6 — Copper-plate (EU_AGG) vs. national aggregation
# Demonstrates the spatial-smoothing effect of continental balancing
# (cf. Van der Wiel et al. 2019: national events rarely coincide continent-wide).
# =================================================================
print("Figure 6 — copper-plate vs. national comparison...")

all_geo = CODES + ['EU_AGG']
cp_freq = {}
for track in RL_TRACKS:
    sub = rl_events_df[rl_events_df['track'] == track]
    cp_freq[track] = (sub.groupby('country')['duration'].sum() / N_VALID_DAYS).reindex(all_geo, fill_value=0.0)

fig, ax = plt.subplots(figsize=(10, 6))
xg = np.arange(len(all_geo))
for i, track in enumerate(RL_TRACKS):
    colors = [track_colors[track]] * len(CODES) + ['black']
    bars = ax.bar(xg + (i - 0.5) * width, cp_freq[track].values, width,
                   label=RL_TRACK_LABEL[track], edgecolor='k', linewidth=0.5)
    for b, geo in zip(bars, all_geo):
        b.set_color(track_colors[track] if geo != 'EU_AGG' else '#333333')
        b.set_alpha(1.0 if geo != 'EU_AGG' else 0.9)

ax.set_xticks(xg)
ax.set_xticklabels(all_geo)
ax.axvline(len(CODES) - 0.5, color='gray', linestyle=':', lw=1)
ax.set_ylabel('Relative frequency of event-days')
ax.set_title("National vs. Copper-Plate (EU_AGG) Dunkelflaute Frequency\n"
             "Continental aggregation smooths out non-simultaneous national shortfalls", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig06_copperplate_comparison.svg", bbox_inches='tight')
plt.show()
plt.close()

#%%
# =================================================================
# FIGURE 7 — Per-cell spatial climatology (supply-side, 50/50 wind-solar mix)
# Grid-resolution complement to the country-level choropleths above.
# =================================================================
print("Figure 7 — per-cell spatial climatology...")

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': PROJ})

panels = [('freq', 'viridis', 'Relative frequency of low-CF days'),
          ('mean_S', 'plasma', 'Mean severity S')]

for ax, (var, cmap, label) in zip(axes, panels):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=OCEAN_COLOR, zorder=0)
    da = cell_stats[var]
    cf = ax.contourf(da.longitude, da.latitude, da, levels=20, cmap=cmap,
                      transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6, edgecolor='white', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.7, edgecolor='black', zorder=3)
    fig.colorbar(cf, ax=ax, orientation='horizontal', fraction=0.05, pad=0.06, label=label)
    ax.set_title(f"Supply-Side {label} (grid-cell, 50/50 wind-solar mix)", fontsize=11)

fig.suptitle(f"Pixel-Level Supply-Side Dunkelflaute Climatology "
             f"(n={int(cell_stats.attrs.get('n_winters', N_WINTERS))} winters)", fontsize=14, y=0.98)
plt.savefig(OUT_DIR / "fig07_cell_climatology_map.png", dpi=250, bbox_inches='tight')
plt.show()
plt.close()

#%%
# =================================================================
# FIGURE 8 — Z500 anomaly composite over RL event dates
# Flagged as the highest-leverage addition: demonstrates the shared
# anticyclonic-blocking driver rather than merely asserting it.
#
# CAVEAT: the ERA5 domain downloaded for this subproject is truncated to
# the regional bounding box used for capacity-factor work (N62 W-14 S42 E18).
# It does not capture the full synoptic-scale blocking pattern — the ridge
# axis / blocking centre may lie outside this window (e.g. over Scandinavia
# or the Norwegian Sea). Interpret the composite as a regional footprint of
# the blocking signature, not a full-domain detection.
# =================================================================
print("Figure 8 — Z500 anomaly composite...")

ERA5_dat = os.environ.get("ERA5_dat")
z_path = Path(ERA5_dat) / "df_dat" / "processed" / "df_dat_cleaned.zarr"

ds_z = xr.open_zarr(z_path, consolidated=True)
z = ds_z['z']
if 'level' in z.dims:
    z = z.isel(level=0)

# Hourly -> daily mean (mirrors the coarsen(time=24) step in 04_demand_model.py)
z_daily = z.coarsen(time=24, boundary='trim', coord_func='min').mean()
times_z = pd.DatetimeIndex(z_daily.time.values)
is_winter_z = times_z.month.isin(WINTER_MONTHS)
z_clim = z_daily.isel(time=is_winter_z).mean('time').compute()

def event_day_index(events_df, track):
    sub = events_df[(events_df['track'] == track) & (events_df['country'] != 'EU_AGG')]
    days = set()
    for _, row in sub.iterrows():
        days.update(pd.date_range(row['start_date'], row['end_date']))
    return pd.DatetimeIndex(sorted(days))

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': PROJ})
anom_max = 0
anoms = {}
for track in RL_TRACKS:
    days = event_day_index(rl_events_df, track)
    z_track = z_daily.sel(time=z_daily.time.isin(days.values)).mean('time').compute()
    anom = z_track - z_clim
    anoms[track] = anom
    anom_max = max(anom_max, float(np.nanmax(np.abs(anom))))

norm_anom = TwoSlopeNorm(vmin=-anom_max, vcenter=0, vmax=anom_max)

for ax, track in zip(axes, RL_TRACKS):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    cf = ax.contourf(anoms[track].longitude, anoms[track].latitude, anoms[track],
                      levels=21, cmap='RdBu_r', norm=norm_anom, transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.7, edgecolor='black', zorder=3)
    ax.set_title(f"{RL_TRACK_LABEL[track]}\n(n={len(event_day_index(rl_events_df, track))} composited days)",
                 fontsize=11)

sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm_anom)
fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08, shrink=0.6,
             label='Z500 anomaly (m) relative to NDJFM climatology')

fig.suptitle("500 hPa Geopotential Height Anomaly Composite over Dunkelflaute Event Days\n"
             "(regional domain — see caveat in script comments)", fontsize=13, y=1.02)
plt.savefig(OUT_DIR / "fig08_z500_composite.png", dpi=250, bbox_inches='tight')
plt.show()
plt.close()

print(f"\nAll figures written to: {OUT_DIR}")
# %%
