#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig6_scenario_severity_maps.py
Cumulative Dunkelflaute burden per country: present-day vs fully-built fleet.
Metric = firm-energy requirement (GWh/winter), absolute and scenario-invariant,
loaded from scenario_summary_{SCENARIO}.parquet written by 05_DF_RL_ID.py.
Standalone: assumes both scenarios have been run.
"""
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#%%
# Config

OUT_DIR       = Path('./../Results/DF_RL')
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
PATH_FIG      = Path('./../Figs/EGU26/')
PATH_FIG.mkdir(parents=True, exist_ok=True)

METRIC = 'firm_gwh_per_winter'   # absolute, scenario-invariant. Alt: bring 'total_excess' via annual_df.
METRIC_LABEL = 'Firm-energy requirement (GWh / winter)'

# ADMIN name -> code, to attach values to polygons
NAME2CODE = {"France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE",
             "Denmark": "DK", "United Kingdom": "GB", "Ireland": "IE"}

LAEA = ccrs.LambertAzimuthalEqualArea(central_longitude=10.0, central_latitude=52.0)
PC   = ccrs.PlateCarree()

#%%
# Load both scenario summaries and the country polygons

sum_pd = pd.read_parquet(OUT_DIR / 'scenario_summary_present_day.parquet')
sum_fb = pd.read_parquet(OUT_DIR / 'scenario_summary_fully_built.parquet')

vals = {'present_day': sum_pd[METRIC], 'fully_built': sum_fb[METRIC]}
print("Values being mapped:")
print(pd.DataFrame(vals).round(0).to_string())

countries_gdf = gpd.read_file(str(COUNTRIES_SHP))
study = countries_gdf[countries_gdf['ADMIN'].isin(NAME2CODE)].copy()
study['code'] = study['ADMIN'].map(NAME2CODE)
study = study.dissolve(by='code').reset_index()          # one polygon per country

# GB polygon carries all of Britain incl. NI-adjacent isles; leave as-is (matches ENTSO-E GB zone caveat).

DOMAIN_EXTENT = [-14.0, 20.0, 41.0, 62.5]                 # matches the DF study domain

#%%
# Shared color scale (vmax from present-day = the larger scenario)

# log-ish spread across countries -> robust vmax on the larger scenario
all_pos = np.concatenate([vals['present_day'].values, vals['fully_built'].values])
all_pos = all_pos[all_pos > 0]
VMIN = 0.0
VMAX = np.nanpercentile(vals['present_day'].values, 99)   # anchor to present-day range
CMAP = 'YlOrRd'
norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)

print(f"\nShared scale: vmin={VMIN:.0f}, vmax={VMAX:.0f} GWh/winter")

#%%
# Map style (mirrors the capacity-map helper)

def format_map(ax):
    ax.add_feature(cfeature.OCEAN, facecolor='#C6E2F5', zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor='0.93',    zorder=0)
    ax.add_feature(cfeature.BORDERS,   lw=0.9, edgecolor='0.15', zorder=4)
    ax.add_feature(cfeature.COASTLINE, lw=0.9,                   zorder=4)
    ax.set_extent(DOMAIN_EXTENT, crs=PC)
    gl = ax.gridlines(crs=PC, draw_labels=True, lw=0.25, color='0.6', alpha=0.5, ls='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7}; gl.ylabel_style = {'size': 7}

def draw_scenario(ax, value_series, title):
    format_map(ax)
    for _, row in study.iterrows():
        v = value_series.get(row['code'], np.nan)
        face = CMAP_call(v)
        ax.add_geometries([row.geometry], crs=PC, facecolor=face,
                          edgecolor='0.25', linewidth=0.6, zorder=2)
        # annotate value at polygon centroid
        c = row.geometry.representative_point()
        ax.text(c.x, c.y, f"{v:,.0f}", transform=PC, ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if v > 0.55 * VMAX else '0.15', zorder=5)
    ax.set_title(title, fontsize=11, pad=8, fontweight='bold')

# fill lookup honoring the shared norm/cmap
_cmap = mpl.cm.get_cmap(CMAP)
def CMAP_call(v):
    if not np.isfinite(v):
        return '0.85'
    return _cmap(norm(v))

#%%
# Two-panel figure

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), subplot_kw={'projection': LAEA})

draw_scenario(axes[0], vals['present_day'], "Present-day fleet")
draw_scenario(axes[1], vals['fully_built'], "Fully-built fleet")

# shared colorbar
sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                    pad=0.06, shrink=0.55, aspect=35, extend='max')
cbar.set_label(METRIC_LABEL, fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.suptitle("Winter Dunkelflaute burden: benefit of full renewable build-out",
             fontsize=13, y=1.02)

plt.savefig(PATH_FIG / 'fig6_scenario_severity_maps.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig(PATH_FIG / 'fig6_scenario_severity_maps.svg', bbox_inches='tight')
plt.show()
print(f"Saved to {PATH_FIG}")
# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig6_scenario_metric_maps.py
Per-country Dunkelflaute event characteristics: present-day vs fully-built fleet.
3 x 2 grid of maps  ->  rows = metric, columns = scenario:
    row 1: average yearly event DURATION   (from the annual/event table)
    row 2: average yearly event SEVERITY   (from the annual/event table)
    row 3: firm-energy requirement         (from scenario_summary_*, original fig6)
Standalone: assumes both scenarios have been run by 05_DF_RL_ID.py.

NOTE: scenario_summary_*.parquet holds only DERIVED per-country stats
      (std_ratio, corr_D_S, frac_rl_in_supply, jaccard, amplification,
       firm_gwh_per_winter, rl0_day_frac) -- it has NO raw duration/severity.
      Those come from the per-event / annual table (ANNUAL_FILE below).
"""
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#%%
# Config

OUT_DIR       = Path('./../Results/DF_RL')
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
PATH_FIG      = Path('./../Figs/EGU26/')
PATH_FIG.mkdir(parents=True, exist_ok=True)

# ---- Where duration & severity come from -----------------------------------
# CONFIRM these 4 names against what 05_DF_RL_ID.py writes. The annual/event
# table is read per scenario; each country's value = mean over its rows
# ("average yearly"). If duration/severity instead live in ONE combined file
# with a 'scenario' column, see the alt loader note in load_event_metric().
ANNUAL_FILE  = 'annual_{scenario}.parquet'   # <-- CONFIRM filename pattern
COUNTRY_KEY  = 'country'                      # <-- CONFIRM country column (or index name)
DURATION_COL = 'duration'                     # <-- CONFIRM column holding event duration
SEVERITY_COL = 'severity'                     # <-- CONFIRM column holding event severity

# ---- Metrics to map, one row each (top -> bottom) --------------------------
METRICS = [
    dict(key='duration', source='annual',  col=DURATION_COL,
         short='Duration', label='Avg. event duration (days)',
         cmap='YlGnBu', fmt='{:,.1f}'),
    dict(key='severity', source='annual',  col=SEVERITY_COL,
         short='Severity', label='Avg. event severity',
         cmap='YlOrBr', fmt='{:,.1f}'),
    dict(key='firm',     source='summary', col='firm_gwh_per_winter',
         short='Firm energy', label='Firm-energy requirement (GWh / winter)',
         cmap='YlOrRd', fmt='{:,.0f}'),
]

SCENARIOS = [('present_day', 'Present-day fleet'),
             ('fully_built', 'Fully-built fleet')]

# ADMIN name -> code, to attach values to polygons
NAME2CODE = {"France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE",
             "Denmark": "DK", "United Kingdom": "GB", "Ireland": "IE"}

LAEA = ccrs.LambertAzimuthalEqualArea(central_longitude=10.0, central_latitude=52.0)
PC   = ccrs.PlateCarree()

#%%
# Load scenario summaries (firm-energy) and country polygons

summaries = {
    'present_day': pd.read_parquet(OUT_DIR / 'scenario_summary_present_day.parquet'),
    'fully_built': pd.read_parquet(OUT_DIR / 'scenario_summary_fully_built.parquet'),
}

countries_gdf = gpd.read_file(str(COUNTRIES_SHP))
study = countries_gdf[countries_gdf['ADMIN'].isin(NAME2CODE)].copy()
study['code'] = study['ADMIN'].map(NAME2CODE)
study = study.dissolve(by='code').reset_index()          # one polygon per country

# GB polygon carries all of Britain incl. NI-adjacent isles; leave as-is (matches ENTSO-E GB zone caveat).

DOMAIN_EXTENT = [-14.0, 20.0, 41.0, 62.5]                 # matches the DF study domain

#%%
# Value loaders -> each returns a Series indexed by country code

def load_summary_metric(scenario, col):
    """Firm-energy etc., already one row per country in the summary."""
    return summaries[scenario][col]

def load_event_metric(scenario, col):
    """Average-yearly duration/severity: mean over the annual/event table rows,
    per country. Country may be the index or a column."""
    df = pd.read_parquet(OUT_DIR / ANNUAL_FILE.format(scenario=scenario))
    # --- alt: single combined file with a 'scenario' column ------------------
    # df = pd.read_parquet(OUT_DIR / 'annual_events.parquet')
    # df = df[df['scenario'] == scenario]
    # -------------------------------------------------------------------------
    if COUNTRY_KEY in df.columns:
        grp = df.groupby(COUNTRY_KEY)[col]
    else:                                   # country is the (row) index
        grp = df.groupby(level=0)[col]
    return grp.mean()                       # "average yearly" value per country

def get_series(scenario, metric):
    if metric['source'] == 'summary':
        return load_summary_metric(scenario, metric['col'])
    return load_event_metric(scenario, metric['col'])

# Pre-compute every (metric, scenario) series once
series = {(m['key'], s): get_series(s, m) for m in METRICS for s, _ in SCENARIOS}

print("Values being mapped:")
for m in METRICS:
    tbl = pd.DataFrame({s: series[(m['key'], s)] for s, _ in SCENARIOS})
    print(f"\n{m['short']} ({m['col']}):")
    print(tbl.round(2).to_string())

#%%
# Per-metric colour scale: each row gets its own norm + cmap (units differ).
# vmax anchored to the present-day 99th percentile (present-day is the larger
# scenario -- full build-out reduces the burden). To anchor to the max across
# BOTH scenarios instead, use the 'combined' line.

def build_norm(metric):
    pd_vals = series[(metric['key'], 'present_day')].to_numpy(dtype=float)
    # combined = np.concatenate([series[(metric['key'], s)].to_numpy(float) for s, _ in SCENARIOS])
    vmax = np.nanpercentile(pd_vals, 99)
    if not np.isfinite(vmax) or vmax <= 0:               # fallback if pctile degenerate
        vmax = np.nanmax(pd_vals)
    return mpl.colors.Normalize(vmin=0.0, vmax=float(vmax))

for m in METRICS:
    m['norm']     = build_norm(m)
    m['cmap_obj'] = mpl.cm.get_cmap(m['cmap'])
    print(f"{m['short']:>12} scale: vmin=0, vmax={m['norm'].vmax:.2f}")

#%%
# Map style (mirrors the capacity-map helper)

def format_map(ax):
    ax.add_feature(cfeature.OCEAN, facecolor='#C6E2F5', zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor='0.93',    zorder=0)
    ax.add_feature(cfeature.BORDERS,   lw=0.9, edgecolor='0.15', zorder=4)
    ax.add_feature(cfeature.COASTLINE, lw=0.9,                   zorder=4)
    ax.set_extent(DOMAIN_EXTENT, crs=PC)
    gl = ax.gridlines(crs=PC, draw_labels=True, lw=0.25, color='0.6', alpha=0.5, ls='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7}; gl.ylabel_style = {'size': 7}

def fill_color(v, metric):
    if not np.isfinite(v):
        return '0.85'
    return metric['cmap_obj'](metric['norm'](v))

def draw_panel(ax, value_series, metric, title=None):
    format_map(ax)
    vmax = metric['norm'].vmax
    for _, row in study.iterrows():
        v = value_series.get(row['code'], np.nan)
        ax.add_geometries([row.geometry], crs=PC, facecolor=fill_color(v, metric),
                          edgecolor='0.25', linewidth=0.6, zorder=2)
        # annotate value at polygon representative point
        c = row.geometry.representative_point()
        txt = metric['fmt'].format(v) if np.isfinite(v) else '--'
        ax.text(c.x, c.y, txt, transform=PC, ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if (np.isfinite(v) and v > 0.55 * vmax) else '0.15',
                zorder=5)
    if title:
        ax.set_title(title, fontsize=11, pad=8, fontweight='bold')

#%%
# 3 x 2 figure: rows = metric, cols = scenario

nrows, ncols = len(METRICS), len(SCENARIOS)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 15.5),
                         subplot_kw={'projection': LAEA})
axes = np.atleast_2d(axes)

for i, metric in enumerate(METRICS):
    for j, (scen_key, scen_title) in enumerate(SCENARIOS):
        ax = axes[i, j]
        draw_panel(ax, series[(metric['key'], scen_key)], metric,
                   title=scen_title if i == 0 else None)    # scenario titles on top row only

    # left-hand row label (rotated); nudge x more negative if it clips y-tick labels
    axes[i, 0].text(-0.20, 0.5, metric['short'], transform=axes[i, 0].transAxes,
                    rotation=90, va='center', ha='center',
                    fontsize=12, fontweight='bold')

    # one colorbar per row, to the right of that row
    sm = ScalarMappable(norm=metric['norm'], cmap=metric['cmap']); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[i, :], orientation='vertical',
                        pad=0.02, shrink=0.85, aspect=25, extend='max')
    cbar.set_label(metric['label'], fontsize=9)
    cbar.ax.tick_params(labelsize=8)

fig.suptitle("Winter Dunkelflaute events: duration, severity and firm-energy burden",
             fontsize=14, y=0.995)

plt.savefig(PATH_FIG / 'fig6_scenario_metric_maps.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig(PATH_FIG / 'fig6_scenario_metric_maps.svg', bbox_inches='tight')
plt.show()
print(f"Saved to {PATH_FIG}")
# %%