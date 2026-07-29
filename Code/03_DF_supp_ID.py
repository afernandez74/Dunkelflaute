#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_DF_supply_ID.py
==================
Supply-side dunkelflaute detection on raw combined capacity factors
(Otero et al. 2022 "low wind and solar production" events; cf. Raynaud et al.
2018, Meng et al. 2025).

Purely meteorological: events are periods of low *potential* generation.
Absolute installed capacity (MW) cancels out of percentile-based detection;
only the dimensionless wind/solar mix weight enters (capacity shares).

    CF_sys = W_WIND * CF_wind + W_SOLAR * CF_solar_24h

Detection grammar:
    within-NDJFM percentile threshold (lower tail)
    -> flag on the FULL daily index masked to winter
       (never on a season-subset array: runs must not bridge Mar -> Nov)
    -> gap pooling (events separated by >= 2 days are distinct; Otero'22)
    -> minimum duration filter
    -> severity = summed standardized threshold shortfall (Otero'22)

Two tracks:
    A. Per-cell   : threshold per grid cell -> frequency / severity maps
                    (spatial climatology; CDHW Fig 2 analog)
    B. Per-country: buffered-mask country-mean CF_sys -> event catalogue
                    + annual aggregates (analog of planned RL outputs)

Country masks include a coastal buffer (~200 km) so offshore cells are
assigned to their nearest country -- offshore wind included by design.

Inputs
------
  ./../Results/CF_daily/CF_wind_daily.zarr        (02_calc_CF_daily.py)
  ./../Results/CF_daily/CF_solar_24h_daily.zarr   (02_calc_CF_daily.py)
  ~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp

Outputs
-------
  ./../Results/DF_supply/df_supply_cell_stats.nc      per-cell climatology
  ./../Results/DF_supply/df_supply_events.parquet     one row per event
  ./../Results/DF_supply/df_supply_annual.parquet     one row per (country, winter), zero-filled

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
# Config 

# Portfolio mix weights (dimensionless capacity SHARES, w + s = 1).
# TODO: replace with real per-country capacity shares once
#       Data/installed_capacity.csv is populated (IRENA / ENTSO-E vintage).
#       Absolute MW is irrelevant for detection; only the mix enters. DONE BELOW

# reference w-solar and w-wind for track A
REF_W_SOLAR = 0.25
REF_W_WIND  = 1.0 - REF_W_SOLAR
CF_sys = (REF_W_WIND * CF_wind_dly + REF_W_SOLAR * CF_solar_dly).rename('CF_sys')

WINTER_MONTHS = [11, 12, 1, 2, 3]   # extended winter (NDJFM)
THRESHOLD_PCT = 10                  # lower-tail within-season percentile
SEVERE_PCT    = 5                   # severe tier (optional flag per event)
MAX_GAP       = 1                   # Otero'22: events separated by >= 2 days
                                    # are distinct -> 1-day gaps are pooled
MIN_DURATION  = 3                   # days; consistent with CDHW filtering

BUFFER_DEG    = 1.8                 # ~200 km coastal buffer for offshore cells
                                    # (degrees; anisotropic in lon -- acceptable v1)


# Absolute renewable scarcity threshold
# Days below this system CF are considered absolute supply scarcity events
ABS_CF_THRESHOLD = 0.15

COUNTRIES = {                       # Natural Earth ADMIN name -> code
    "France"        : "FR",         # (ADMIN, not SOVEREIGNT: keeps Greenland
    "Belgium"       : "BE",         #  out of DK and overseas out of GB/FR)
    "Netherlands"   : "NL",
    "Germany"       : "DE",
    "Denmark"       : "DK",
    "United Kingdom": "GB",         # NOTE: ENTSO-E GB zone excludes Northern
    "Ireland"       : "IE",         # Ireland (NI is in IE_SEM). TODO before RL
}                                   # validation: move NI polygon to IE.

CF_DAILY_DIR  = Path('./../Results/CF_daily')
OUT_DIR       = Path('./../Results/DF_supply')
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()

OUT_DIR.mkdir(parents=True, exist_ok=True)

#%%
# load installed capacity data
IC_PATH = Path('./../Data/IC/IC.csv')
IC = pd.read_csv(IC_PATH, index_col=0)
IC.index = IC.index.map({          # IRENA country names -> your codes
    'France': 'FR', 'Belgium': 'BE', 'Netherlands': 'NL', 'Germany': 'DE',
    'Denmark': 'DK', 'United Kingdom': 'GB', 'Ireland': 'IE',
})

# Per-country capacity shares of the three sources (sum to 1 within country)
IC_TOTAL_REN = IC[['Onshore Wind (MW)', 'Offshore Wind (MW)', 'Solar (MW)']].sum(axis=1)
W = pd.DataFrame({
    'won':  IC['Onshore Wind (MW)']  / IC_TOTAL_REN,
    'woff': IC['Offshore Wind (MW)'] / IC_TOTAL_REN,
    'wsol': IC['Solar (MW)']         / IC_TOTAL_REN,
})
IC_CAPACITY_TOTAL = IC_TOTAL_REN     # MW, for the eventual GWh severity translation

CAP_VINTAGE = 2024
print("Per-country renewable mix shares (onshore / offshore / solar):")
print(W.round(3))

#%%
# Load daily capacity factors, build system CF ─────────────────────────────

CHUNKS = {'time': -1, 'latitude': 20, 'longitude': 20}

CF_wind_dly  = xr.open_zarr(CF_DAILY_DIR / 'CF_wind_daily.zarr',
                            consolidated=True, chunks = {})['CF_wind_dly']
CF_solar_dly = xr.open_zarr(CF_DAILY_DIR / 'CF_solar_24h_daily.zarr',
                            consolidated=True, chunks={})['CF_solar_24h_dly']

CF_wind_dly, CF_solar_dly = xr.align(CF_wind_dly, CF_solar_dly, join='inner')

CF_sys = (W_WIND * CF_wind_dly + W_SOLAR * CF_solar_dly).rename('CF_sys')
CF_sys.attrs.update({
    'long_name'   : 'Combined system capacity factor (potential generation per unit capacity)',
    'description' : f'{W_WIND:.2f} x CF_wind + {W_SOLAR:.2f} x CF_solar_24h',
    'units'       : 'dimensionless',
})
CF_sys = CF_sys.compute()

times    = pd.DatetimeIndex(CF_sys.time.values)
month    = times.month.values
n_lat    = CF_sys.sizes['latitude']
n_lon    = CF_sys.sizes['longitude']

print(f"CF_sys — shape: {CF_sys.shape}")
print(f"  time: {times[0].date()} → {times[-1].date()}")
print(f"  mix : wind {W_WIND:.2f} / solar {W_SOLAR:.2f}")


#%%
# Winter calendar: labels, and the valid-winter mask ───────────────────────
# Winter label: Nov/Dec belong to the following year's winter (winter 2010 =
# Nov 2009 – Mar 2010). 

winter_id = np.where(month >= 11, times.year.values + 1, times.year.values)
is_ndjfm  = np.isin(month, WINTER_MONTHS)

winters_with_nov = set(winter_id[month == 11])
winters_with_mar = set(winter_id[month == 3])
valid_winters    = sorted(winters_with_nov & winters_with_mar)

winter_ok = is_ndjfm & np.isin(winter_id, valid_winters)
n_winters = len(valid_winters)

print(f"Complete winters: {n_winters}  ({valid_winters[0]} → {valid_winters[-1]})")
print(f"Winter days in detection window: {int(winter_ok.sum()):,}")


#%%
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
        
        
        freq_arr[i, j] = len(ev_rows) / n_winters
        if ev_rows:
            durs = [r['duration']   for r in ev_rows]
            Ss   = [r['severity_S'] for r in ev_rows]
            mean_dur_arr[i, j] = np.mean(durs)
            mean_S_arr[i, j]   = np.mean(Ss)
            tot_S_yr_arr[i, j] = np.sum(Ss) / n_winters

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
# ─── Track B: per-country detection → event catalogue (relative + absolute) ───

country_codes = list(COUNTRIES.values())
all_rows      = []
thresholds    = {}
event_masks   = {}          # relative-track masks, kept for diagnostics + concurrence
abs_event_masks = {}        # absolute-track masks

for code in country_codes:
    x = cf_country[code].values

    # Relative (within-NDJFM percentile) track
    evt, thr, thr_sev, sigma = detect_events_1d(
        x, winter_ok, thr_pct=THRESHOLD_PCT, severe_pct=SEVERE_PCT,
        max_gap=MAX_GAP, min_duration=MIN_DURATION)
    thresholds[code]  = thr
    event_masks[code] = evt

    for r in characterize_events_1d(evt, x, thr, thr_sev, sigma):
        s_idx, e_idx = r.pop('start_idx'), r.pop('end_idx')
        all_rows.append(dict(
            country=code, track='relative',
            start_date=times[s_idx], end_date=times[e_idx],
            winter=int(winter_id[s_idx]),          # single source of truth
            thr_cf=float(thr), **r,
        ))

    # Absolute (fixed CF floor) track
    abs_evt, abs_thr, _, abs_sigma = detect_events_1d(
        x, winter_ok, max_gap=MAX_GAP, min_duration=MIN_DURATION,
        absolute_threshold=ABS_CF_THRESHOLD)
    abs_event_masks[code] = abs_evt

    for r in characterize_events_1d(abs_evt, x, abs_thr, abs_thr, abs_sigma):
        r.pop('is_severe')                          # meaningless when thr_sev == thr
        s_idx, e_idx = r.pop('start_idx'), r.pop('end_idx')
        all_rows.append(dict(
            country=code, track='absolute',
            start_date=times[s_idx], end_date=times[e_idx],
            winter=int(winter_id[s_idx]),
            thr_cf=float(ABS_CF_THRESHOLD), **r,
        ))

ev_df = pd.DataFrame(all_rows)

print(f"\nTotal events — relative: {(ev_df.track=='relative').sum()}, "
      f"absolute: {(ev_df.track=='absolute').sum()}")
print("\nRelative thresholds (CF_sys, NDJFM "
      f"{THRESHOLD_PCT}th pct): " +
      ", ".join(f"{c}: {v:.3f}" for c, v in thresholds.items()))
print("\nPer-country means (relative track):")
print(ev_df[ev_df.track == 'relative']
      .groupby('country')[['duration', 'severity_S', 'deficit_cf_days']]
      .mean().round(2).to_string())


#%%
# ─── Annual aggregates (zero-filled over complete winters, per track) ─────────
# GWh translation is meaningful for the absolute track (shortfall = below a fixed
# fraction of nameplate); for the relative track deficit_cf_days is vs an arbitrary
# percentile, so GWh is left as a relative diagnostic only.

def aggregate_annual(df_track):
    a = (df_track.groupby(['country', 'winter'])
         .agg(n_events   =('duration',        'size'),
              n_severe   =('is_severe',        'sum') if 'is_severe' in df_track
                          else ('duration', 'size'),
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
# in the annual aggregation or post-hoc:
annual['deficit_GWh'] = annual.apply(
    lambda r: r['total_cf'] * IC_CAPACITY_TOTAL.get(r['country'], np.nan) * 24 / 1000,
    axis=1)
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
im = ax.pcolormesh(lon, lat, onshore_freq, transform=ccrs.PlateCarree(),
                   cmap='YlOrRd', vmin=vmin, vmax=vmax, shading='nearest', zorder=1)
# Offshore — same color scale, hatched to mark the sea zone
ax.pcolormesh(lon, lat, offshore_freq, transform=ccrs.PlateCarree(),
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
