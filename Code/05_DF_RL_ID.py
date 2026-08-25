#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_DF_RL.py — Residual-Load Dunkelflaute Detection
Detects winter energy compound events based on residual load (Demand - Generation).
Tracks:
  T1 (Relative): RL > 90th percentile of winter distribution.
  T2 (Absolute): RL > 0 MW (Unserved energy).
"""
#%%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.feature as cfeature
from scipy.stats import kendalltau
from pathlib import Path
from scipy.ndimage import label


import geopandas as gpd
import shapely
import cartopy.crs as ccrs
import xarray as xr

#%%
# Config and params

SCENARIO      = 'present_day'      # 'present_day' or 'fully_built'

WINTER_MONTHS = [11, 12, 1, 2, 3]  # extended winter (NDJFM)
RL_PCT        = 90                 # upper-tail percentile for relative RL events (%)
RL_ABS_FLOOR  = 0.0                # absolute unserved-energy threshold (RL > 0 MW)
MAX_GAP       = 1                  # events separated by < 2 days are pooled
MIN_DURATION  = 3                  # min event duration (days)
CAP_VINTAGE   = 2024               # IRENA reference year

COUNTRIES = {                      # Natural Earth ADMIN name -> code
    "France"        : "FR",
    "Belgium"       : "BE",
    "Netherlands"   : "NL",
    "Germany"       : "DE",
    "Denmark"       : "DK",
    "United Kingdom": "GB",
    "Ireland"       : "IE",
}

# supply catalogue track to compare present-day RL against
SUPPLY_TRACK = 'IRENA_Relative_10pct'

# I/O
CF_DAILY_DIR      = Path('./../Results/CF_daily')
SUPPLY_DIR        = Path('./../Results/DF_supply')          # 03 outputs
DEMAND_PATH       = Path('./../Results/demand/demand_weather.parquet')  # 04 output — CONFIRM PATH
OUT_DIR           = Path('./../Results/DF_RL')
IC_PATH           = Path('./../Data/IRENA_IC/IC.csv')
POT_PATH          = Path('./../Data/Hu_IC/CF_info_grid.csv')
COUNTRIES_SHP     = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
EEZ_SHP           = Path('./../Data/EEZ/World_EEZ_v12_20231025/eez_v12.shp')

OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Load installed capacity (IRENA), max-installable potential (Hu), and daily CFs

# IRENA national installed capacity (MW), indexed by country code
IC = pd.read_csv(IC_PATH, index_col=0)
IC.index = IC.index.map({
    'France': 'FR', 'Belgium': 'BE', 'Netherlands': 'NL', 'Germany': 'DE',
    'Denmark': 'DK', 'United Kingdom': 'GB', 'Ireland': 'IE',
})

# Hu maximum-installable potential (gridded)
pot_raw = pd.read_csv(POT_PATH)[['lon', 'lat',
                                 'potential_onshore', 'potential_offshore', 'potential_PV']]
ds_pot = pot_raw.set_index(['lat', 'lon']).to_xarray()
ds_pot = ds_pot.rename({'lat': 'latitude', 'lon': 'longitude',
                        'potential_onshore': 'pot_on',
                        'potential_offshore': 'pot_off',
                        'potential_PV': 'pot_pv'})

# daily capacity factors
CF_wind_dly  = xr.open_zarr(CF_DAILY_DIR / 'CF_wind_daily.zarr',
                            consolidated=True)['CF_wind_dly']
CF_solar_dly = xr.open_zarr(CF_DAILY_DIR / 'CF_solar_24h_daily.zarr',
                            consolidated=True)['CF_solar_24h_dly']

# align Hu grid to ERA5 grid (reindex only, no interpolation)
pot_grid = ds_pot.reindex(latitude=CF_wind_dly.latitude,
                          longitude=CF_wind_dly.longitude,
                          fill_value=0.0)

cf_times = pd.DatetimeIndex(CF_wind_dly.time.values)
print(f"CF grid loaded: {len(cf_times)} days "
      f"({cf_times[0].date()} -> {cf_times[-1].date()})")

#%%
# Build country spatial masks (onshore + coastal offshore)

def build_masks(template_da, land_shp_path, eez_shp_path):
    land_gdf = gpd.read_file(str(land_shp_path))
    eez_gdf  = gpd.read_file(str(eez_shp_path))

    lons, lats = np.meshgrid(template_da.longitude.values, template_da.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())

    all_land    = land_gdf.union_all()
    is_any_land = (shapely.distance(all_land, pts) == 0.0).reshape(lats.shape)

    def _to_da(mask_array):
        return xr.DataArray(mask_array,
                            coords={'latitude': template_da.latitude,
                                    'longitude': template_da.longitude},
                            dims=['latitude', 'longitude'])

    masks_on, masks_off = {}, {}
    for name, code in COUNTRIES.items():
        land_geom = land_gdf.loc[land_gdf['ADMIN'] == name].union_all()
        if code == 'FR':
            land_geom = land_geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))  # Corsica
        masks_on[code] = _to_da(shapely.within(pts, land_geom).reshape(lats.shape))

        eez_geom = eez_gdf.loc[eez_gdf['SOVEREIGN1'] == name].union_all()
        if code == 'GB':
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 55.0, -10.0, 60.0))  # Rockall
        if code == 'DK':
            eez_geom = eez_geom.difference(shapely.geometry.box(-75.0, 58.0, -10.0, 85.0))  # Greenland
            eez_geom = eez_geom.difference(shapely.geometry.box(-15.0, 59.0, 0.0, 65.0))    # Faroe
        in_eez = shapely.within(pts, eez_geom).reshape(lats.shape)
        masks_off[code] = _to_da(in_eez & ~is_any_land)

    return masks_on, masks_off

masks_on, masks_off = build_masks(CF_wind_dly.isel(time=0), COUNTRIES_SHP, EEZ_SHP)
mask_land_sea = {code: (masks_on[code] | masks_off[code]) for code in COUNTRIES.values()}
print("Masks built.")
#%%
# Build present-day capacity: distribute IRENA national MW by Hu potential ratios

cap_on  = xr.zeros_like(pot_grid['pot_on'])
cap_off = xr.zeros_like(pot_grid['pot_off'])
cap_pv  = xr.zeros_like(pot_grid['pot_pv'])

for code in COUNTRIES.values():
    m = mask_land_sea[code]

    # country-summed Hu potential per technology (ratio denominators)
    sum_on  = float(pot_grid['pot_on'].where(m, 0.0).sum())
    sum_off = float(pot_grid['pot_off'].where(m, 0.0).sum())
    sum_pv  = float(pot_grid['pot_pv'].where(m, 0.0).sum())

    # IRENA national installed capacity (MW)
    ic_on, ic_off, ic_pv = IC.loc[code, ['Onshore Wind (MW)', 'Offshore Wind (MW)', 'Solar (MW)']]

    # allocate national MW by within-country Hu ratio
    if sum_on  > 0: cap_on  = cap_on  + ic_on  * pot_grid['pot_on'].where(m, 0.0)  / sum_on
    if sum_off > 0: cap_off = cap_off + ic_off * pot_grid['pot_off'].where(m, 0.0) / sum_off
    if sum_pv  > 0: cap_pv  = cap_pv  + ic_pv  * pot_grid['pot_pv'].where(m, 0.0)  / sum_pv

    if sum_off == 0 and ic_off > 0:
        print(f"  WARNING {code}: IRENA offshore {ic_off:.0f} MW but no offshore Hu potential — dropped.")

print("Present-day gridded capacity built (MW).")
print(f"  Total distributed capacity: {float((cap_on + cap_off + cap_pv).sum()):,.0f} MW")
#%%
# Country-level potential supply (Supply_max, MW)

if SCENARIO == 'fully_built':
    # 03 already computed this: Hu absolute MW summed per country
    supply_df = pd.read_parquet(SUPPLY_DIR / 'capgen_pot_absolute.parquet')
    supply_df.index = pd.DatetimeIndex(supply_df.index)
    print("Fully-built supply loaded from 03 (capgen_pot_absolute).")
else:
    gen_grid = CF_wind_dly * (cap_on + cap_off) + CF_solar_dly * cap_pv   # (time, lat, lon), MW

    supply_max = {code: gen_grid.where(mask_land_sea[code]).sum(['latitude', 'longitude']).values
                  for code in COUNTRIES.values()}
    supply_df = pd.DataFrame(supply_max, index=cf_times)
    print("Present-day supply (MW) built for all countries.")

print(supply_df.mean().round(0).to_string())
#%%
#
# Load weather-driven demand and compute residual load (RL = Demand - Supply)

demand_df = pd.read_parquet(DEMAND_PATH)          # daily counterfactual load (MW)
demand_df.index = pd.DatetimeIndex(demand_df.index)

common_dates = supply_df.index.intersection(demand_df.index)
common_ctry  = [c for c in COUNTRIES.values() if c in demand_df.columns and c in supply_df.columns]

rl_df = demand_df.loc[common_dates, common_ctry] - supply_df.loc[common_dates, common_ctry]

print(f"RL computed: {len(common_ctry)} countries, {len(common_dates)} days "
      f"({common_dates.min().date()} -> {common_dates.max().date()})")
print("Mean RL (MW):"); print(rl_df.mean().round(0).to_string())
#%%
# Winter (NDJFM) masks, aligned to the RL series

times  = rl_df.index
months = times.month.values
years  = times.year.values

winter_id = np.where(months >= 11, years + 1, years)
is_ndjfm  = np.isin(months, WINTER_MONTHS)

winters_with_nov = set(winter_id[months == 11])
winters_with_mar = set(winter_id[months == 3])
valid_winters    = sorted(winters_with_nov & winters_with_mar)

winter_ok = is_ndjfm & np.isin(winter_id, valid_winters)
n_winters = len(valid_winters)

print(f"Complete winters: {n_winters} ({valid_winters[0]} -> {valid_winters[-1]})")

#%%
# Event detection and characterization functions (upper-tail: RL exceedances)

def merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """Pool Trues separated by <= max_gap; drop merged runs shorter than min_duration."""
    if not np.any(flag):
        return np.zeros_like(flag, dtype=bool)

    T, runs, in_run, start = len(flag), [], False, None
    for t in range(T):
        if flag[t] and not in_run:
            start, in_run = t, True
        elif not flag[t] and in_run:
            runs.append([start, t - 1]); in_run = False
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

def detect_events_1d(x, mask_valid, thr_pct=None, abs_thr=None, tail='upper',
                     max_gap=1, min_duration=3):
    """Threshold on masked winter values; flag exceedances (upper) or shortfalls (lower)."""
    wv = x[mask_valid]
    if np.all(np.isnan(wv)):
        return None, np.nan, np.nan

    sigma = np.nanstd(wv)
    thr   = abs_thr if abs_thr is not None else np.nanpercentile(wv, thr_pct)

    raw_flag = ((x > thr) if tail == 'upper' else (x < thr)) & mask_valid
    evt_mask = merge_and_filter_runs_1d(raw_flag, max_gap, min_duration)
    return evt_mask, thr, sigma

def characterize_events_1d(evt_mask, x, thr, sigma, tail='upper'):
    """Per-event severity. Gap days (below thr for upper tail) contribute 0."""
    rows = []
    lab, num = label(evt_mask)
    for k in range(1, num + 1):
        idx = np.where(lab == k)[0]
        seg = x[idx[0]: idx[-1] + 1]

        exceed = np.clip(seg - thr, 0, None) if tail == 'upper' else np.clip(thr - seg, 0, None)

        rows.append(dict(
            start_idx  = int(idx[0]),
            end_idx    = int(idx[-1]),
            duration   = int(len(seg)),
            peak_val   = float(np.nanmax(seg)),          # peak RL (MW)
            mean_val   = float(np.nanmean(seg)),
            severity_S = float(np.sum(exceed) / sigma),  # σ-normalized summed exceedance
            excess_sum = float(np.sum(exceed)),          # MW·days above threshold
        ))
    return rows

#%%
# Run RL event detection (two tracks)

def run_detection(df_series, track_name, threshold_type, thr_val):
    all_events = []
    for code in df_series.columns:
        x = df_series[code].values
        if threshold_type == 'percentile':
            evt_mask, thr, sigma = detect_events_1d(x, winter_ok, thr_pct=thr_val, tail='upper',
                                                    max_gap=MAX_GAP, min_duration=MIN_DURATION)
        else:  # absolute
            evt_mask, thr, sigma = detect_events_1d(x, winter_ok, abs_thr=thr_val, tail='upper',
                                                    max_gap=MAX_GAP, min_duration=MIN_DURATION)

        for r in characterize_events_1d(evt_mask, x, thr, sigma, tail='upper'):
            s_idx, e_idx = r.pop('start_idx'), r.pop('end_idx')
            all_events.append(dict(country=code, track=track_name,
                                   start_date=times[s_idx], end_date=times[e_idx],
                                   winter=int(winter_id[s_idx]), threshold=float(thr), **r))
    return pd.DataFrame(all_events)

events_rl_rel = run_detection(rl_df, 'RL_Relative_90pct',   'percentile', RL_PCT)
events_rl_abs = run_detection(rl_df, 'RL_Absolute_unserved', 'absolute',   RL_ABS_FLOOR)
master_rl_df  = pd.concat([events_rl_rel, events_rl_abs], ignore_index=True)

print(master_rl_df.groupby('track').agg(n=('duration', 'size'),
                                        mean_dur=('duration', 'mean')).to_string())
#%%
# Annual aggregations (zero-filled)

annual_records = []
for track_name, df_track in master_rl_df.groupby('track'):
    agg = (df_track.groupby(['country', 'winter'])
           .agg(n_events=('duration', 'size'), total_dur=('duration', 'sum'),
                mean_dur=('duration', 'mean'), max_dur=('duration', 'max'),
                total_S=('severity_S', 'sum'), total_excess=('excess_sum', 'sum'))
           .reset_index())

    full_idx = pd.MultiIndex.from_product([common_ctry, valid_winters],
                                          names=['country', 'winter'])
    agg = agg.set_index(['country', 'winter']).reindex(full_idx, fill_value=0).reset_index()
    agg.loc[agg['n_events'] == 0, ['mean_dur', 'max_dur']] = np.nan
    agg['track'] = track_name
    annual_records.append(agg)

annual_df = pd.concat(annual_records, ignore_index=True)
print(annual_df.groupby('track')[['n_events', 'total_S']].sum().to_string())

#%%
# Supply-only vs Residual-Load event overlap (central empirical result)

supply_events = pd.read_parquet(SUPPLY_DIR / 'df_supply_events.parquet')
supply_rel = supply_events[supply_events['track'] == SUPPLY_TRACK]
rl_rel     = master_rl_df[master_rl_df['track'] == 'RL_Relative_90pct']

winter_dates = times[winter_ok]

def event_day_bool(ev_df, index):
    out = pd.DataFrame(False, index=index, columns=common_ctry)
    for _, r in ev_df.iterrows():
        if r['country'] in out.columns:
            out.loc[(index >= r['start_date']) & (index <= r['end_date']), r['country']] = True
    return out

sup_bool = event_day_bool(supply_rel, winter_dates)
rl_bool  = event_day_bool(rl_rel, winter_dates)

overlap_rows = []
for code in common_ctry:
    a, b = sup_bool[code].values, rl_bool[code].values
    both, union = int((a & b).sum()), int((a | b).sum())
    overlap_rows.append(dict(
        country=code,
        supply_days=int(a.sum()), rl_days=int(b.sum()),
        both=both, supply_only=int((a & ~b).sum()), rl_only=int((b & ~a).sum()),
        jaccard=(both / union) if union else 0.0,
        frac_rl_in_supply=(both / b.sum()) if b.sum() else np.nan,   # of RL days, share supply also caught
        frac_supply_in_rl=(both / a.sum()) if a.sum() else np.nan,   # of supply days, share grid-relevant
    ))
overlap_df = pd.DataFrame(overlap_rows)
print(overlap_df.round(2).to_string(index=False))
print(f"\nDomain-mean Jaccard: {overlap_df['jaccard'].mean():.2f} | "
      f"mean RL-day recall by supply: {overlap_df['frac_rl_in_supply'].mean():.2f}")
#%%
# Sanity check: DE residual load with detected events

sample = 'FR'
t0, t1 = pd.Timestamp('2016-10-01'), pd.Timestamp('2019-04-01')
m = (times >= t0) & (times <= t1)
thr = np.nanpercentile(rl_df[sample].values[winter_ok], RL_PCT)

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(times[m], rl_df[sample].values[m], color='k', lw=1, label='Residual load (MW)')
ax.axhline(thr, color='#d62728', ls='--', lw=1.5, label=f'{RL_PCT}th pct ({thr:,.0f} MW)')

for yr in range(t0.year, t1.year + 1):
    ax.axvspan(pd.Timestamp(f'{yr}-11-01'), pd.Timestamp(f'{yr+1}-03-31'),
               color='gray', alpha=0.1, label='NDJFM' if yr == t0.year else "")
for _, r in rl_rel[(rl_rel['country'] == sample) &
                   (rl_rel['start_date'] >= t0) & (rl_rel['start_date'] <= t1)].iterrows():
    ax.axvspan(r['start_date'], r['end_date'], color='#d62728', alpha=0.30)

ax.set_title(f"{sample} — Residual Load Events ({SCENARIO})", fontweight='bold', loc='left')
ax.set_ylabel("Residual load (MW)")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
h, l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
plt.tight_layout(); plt.show()

#%%
# Overlap figure + save outputs

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(overlap_df))
ax.bar(x, overlap_df['both'],        label='Both (supply & RL)', color='#4c72b0')
ax.bar(x, overlap_df['supply_only'], bottom=overlap_df['both'],
       label='Supply-only (not grid-relevant)', color='#dd8452')
ax.bar(x, overlap_df['rl_only'],     bottom=overlap_df['both'] + overlap_df['supply_only'],
       label='RL-only (missed by supply)', color='#c44e52')
ax.set_xticks(x); ax.set_xticklabels(overlap_df['country'])
ax.set_ylabel("Event-days"); ax.set_title("Supply-only vs Residual-Load event days",
                                           fontweight='bold', pad=10)
ax.grid(axis='y', ls='--', alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()

rl_df.to_parquet(OUT_DIR / f'rl_series_{SCENARIO}.parquet')
supply_df.to_parquet(OUT_DIR / f'supply_{SCENARIO}.parquet')
master_rl_df.to_parquet(OUT_DIR / f'rl_events_{SCENARIO}.parquet')
annual_df.to_parquet(OUT_DIR / f'rl_annual_{SCENARIO}.parquet')
overlap_df.to_parquet(OUT_DIR / f'rl_supply_overlap_{SCENARIO}.parquet')
print(f"\nSaved to {OUT_DIR}")

#%%
# Check 1: RL/supply overlap vs van der Wiel (~21% co-occurrence expected)

# frac_rl_in_supply = of RL-relevant days, share supply-screening also caught.
# Low = supply screening misses grid-relevant stress (the result we want).
print("Per-country RL–supply co-occurrence:")
print(overlap_df[['country', 'frac_rl_in_supply', 'jaccard']].round(2).to_string(index=False))
print(f"\nDomain mean frac_rl_in_supply: {overlap_df['frac_rl_in_supply'].mean():.2f} "
      f"(van der Wiel ballpark ~0.21)")
print(f"Domain mean Jaccard:           {overlap_df['jaccard'].mean():.2f}")
# %%

# Check 2: does present-day RL > 0 saturate? (expected: nearly every winter day)

n_winter_days = int(winter_ok.sum())
abs_evt = master_rl_df[master_rl_df['track'] == 'RL_Absolute_unserved']

# event-days flagged by the absolute track, per country
abs_days = {c: 0 for c in rl_df.columns}
for _, r in abs_evt.iterrows():
    span = (times >= r['start_date']) & (times <= r['end_date']) & winter_ok
    abs_days[r['country']] += int(span.sum())

sat = pd.Series(abs_days) / n_winter_days
print("Share of winter days with RL > 0 (unserved), per country:")
print(sat.round(2).to_string())
print(f"\nDomain mean saturation: {sat.mean():.2f}  "
      f"(near 1.0 = expected under present-day; well below = check IRENA totals)")

# raw sign check, independent of event-merging/duration filter
frac_positive = (rl_df.values[winter_ok] > 0).mean()
print(f"Fraction of all country-winter-days with RL > 0: {frac_positive:.2f}")
#%%
# Diagnostic: what drives RL variance, demand or supply? (winter days, per country)

wd = winter_ok
rows = []
for code in common_ctry:
    D = demand_df.loc[times, code].values[wd]
    S = supply_df.loc[times, code].values[wd]
    R = rl_df[code].values[wd]
    rows.append(dict(
        country   = code,
        std_D     = np.nanstd(D),
        std_S     = np.nanstd(S),
        std_ratio = np.nanstd(S) / np.nanstd(D),   # >1 : supply drives RL variance
        corr_D_S  = np.corrcoef(D, S)[0, 1],        # <0 : blocking couples hi-demand / lo-supply
        corr_RL_S = np.corrcoef(R, S)[0, 1],
        corr_RL_D = np.corrcoef(R, D)[0, 1],
    ))
var_df = pd.DataFrame(rows)
print(var_df.round(2).to_string(index=False))

#%%
# Does supply's share of RL variance explain the overlap gradient across countries?

m = var_df.merge(overlap_df[['country', 'frac_rl_in_supply']], on='country')
r = np.corrcoef(m['std_ratio'], m['frac_rl_in_supply'])[0, 1]
print(m[['country', 'std_ratio', 'corr_RL_S', 'frac_rl_in_supply']].round(2).to_string(index=False))
print(f"\ncorr(std_ratio, frac_rl_in_supply) across countries = {r:.2f}")
#%%
# Cross-check: do RL events track demand peaks (FR) or supply lulls (DK)?

dem_aligned = demand_df.loc[times, common_ctry]
events_dem  = run_detection(dem_aligned, 'DEM_90pct', 'percentile', 90)  # high-demand upper tail
dem_bool    = event_day_bool(events_dem, winter_dates)

rows = []
for code in common_ctry:
    b = rl_bool[code].values
    n = b.sum()
    rows.append(dict(
        country      = code,
        rl_in_supply = (sup_bool[code].values & b).sum() / n if n else np.nan,
        rl_in_demand = (dem_bool[code].values & b).sum() / n if n else np.nan,
    ))
print(pd.DataFrame(rows).round(2).to_string(index=False))
# %%
