#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_DF_RL.py
===========
Residual-load dunkelflaute detection (Otero'22), per country, NW Europe.
RL = demand - (CF_wind*IC_wind + CF_solar*IC_solar). Combines wind+solar CF.
Produces per-event catalogue and annual aggregates analogous to the CDHW outputs.

Inputs
------
  ./../Results/CF_daily/CF_wind_daily.zarr        (02_calc_CF_daily.py)
  ./../Results/CF_daily/CF_solar_24h_daily.zarr   (02_calc_CF_daily.py)
  ./../Results/demand/demand_weather.parquet      (05_demand_model.py; date x country, MW)
  ~/CDHW_ag/Data/countries_shp/ne_110m_admin_0_countries.shp

Outputs
-------
  ./../Results/DF_RL/df_events_idx.parquet   one row per event
  ./../Results/DF_RL/df_annual_idx.parquet   one row per (country, winter)
"""
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely.vectorized
from scipy.ndimage import label

# ─── Config ───────────────────────────────────────────────────────────────────
COUNTRIES = {"France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE"}

# Present-day installed capacity [MW] — FILL FROM ENTSO-E Transparency / IRENA
IC_WIND  = {"FR": 0.0, "BE": 0.0, "NL": 0.0, "DE": 0.0}
IC_SOLAR = {"FR": 0.0, "BE": 0.0, "NL": 0.0, "DE": 0.0}

WINTER_MONTHS = [11, 12, 1, 2, 3]   # extended winter (NDJFM)
INITIAL_PCT   = 90                  # Otero'22 moderate threshold (within-season)
SEVERE_PCT    = 98.6                # ~top 1.4% (Biewald best-fit) — for a severe tier
MAX_GAP       = 2                   # Otero pools events separated by <= 2 days
MIN_DURATION  = 3

RESULTS = Path("./../Results")
SHP = Path("~/CDHW_ag/Data/countries_shp/ne_110m_admin_0_countries.shp").expanduser()

#%%
# ─── Load daily CFs (lazy → compute country means only) ────────────────────────
CHUNKS = {"time": -1, "latitude": 20, "longitude": 20}
CF_wind  = xr.open_zarr(RESULTS / "CF_daily/CF_wind_daily.zarr",
                        consolidated=True, chunks=CHUNKS)["CF_wind_dly"]
CF_solar = xr.open_zarr(RESULTS / "CF_daily/CF_solar_24h_daily.zarr",
                        consolidated=True, chunks=CHUNKS)["CF_solar_24h_dly"]
CF_wind, CF_solar = xr.align(CF_wind, CF_solar, join="inner")

#%%
# ─── Country masks (uniform-within-country v1; hook for capacity weights) ──────
def build_country_masks(template, shp_path):
    """Boolean (lat,lon) mask per country code. Excludes Corsica from FR."""
    gdf = gpd.read_file(str(shp_path))
    lons, lats = np.meshgrid(template.longitude.values, template.latitude.values)
    corsica = (8.5, 41.3, 9.6, 43.1)  # bbox, drop from France
    masks = {}
    for name, code in COUNTRIES.items():
        geom = gdf[gdf["SOVEREIGNT"] == name].union_all()
        m = shapely.vectorized.contains(geom, lons, lats)
        if code == "FR":  # crude Corsica drop on the grid
            m &= ~((lons >= corsica[0]) & (lons <= corsica[2]) &
                   (lats >= corsica[1]) & (lats <= corsica[3]))
        masks[code] = xr.DataArray(
            m, coords={"latitude": template.latitude,
                       "longitude": template.longitude},
            dims=["latitude", "longitude"])
    return masks

def country_mean(da, mask, weights=None):
    """(Capacity-)weighted country-mean daily CF → pandas Series."""
    dm = da.where(mask)
    if weights is None:
        s = dm.mean(["latitude", "longitude"])
    else:
        w = weights.where(mask)
        s = (dm * w).sum(["latitude", "longitude"]) / w.sum(["latitude", "longitude"])
    return s.compute().to_pandas()

masks = build_country_masks(CF_wind.isel(time=0), SHP)
print(f"Land cells per country: " +
      ", ".join(f"{c}:{int(m.sum())}" for c, m in masks.items()))

cf_wind_df  = pd.DataFrame({c: country_mean(CF_wind,  masks[c]) for c in masks})
cf_solar_df = pd.DataFrame({c: country_mean(CF_solar, masks[c]) for c in masks})

#%%
# ─── Residual load [MW] : demand − (wind + solar production) ───────────────────
demand_df = pd.read_parquet(RESULTS / "demand/demand_weather.parquet")  # date x country, MW
demand_df.index = pd.to_datetime(demand_df.index)
demand_df, cf_wind_df, cf_solar_df = (
    demand_df.align(cf_wind_df, join="inner", axis=0)[0],
    demand_df.align(cf_wind_df, join="inner", axis=0)[1],
    cf_solar_df.reindex(demand_df.align(cf_wind_df, join="inner", axis=0)[0].index),
)

wind_prod  = cf_wind_df.mul(pd.Series(IC_WIND))
solar_prod = cf_solar_df.mul(pd.Series(IC_SOLAR))
RL_df = demand_df - wind_prod - solar_prod          # date x country, MW
RL_df["DOMAIN"] = RL_df.sum(axis=1)                 # copper-plate aggregate

#%%
# ─── Event detection + characterization ────────────────────────────────────────
def merge_and_filter(flag, max_gap, min_dur):
    """Pool runs separated by <= max_gap, drop runs < min_dur. Returns bool array."""
    if not flag.any():
        return np.zeros_like(flag, dtype=bool)
    runs, in_run, start = [], False, None
    for t, f in enumerate(flag):
        if f and not in_run:
            start, in_run = t, True
        elif not f and in_run:
            runs.append([start, t - 1]); in_run = False
    if in_run:
        runs.append([start, len(flag) - 1])
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = np.zeros_like(flag, dtype=bool)
    for s, e in merged:
        if e - s + 1 >= min_dur:
            out[s:e + 1] = True
    return out

def winter_label(ts):
    """Label Nov–Mar by the year of the January (Nov/Dec → next year's winter)."""
    return ts.year + 1 if ts.month >= 11 else ts.year

def detect_country(rl, code):
    win_mask = rl.index.month.isin(WINTER_MONTHS)
    win = rl[win_mask]
    thr   = np.nanpercentile(win.values, INITIAL_PCT)
    thr_s = np.nanpercentile(win.values, SEVERE_PCT)
    sigma = np.nanstd(win.values)

    raw  = (rl.values > thr) & win_mask                 # winter days above threshold
    evt  = merge_and_filter(raw, MAX_GAP, MIN_DURATION)

    rows, lab, n = [], *label(evt)
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        seg = rl.iloc[idx[0]:idx[-1] + 1]
        exceed = seg.values - thr                        # signed (pooled days may dip)
        if not (seg.values > thr_s).any():               # severity filter (≥1 severe day)
            severe = False
        else:
            severe = True
        rows.append(dict(
            country     = code,
            start_date  = seg.index[0],
            end_date    = seg.index[-1],
            duration    = len(seg),
            winter      = winter_label(seg.index[0]),
            peak_rl     = float(seg.max()),              # capacity-constrained need [MW]
            mean_rl     = float(seg.mean()),
            severity_S  = float((exceed / sigma).sum()), # Otero'22 standardized (net)
            deficit_GWh = float(np.clip(exceed, 0, None).sum() * 24 / 1000),  # energy need
            is_severe   = severe,
            thr_MW      = float(thr),
        ))
    return rows

events = [r for code in RL_df.columns for r in detect_country(RL_df[code], code)]
ev_df  = pd.DataFrame(events)
print(f"\nTotal events: {len(ev_df)}")
print(ev_df.groupby("country")[["duration", "severity_S", "deficit_GWh"]].mean())

#%%
# ─── Annual aggregates + save ──────────────────────────────────────────────────
n_winters = ev_df["winter"].nunique() or 1
annual = (ev_df.groupby(["country", "winter"])
          .agg(n_events=("duration", "size"),
               total_dur=("duration", "sum"),
               mean_dur=("duration", "mean"),
               total_S=("severity_S", "sum"),
               total_GWh=("deficit_GWh", "sum"))
          .reset_index())

# fill missing (country, winter) combinations with zeros for honest frequency trends
full_idx = pd.MultiIndex.from_product(
    [ev_df["country"].unique(), range(ev_df["winter"].min(), ev_df["winter"].max() + 1)],
    names=["country", "winter"])
annual = (annual.set_index(["country", "winter"]).reindex(full_idx, fill_value=0)
          .reset_index())

out = RESULTS / "DF_RL"; out.mkdir(parents=True, exist_ok=True)
ev_df.to_parquet(out / "df_events_idx.parquet")
annual.to_parquet(out / "df_annual_idx.parquet")
print(f"\nSaved → {out/'df_events_idx.parquet'} and df_annual_idx.parquet")
# %%