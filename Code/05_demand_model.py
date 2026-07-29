#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_demand_model.py — temperature-driven demand (Bloomfield 2021 / van der Wiel 2019)
====================================================================================
Fit an HDD/CDD regression on ENTSO-E observed load over a clean TRAIN window,
retain only the weather-dependent part with the system background level frozen
at REF vintage, then project weather-only demand across the full ERA5 record.

Weather-only demand (not a demand *forecast*): the frozen-vintage design removes
the secular growth/efficiency trend and the weekday cycle, leaving the component
of demand that responds to temperature. This is the demand term that later feeds
the residual-load arm (RL = demand - VRE production).

Assumptions (v1)
----------------
- Homogeneous population density: country-mean temperature is an unweighted
  spatial mean over onshore (land) cells. Bloomfield 2021 uses unweighted
  country-mean temperature, so this is consistent, not a simplification to flag.
- Onshore land cells only (demand is driven by populated land temperature);
  the offshore buffer used by the supply arm is irrelevant here.
- Country masks reconciled with 03_DF_supply_ID.py: Natural Earth ADMIN field,
  full domain covering GB/IE/DK, ascending-latitude slice.

Inputs
------
  $ERA5_dat/df_dat/processed/df_dat_cleaned.zarr   (t2m, full record)
  ./../Data/entsoe_load/load_*.parquet             (ENTSO-E observed daily load)
  ~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp

Output
------
  ./../Results/demand/demand_weather.parquet   (date x country, daily-mean MW)
  ./../Results/demand/demand_model_coefficients.csv

@author: afer
"""
#%%
# Imports

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import geopandas as gpd
import regionmask
import shapely

import matplotlib.pyplot as plt


#%%
# ─── Config ───────────────────────────────────────────────────────────────────

HDD_T, CDD_T = 15.5, 22.0             # EEA / Bloomfield thresholds (deg C)
TRAIN = ("2015-01-01", "2019-12-31")  # ENTSO-E clean window; excludes COVID-2020
REF   = "2017-07-01"                  # vintage at which background level is frozen

COUNTRIES = {                         # Natural Earth ADMIN name -> code
    "France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE",
    "Denmark": "DK", "United Kingdom": "GB", "Ireland": "IE",
}

# Domain matched to 03_DF_supply_ID.py so demand and VRE production share the
# same country footprints. Ascending-latitude slice (the store is sorted
# ascending by 00b_ETL). Wide enough to cover Scotland, Ireland, Denmark.
LAT_SLICE = slice(41, 62)
LON_SLICE = slice(-14, 16)

DATA          = Path("./../Data/")
RESULTS       = Path("./../Results/")
COUNTRIES_SHP = Path("~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp").expanduser()
LOAD_DIR      = DATA / "entsoe_load"

ERA5_DAT = os.environ.get("ERA5_dat")
if ERA5_DAT is None:
    raise EnvironmentError("Environment variable 'ERA5_dat' is not set.")


#%%
# ─── Load ERA5 t2m and subset domain ──────────────────────────────────────────

ds  = xr.open_zarr(Path(ERA5_DAT) / "df_dat/processed/df_dat_cleaned.zarr",
                   consolidated=True)
t2m = ds["t2m"].sel(latitude=LAT_SLICE, longitude=LON_SLICE)

print("ERA5 t2m subset:")
print(f"  time : {str(t2m.time.values[0])[:10]} → {str(t2m.time.values[-1])[:10]}"
      f"  ({t2m.sizes['time']} steps)")
print(f"  lat  : {float(t2m.latitude.min()):.2f} → {float(t2m.latitude.max()):.2f}"
      f"  ({t2m.sizes['latitude']} cells)")
print(f"  lon  : {float(t2m.longitude.min()):.2f} → {float(t2m.longitude.max()):.2f}"
      f"  ({t2m.sizes['longitude']} cells)")

if t2m.sizes["latitude"] == 0 or t2m.sizes["longitude"] == 0:
    raise ValueError("Empty spatial subset — check LAT_SLICE / LON_SLICE ordering "
                     "against the store's coordinate sort order.")


#%%
# ─── Build onshore (land) country masks — ADMIN field, matches Track B ────────

def build_masks(template, shp_path):
    """Boolean onshore mask per country (True inside the land polygon)."""
    gdf = gpd.read_file(str(shp_path))
    lons, lats = np.meshgrid(template.longitude.values, template.latitude.values)
    pts = shapely.points(lons.ravel(), lats.ravel())

    masks = {}
    for country, code in COUNTRIES.items():
        geom = gdf.loc[gdf["ADMIN"] == country].union_all()
        if code == "FR":   # exclude Corsica — consistent with 03 / ENTSO-E FR
            geom = geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
        inside = (shapely.distance(geom, pts).reshape(lats.shape) == 0.0)
        masks[code] = xr.DataArray(
            inside,
            coords={"latitude": template.latitude, "longitude": template.longitude},
            dims=["latitude", "longitude"],
        )
    return masks

masks = build_masks(t2m.isel(time=0), COUNTRIES_SHP)
print("Onshore cells per country: " +
      ", ".join(f"{c}: {int(m.sum())}" for c, m in masks.items()))


#%%
# ─── Country-mean daily temperature [°C] ──────────────────────────────────────
# Average spatially first (collapses the grid), then resample to daily, then
# compute once. Unweighted spatial mean = homogeneous population assumption
# (Bloomfield 2021 uses unweighted country-mean temperature).

country_hourly = {}
for c, mask in masks.items():
    print(f"Preparing temperature series: {c}")
    country_hourly[c] = t2m.where(mask).mean(["latitude", "longitude"])

temp = xr.Dataset(country_hourly)
temp = (
    temp
    .coarsen(time=24, boundary="trim", coord_func="min")
    .mean() - 273.15
)  # K -> °C, daily mean
temp = temp.compute().to_dataframe()

temp.index = pd.to_datetime(temp.index)

print(f"\nCountry-mean T: {temp.index[0].date()} → {temp.index[-1].date()}, "
      f"{len(temp)} days")


#%%
# ─── ENTSO-E observed load (daily mean MW), restricted to TRAIN ───────────────

load = {}
for f in sorted(LOAD_DIR.glob("load_*.parquet")):
    cc = f.stem.split("_")[1]                    # load_NL.parquet -> NL
    s  = pd.read_parquet(f)["load"]
    s.index = pd.to_datetime(s.index)
    load[cc] = s

load = pd.DataFrame(load).sort_index().loc[TRAIN[0]:TRAIN[1]]
print(f"Observed load (TRAIN): {load.index[0].date()} → {load.index[-1].date()} "
      f"({len(load)} days); countries: {list(load.columns)}")


#%%
# ─── Demand model: fit on TRAIN, project weather-only over full record ────────

def fit_demand(load_s, temp_s):
    """OLS: load ~ const + trend + HDD + CDD + day-of-week dummies."""
    df = pd.concat([load_s.rename("load"), temp_s.rename("T")], axis=1).dropna()
    df["HDD"]   = (HDD_T - df["T"]).clip(lower=0)
    df["CDD"]   = (df["T"] - CDD_T).clip(lower=0)
    df["trend"] = (df.index - df.index[0]).days
    dow = pd.get_dummies(df.index.dayofweek, prefix="d", drop_first=True).astype(float)
    dow.index = df.index
    X = sm.add_constant(pd.concat([df[["trend", "HDD", "CDD"]], dow], axis=1))
    fit = sm.OLS(df["load"], X).fit()
    return fit, df.index[0]


def weather_demand(fit, t0, temp_full):
    """Weather-only demand: frozen background (avg weekday, REF vintage) + HDD/CDD."""
    p = fit.params
    dow_cols = [c for c in p.index if c.startswith("d_")]
    dow_mean = p[dow_cols].sum() / 7.0                     # avg day, not Monday-ref
    a0  = p["const"] + dow_mean + p["trend"] * (pd.Timestamp(REF) - t0).days
    HDD = (HDD_T - temp_full).clip(lower=0)
    CDD = (temp_full - CDD_T).clip(lower=0)
    return a0 + p["HDD"] * HDD + p["CDD"] * CDD


temp_train = temp.loc[TRAIN[0]:TRAIN[1]]
demand, fits = {}, {}

for cc in temp.columns:
    if cc not in load.columns:
        print(f"{cc}: no ENTSO-E load — skipped")
        continue
    fit, t0 = fit_demand(load[cc], temp_train[cc])
    fits[cc]   = fit
    demand[cc] = weather_demand(fit, t0, temp[cc])
    print(f"{cc}: R²={fit.rsquared:.3f}  "
          f"HDD={fit.params['HDD']:.0f}  CDD={fit.params['CDD']:.0f} MW/°C  "
          f"(n={int(fit.nobs)})")

demand = pd.DataFrame(demand)
demand.index.name = "date"
coef = pd.DataFrame({cc: f.params for cc, f in fits.items()}).T


#%%
#%%
# ─── Diagnostic: observed vs weather-only demand over TRAIN (FR and NL) ────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for ax, cc in zip(axes, ["FR", "NL"]):
    obs      = load[cc]
    _, t0    = fit_demand(load[cc], temp_train[cc])
    modelled = weather_demand(fits[cc], t0, temp_train[cc])

    ax.plot(obs.index, obs.values, color="#333", lw=0.7, label="observed load")
    ax.plot(modelled.index, modelled.values, color="#d62728", lw=0.9,
            label="weather-only demand (frozen vintage)")
    ax.set_title(f"{cc} — observed vs weather-only demand (TRAIN)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("MW")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, ls="--")

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()


#%%
# ─── Save ─────────────────────────────────────────────────────────────────────

out = RESULTS / "demand"
out.mkdir(parents=True, exist_ok=True)

demand.to_parquet(out / "demand_weather.parquet")
coef.to_csv(out / "demand_model_coefficients.csv")

print(f"\nSaved → {out / 'demand_weather.parquet'}  "
      f"({demand.index[0].date()} → {demand.index[-1].date()}, "
      f"{demand.shape[1]} countries)")
print(f"Saved → {out / 'demand_model_coefficients.csv'}")

# %%