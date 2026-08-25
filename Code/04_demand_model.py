#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_demand_model.py — temperature-driven demand (Bloomfield 2021 / van der Wiel 2019)
====================================================================================
Fit an HDD/CDD regression on ENTSO-E observed load over a clean TRAIN window,
retain only the weather-dependent part with the system background level frozen
at REF vintage, then project weather-only demand across the full ERA5 record.

Assumptions (v1)
----------------
- Homogeneous population density: country-mean temperature is an unweighted
  spatial mean over onshore (land) cells.
- Excludes public holidays during TRAIN to prevent HDD coefficient depression.

@author: afer
"""
#%%
# Imports

import os
import holidays
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import geopandas as gpd
import shapely

import matplotlib.pyplot as plt

#%%
# Config params

HDD_T, CDD_T = 15.5, 22.0             # EEA / Bloomfield thresholds (deg C)
TRAIN = ("2015-01-01", "2019-12-31")  # ENTSO-E clean window; excludes COVID-2020
REF   = "2017-07-01"                  # vintage at which background level is frozen

COUNTRIES = {                         # Natural Earth ADMIN name -> code
    "France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE",
    "Denmark": "DK", "United Kingdom": "GB", "Ireland": "IE",
}

# Domain matched to 03_DF_supply_ID.py so demand and VRE production share the
# same country footprints. 

LAT_SLICE = slice(41, 62)
LON_SLICE = slice(-14, 16)

DATA          = Path("./../Data/")
RESULTS       = Path("./../Results/")
COUNTRIES_SHP = Path("~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp").expanduser()
LOAD_DIR      = DATA / "entsoe_load"

# ERA5_DAT = os.environ.get("ERA5_dat")
ERA5_DAT = Path("/Volumes/AlejoED/WELCOME/")

#%%
# ====================================================
# Load ERA5 t2m and subset domain 
# ====================================================


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
# Build onshore (land) country masks
def build_masks(template, shp_path):
    """Boolean onshore mask per country"""
    gdf = gpd.read_file(str(shp_path))
    lons, lats = np.meshgrid(template.longitude.values, template.latitude.values)
    
    # Create GeoSeries of grid points for faster spatial querying
    pts = gpd.GeoSeries(gpd.points_from_xy(lons.ravel(), lats.ravel()))
    masks = {}
    # loop over countries
    for country, code in COUNTRIES.items():
        geom = gdf.loc[gdf["ADMIN"] == country].union_all()
        if code == "FR":   
            geom = geom.difference(shapely.geometry.box(8.5, 41.3, 9.6, 43.1))
            
        inside = pts.within(geom).values.reshape(lats.shape) 
        
        masks[code] = xr.DataArray(
            inside,
            coords={"latitude": template.latitude, "longitude": template.longitude},
            dims=["latitude", "longitude"],
        )
    return masks

masks = build_masks(t2m.isel(time=0), COUNTRIES_SHP)

#%%
# =============================================
#  Country-mean daily temperature [°C] 
# =============================================
print("\nCalculating country-mean daily temperatures...")

country_hourly = {c: t2m.where(mask).mean(["latitude", "longitude"]) for c, mask in masks.items()}

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
# ===============================================
# ENTSO-E observed load (daily mean MW), restricted to TRAIN 
# ===============================================

load = {}
for f in sorted(LOAD_DIR.glob("load_*.parquet")):
    cc = f.stem.split("_")[1]                    
    s  = pd.read_parquet(f)["load"]
    load[cc] = s

load = pd.DataFrame(load).sort_index().loc[TRAIN[0]:TRAIN[1]]
load.index = pd.to_datetime(load.index)

#%%
# =========================================================
#  Demand model: weather-driven demand functions (fit and isolate)
# =========================================================
def get_country_holidays(years, country_code):
    """Fetch holidays to mask out anomalous low-load days."""
    try:
        # Standardize codes (e.g., GB instead of UK for the holidays library)
        country_holidays = holidays.country_holidays(country_code, years=years)
        return list(country_holidays.keys())
    except NotImplementedError:
        return []

def fit_demand(load_s, temp_s, cc_code):
    """OLS: load ~ const + trend + HDD + CDD + day-of-week dummies."""
    df = pd.concat([load_s.rename("load"), temp_s.rename("T")], axis=1).dropna()
    
    # Remove public holidays to prevent skewing the HDD coefficient
    holiday_dates = pd.to_datetime(get_country_holidays(df.index.year.unique(), cc_code))
    df = df[~df.index.isin(holiday_dates)]
    
    df["HDD"]   = (HDD_T - df["T"]).clip(lower=0)
    df["CDD"]   = (df["T"] - CDD_T).clip(lower=0)
    df["trend"] = (df.index - df.index[0]).days
    dow = pd.get_dummies(df.index.dayofweek, prefix="d", drop_first=True).astype(float)
    dow.index = df.index

    X = sm.add_constant(pd.concat([df[["trend", "HDD", "CDD"]], dow], axis=1))
    fit = sm.OLS(df["load"], X).fit()
    return fit, df.index[0]

def weather_demand(fit, t0, temp_full):
    """Weather-only demand: frozen background + HDD/CDD."""
    p = fit.params
    dow_cols = [c for c in p.index if c.startswith("d_")]
    dow_mean = p[dow_cols].sum() / 7.0                     
    a0  = p["const"] + dow_mean + p["trend"] * (pd.Timestamp(REF) - t0).days
    
    HDD = (HDD_T - temp_full).clip(lower=0)
    CDD = (temp_full - CDD_T).clip(lower=0)
    
    # .values strips xarray/pandas metadata to ensure clean alignment
    return a0 + p["HDD"] * HDD.values + p["CDD"] * CDD.values

#%%
# =========================================================
# Calculate weather-driven demand model (execution)
# =========================================================

temp_train = temp.loc[TRAIN[0]:TRAIN[1]]
demand, fits = {}, {}

print("\nFitting regression models...")
for cc in temp.columns:
    if cc not in load.columns:
        continue
    fit, t0 = fit_demand(load[cc], temp_train[cc], cc)
    fits[cc] = fit
    demand[cc] = weather_demand(fit, t0, temp[cc])
    print(f"{cc}: R²={fit.rsquared:.3f} | HDD={fit.params['HDD']:.0f} | CDD={fit.params['CDD']:.0f} MW/°C (n={int(fit.nobs)})")

demand = pd.DataFrame(demand, index=temp.index)
demand.index.name = "date"
coef = pd.DataFrame({cc: f.params for cc, f in fits.items()}).T

#%%
# ===============================================
# Diagnostic: observed vs weather-only demand over TRAIN (FR and NL) ────────
# ===============================================

# Added sharex=True to link the x-axes for easier visual comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, cc in zip(axes, ["FR", "NL"]):
    # 'load' is already restricted to the TRAIN window from earlier in the script
    obs = load[cc]
    
    # Slice the pre-calculated demand dataframe to match the TRAIN window
    modelled = demand[cc].loc[TRAIN[0]:TRAIN[1]]

    # Plotting
    ax.plot(obs.index, obs.values, color="#333", lw=0.7, alpha=0.7, 
            label="Observed Load")
    ax.plot(modelled.index, modelled.values, color="#d62728", lw=1.2, alpha=0.9,
            label="Weather-only Demand (Frozen Vintage)")
    
    # Formatting
    ax.set_title(f"{cc} — Observed vs. Weather-only Demand (TRAIN)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, ls="--")

axes[-1].set_xlabel("Date", fontsize=11)
plt.tight_layout()
plt.show()

#%%
# ===============================================
# Save results
# ===============================================

out = RESULTS / "demand"
out.mkdir(parents=True, exist_ok=True)

demand.to_parquet(out / "demand_weather.parquet")
coef.to_csv(out / "demand_model_coefficients.csv")

print(f"\nSaved → {out / 'demand_weather.parquet'}  "
      f"({demand.index[0].date()} → {demand.index[-1].date()}, "
      f"{demand.shape[1]} countries)")
print(f"Saved → {out / 'demand_model_coefficients.csv'}")

# %%