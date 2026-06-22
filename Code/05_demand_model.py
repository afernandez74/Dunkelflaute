"""
05_demand_model.py — temperature-driven demand (Bloomfield 2021 / van der Wiel 2019)
Fit HDD/CDD regression on ENTSO-E load, retain weather-dependent part only,
apply across the full ERA5 record. Per country (FR, BE, NL, DE).
"""
#%%
import numpy as np
import pandas as pd 
import xarray as xr 
import statsmodels.api as sm

#%%
HDD_T, CDD_T = 15.5, 22.0          # EEA / Bloomfield thresholds (deg C)
TRAIN = ("2015-01-01", "2019-12-31")
REF   = "2017-07-01"                # date at which the background level is frozen

#%%

def country_temp(t2m, mask, pop_w=None):
    """Daily-mean, (pop-weighted) country-mean 2 m temperature [deg C]."""
    t = (t2m - 273.15).resample(time="1D").mean().where(mask)
    w = (pop_w.where(mask) if pop_w is not None else xr.ones_like(t.isel(time=0)))
    return (t * w).sum(["latitude", "longitude"]) / w.sum()

def fit_demand(load, temp):
    df = pd.concat([load.rename("load"), temp.rename("T")], axis=1).dropna()
    df["HDD"] = (HDD_T - df["T"]).clip(lower=0)
    df["CDD"] = (df["T"] - CDD_T).clip(lower=0)
    df["trend"] = (df.index - df.index[0]).days
    dow = pd.get_dummies(df.index.dayofweek, prefix="d", drop_first=True).astype(float)
    dow.index = df.index
    X = sm.add_constant(pd.concat([df[["trend", "HDD", "CDD"]], dow], axis=1))
    return sm.OLS(df["load"], X).fit(), df.index[0]

def weather_demand(fit, t0, temp_full):
    """Synthetic weather-only demand across the full record (drop dow + trend var.)."""
    p = fit.params
    a0 = p["const"] + p["trend"] * (pd.Timestamp(REF) - t0).days   # frozen background
    HDD = (HDD_T - temp_full).clip(min=0); CDD = (temp_full - CDD_T).clip(min=0)
    return a0 + p["HDD"] * HDD + p["CDD"] * CDD
