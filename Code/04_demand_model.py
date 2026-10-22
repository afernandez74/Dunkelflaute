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
import rasterio
import matplotlib.pyplot as plt
import rioxarray
from rasterio.enums import Resampling


#%%
# Config params

HDD_T, CDD_T = 15.5, 22.0             # EEA / Bloomfield thresholds (deg C)
TRAIN = ("2015-01-01", "2019-12-31")  # ENTSO-E clean window; excludes COVID-2020
REF   = "2017-07-01"                  # vintage at which background level is frozen

COUNTRIES = {                         # Natural Earth ADMIN name -> code
    "France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE",
    "Denmark": "DK", "United Kingdom": "UK", "Ireland": "IE",
}

WINTER_MONTHS = [11, 12, 1, 2, 3]

def select_ndjfm(obj):
    return obj.loc[obj.index.month.isin(WINTER_MONTHS)]

# Domain matched to 03_DF_supply_ID.py so demand and VRE production share the
# same country footprints. 

LAT_SLICE = slice(41, 62)
LON_SLICE = slice(-14, 16)

DATA          = Path("./../Data/")
RESULTS       = Path("./../Results/")
COUNTRIES_SHP = Path("~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp").expanduser()
LOAD_DIR      = DATA / "entsoe_load"
POP_FILE = DATA / "POP" / "gpw_v4_population_density_rev11_2015_15_min.tif"

# ERA5_DAT = os.environ.get("ERA5_dat")
ERA5_DAT = Path("/Volumes/AlejoED/WELCOME/")

#%%
# Load ERA5 t2m and subset domain. 
ds  = xr.open_zarr(Path(ERA5_DAT) / "df_dat/processed/df_dat_cleaned.zarr",
                   consolidated=True)

#%%
# Subset t2m data and Load population dataset and align to ERA5 data
t2m = ds["t2m"].sel(latitude=LAT_SLICE, longitude=LON_SLICE)
t2m = t2m.drop_vars(['level'])
t2m = t2m.rio.write_crs("EPSG: 4326")


print("ERA5 t2m subset:")
print(f"  time : {str(t2m.time.values[0])[:10]} → {str(t2m.time.values[-1])[:10]}"
      f"  ({t2m.sizes['time']} steps)")
print(f"  lat  : {float(t2m.latitude.min()):.2f} → {float(t2m.latitude.max()):.2f}"
      f"  ({t2m.sizes['latitude']} cells)")
print(f"  lon  : {float(t2m.longitude.min()):.2f} → {float(t2m.longitude.max()):.2f}"
      f"  ({t2m.sizes['longitude']} cells)")

# inspect and clean population dataset

with rasterio.open(POP_FILE) as src:
    print(f"Population raster: {POP_FILE.name}")
    print(f"Bounds: {src.bounds}")
    print(f"Resolution: {src.res}")
    print(f"CRS: {src.crs}")
    print(f"NoData: {src.nodata}")

pop = rioxarray.open_rasterio(POP_FILE).squeeze("band", drop = True)
pop_clean =pop.where(pop>=0).where(pop != pop.rio.nodata, np.nan)

# downscale pop to match era5 
pop_era5 = (pop_clean.rio.reproject_match(t2m, resampling=Resampling.average, nodata=np.nan).rename({"y": "latitude", "x":"longitude"}))

pop_era5 = pop_era5.assign_coords(
    latitude=t2m.latitude,
    longitude=t2m.longitude
)
#%%
# Build onshore (land) country masks
def build_masks(template, shp_path):
    """Boolean onshore mask per country"""
    gdf = gpd.read_file(str(shp_path)).to_crs("EPSG:4326")
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
#  Country-mean daily temperature [°C] 
# =============================================

# Calculate population density weights
lat_rad = np.deg2rad(t2m.latitude) 
area_weights = np.cos(lat_rad) # area weights based on latitude

pop_weights = (pop_era5 * area_weights).fillna(0) #population weights

print("\nCalculating country-mean daily temperatures...")

country_hourly = {c: t2m.where(mask).weighted(pop_weights).mean(dim=["latitude", "longitude"])
 for c, mask in masks.items()}

temp = xr.Dataset(country_hourly)
temp = (
    temp
    .coarsen(time=24, boundary="trim", coord_func="min")
    .mean() - 273.15
)  # K -> °C, daily mean

#%%
# ENTSO-E observed load (daily mean MW), restricted to TRAIN 
# ===============================================

load = {}
for cc in COUNTRIES.values():
    f = LOAD_DIR / f"load_{cc}.parquet"
    s  = pd.read_parquet(f)["load"]
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    load[cc] = s

dates = pd.date_range(TRAIN[0], TRAIN[1], freq = "D")
load = pd.DataFrame(load).reindex(dates)
load.index = pd.to_datetime(load.index)

temp = temp.compute().to_dataframe()
temp = temp[list(COUNTRIES.values())]

temp.index = pd.to_datetime(temp.index)
temp.index.name = "date"
temp = temp.sort_index()

# Match the calibration dates used for observed load
temp_train = temp.reindex(dates)

print(temp.head())
#%%
# Calendar controls
# ===============================================

def get_country_holidays(years, country_code):
    """National public holidays used as a regression control."""

    holiday_code = country_code

    country_holidays = holidays.country_holidays(
        holiday_code,
        years=[int(year) for year in years],
    )

    return list(country_holidays.keys())


def special_day_indicator(index, country_code):
    idx = pd.DatetimeIndex(index).normalize()

    public = pd.DatetimeIndex(
        get_country_holidays(idx.year.unique(), country_code)
    ).normalize()

    christmas_period = (
        ((idx.month == 12) & (idx.day >= 24))
        | ((idx.month == 1) & (idx.day <= 2))
    )

    return (idx.isin(public) | christmas_period).astype(float)

#%%
# Demand model: fit and isolate weather dependence
# ===============================================

def demand_inputs(df, include_hdd=True):
    """Regression inputs, with Monday as the reference weekday."""

    columns = ["trend", "special_day"]

    if include_hdd:
        columns.append("HDD")

    X = df[columns].copy()

    for day in range(1, 7):
        X[f"d_{day}"] = (df.index.dayofweek == day).astype(float)

    X = sm.add_constant(X, has_constant="add")

    return X.astype(float)


def fit_demand(load_s, temp_s, cc_code):
    """NDJFM OLS: load ~ trend + HDD + special days + weekday."""

    df = pd.concat(
        [load_s.rename("load"), temp_s.rename("T")],
        axis=1,
    ).sort_index()

    df = select_ndjfm(df).copy()

    if np.isinf(df.to_numpy()).any():
        raise ValueError(f"{cc_code}: infinite load or temperature")

    df = df.dropna()

    if df.empty:
        raise ValueError(f"{cc_code}: no matching winter observations")

    if (df["load"] <= 0).any():
        raise ValueError(f"{cc_code}: non-positive national load; inspect data")

    df["HDD"] = (HDD_T - df["T"]).clip(lower=0)
    df["trend"] = (df.index - pd.Timestamp(REF)).days / 365.25
    df["special_day"] = special_day_indicator(df.index, cc_code)

    X = demand_inputs(df)

    if len(df) <= X.shape[1]:
        raise ValueError(f"{cc_code}: insufficient fitting observations")

    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError(f"{cc_code}: regression inputs are linearly dependent")

    fit = sm.OLS(df["load"], X, missing="raise").fit()

    return fit, df


def weather_demand(fit, temp_full):
    """Frozen REF background + temperature-driven heating demand."""

    p = fit.params

    # Average weekday effect over a seven-day week.
    # Monday's effect is zero because it is the reference category.
    dow_mean = p[[f"d_{day}" for day in range(1, 7)]].sum() / 7.0

    # Trend is zero at REF; special-day effect is excluded.
    background = p["const"] + dow_mean

    HDD = (HDD_T - temp_full).clip(lower=0)

    return background + p["HDD"] * HDD

#%%
# Fit final models using the full calibration period
# ===============================================

demand, fits = {}, {}
fit_samples = {}
diagnostic_rows = []

print("\nFitting regression models...")

for cc in COUNTRIES.values():
    fit, fit_df = fit_demand(load[cc], temp_train[cc], cc)

    fits[cc] = fit
    fit_samples[cc] = fit_df
    demand[cc] = weather_demand(fit, temp[cc])

    if fit.params["HDD"] <= 0:
        print(f"CHECK {cc}: non-positive heating-demand coefficient")

    diagnostic_rows.append({
        "country": cc,
        "n_observations": int(fit.nobs),
        "n_hdd_days": int((fit_df["HDD"] > 0).sum()),
        "max_hdd_C": float(fit_df["HDD"].max()),
        "hdd_coefficient_MW_per_C": float(fit.params["HDD"]),
        "trend_MW_per_year": float(fit.params["trend"]),
        "train_rmse_MW": float(np.sqrt(np.mean(fit.resid**2))),
        "train_r2": float(fit.rsquared),
        "train_adjusted_r2": float(fit.rsquared_adj),
    })

demand = pd.DataFrame(demand, index=temp.index)
demand.index.name = "date"

coef = pd.DataFrame({
    cc: fit.params for cc, fit in fits.items()
}).T

diagnostics = (
    pd.DataFrame(diagnostic_rows)
    .set_index("country")
    .sort_index()
)

if not np.isfinite(demand.to_numpy()).all():
    raise ValueError("Projected demand contains missing or infinite values")

if (select_ndjfm(demand) <= 0).any().any():
    raise ValueError("Non-positive projected winter demand; inspect model")

print(diagnostics.round(3))
#%%
# Validation: hold out winter 2018–2019
# ===============================================

VAL_START = "2018-11-01"
VAL_END = "2019-03-31"

validation_rows = []
validation_predictions = {}

for cc in COUNTRIES.values():
    df = fit_samples[cc]

    training = df.loc[df.index < VAL_START].copy()
    testing = df.loc[VAL_START:VAL_END].copy()

    if training.empty or testing.empty:
        raise ValueError(f"{cc}: empty validation training/test sample")

    X_train = demand_inputs(training)
    X_test = demand_inputs(testing)

    if np.linalg.matrix_rank(X_train.to_numpy()) < X_train.shape[1]:
        raise ValueError(f"{cc}: validation regression is rank-deficient")

    # Full model.
    fit_test = sm.OLS(
        training["load"], X_train, missing="raise"
    ).fit()

    predicted = fit_test.predict(X_test)

    # Same observations and controls, but without temperature.
    X_train_base = demand_inputs(training, include_hdd=False)
    X_test_base = demand_inputs(testing, include_hdd=False)

    fit_base = sm.OLS(
        training["load"], X_train_base, missing="raise"
    ).fit()

    predicted_base = fit_base.predict(X_test_base)

    error = predicted - testing["load"]
    error_base = predicted_base - testing["load"]

    rmse = np.sqrt(np.mean(error**2))
    rmse_base = np.sqrt(np.mean(error_base**2))

    # Coldest 10% of the available held-out observations.
    cold = testing["T"] <= testing["T"].quantile(0.10)

    validation_rows.append({
        "country": cc,
        "n_test_days": len(testing),
        "n_cold_days": int(cold.sum()),
        "rmse_with_HDD_MW": rmse,
        "rmse_without_HDD_MW": rmse_base,
        "rmse_improvement_pct": (
            100 * (1 - rmse / rmse_base) if rmse_base > 0 else np.nan
        ),
        "cold_day_bias_MW": error.loc[cold].mean(),
        "cold_day_rmse_MW": np.sqrt(np.mean(error.loc[cold]**2)),
        "validation_fit_HDD_MW_per_C": fit_test.params["HDD"],
        "full_fit_HDD_MW_per_C": fits[cc].params["HDD"],
    })

    validation_predictions[cc] = pd.DataFrame({
        "observed": testing["load"],
        "predicted": predicted,
        "predicted_without_HDD": predicted_base,
    })

validation = (
    pd.DataFrame(validation_rows)
    .set_index("country")
    .sort_index()
)

print("\nHeld-out winter validation:")
print(validation.round(2))
#%%
# Observed and modelled winter demand
# ===============================================

cc = "DE"
year = 2018

df = fit_samples[cc].loc[
    fit_samples[cc].index.year == year
].copy()

if df.empty:
    raise ValueError(f"No fitting observations for {cc} in {year}")

predicted = fits[cc].predict(demand_inputs(df))

# Reindex to preserve the summer gap in the plotted lines.
dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

plot_data = pd.DataFrame({
    "Observed": df["load"],
    "Modelled": predicted,
}).reindex(dates)

fig, ax = plt.subplots(figsize=(13, 4))

ax.plot(
    plot_data.index, plot_data["Observed"] / 1000,
    color="0.25", lw=1.2, label="Observed",
)
ax.plot(
    plot_data.index, plot_data["Modelled"] / 1000,
    color="tab:red", lw=1.2, label="Modelled — full regression",
)

ax.set_title(f"{cc} — observed and modelled winter demand, {year}")
ax.set_ylabel("Daily mean demand (GW)")
ax.set_xlabel("Date")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
# %%

# Save demand model results
# ===============================================

out = RESULTS / "demand_population_weighted"
out.mkdir(parents=True, exist_ok=True)

# Daily mean demand [MW]: dates × countries
demand_out = demand.loc[:, list(COUNTRIES.values())].copy()
demand_out.index = pd.DatetimeIndex(demand_out.index, name="date")
demand_out = demand_out.sort_index()

# Daily population-weighted temperature [°C]: dates × countries
temp_out = temp.loc[:, list(COUNTRIES.values())].copy()
temp_out.index = pd.DatetimeIndex(temp_out.index, name="date")
temp_out = temp_out.sort_index()

# Country-level coefficients and diagnostics
coef_out = coef.copy()
coef_out.index.name = "country"

diagnostics_out = diagnostics.copy()
diagnostics_out.index.name = "country"

validation_out = validation.copy()
validation_out.index.name = "country"

# Observations and full-model predictions used in fitting
fitted_rows = []

for cc, df in fit_samples.items():
    fitted = df.copy()
    fitted["modelled_load_MW"] = fits[cc].predict(demand_inputs(df))
    fitted["residual_MW"] = fitted["load"] - fitted["modelled_load_MW"]
    fitted["country"] = cc
    fitted.index.name = "date"

    fitted_rows.append(fitted.reset_index())

fitted_out = (
    pd.concat(fitted_rows, ignore_index=True)
    .set_index(["country", "date"])
    .sort_index()
)

# Held-out predictions, separate from in-sample fitted values
validation_predictions_out = pd.concat(
    validation_predictions,
    names=["country", "date"],
).sort_index()

outputs = {
    "demand_weather.parquet": demand_out,
    "temperature_population_weighted.parquet": temp_out,
    "demand_model_coefficients.parquet": coef_out,
    "demand_model_diagnostics.parquet": diagnostics_out,
    "demand_model_validation.parquet": validation_out,
    "demand_model_fitted_values.parquet": fitted_out,
    "demand_model_validation_predictions.parquet": validation_predictions_out,
}

for filename, df in outputs.items():
    path = out / filename
    df.to_parquet(path)

    print(f"Saved → {path} | {len(df):,} rows")

print(
    f"\nDemand period: {demand_out.index[0].date()} → "
    f"{demand_out.index[-1].date()}"
)
print("Demand units: daily mean MW")
print("Model application: November–March")