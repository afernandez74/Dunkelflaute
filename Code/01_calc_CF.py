#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_calc_CF.py
=============
Calculate hourly wind and solar capacity factors (CFs) from the ERA5 Zarr
store produced by the ARCO-ERA5 pipeline (00_dwnld_arco_era5.py +
00b_ETL_nc_to_zarr.py).

Scientific background
---------------------
Wind CF — Vestas V90-2.0 MW turbine power curve (Brown et al. 2021):
    An 8th-degree polynomial maps 100 m wind speed to electrical power output.
    The turbine does not generate below cut-in (3 m/s) or above cut-out
    (25 m/s), and produces rated power (2 000 kW) between rated speed and
    cut-out.  CF_wind = WP / P_rated.

Solar CF — Temperature-adjusted PV efficiency model (Brown et al. 2021;
           originally Bett & Thornton 2016):
    CF_solar = η_rel(t2m, G) × G / G_STC
    where the relative efficiency η_rel accounts for:
      - Temperature derating (module heats above ambient; β coefficient)
      - Non-linear irradiance response (log terms c1, c2)
    The output is forced to zero when irradiance ≤ 1 W/m² (night/deep clouds).

ARCO-ERA5 ssrd note
-------------------
ARCO-ERA5 stores surface_solar_radiation_downwards as accumulated J/m² per
hourly analysis step.  Dividing by 3 600 s converts to an average W/m² over
that hour, which is the physically correct input to the PV model.

Inputs
------
  $ERA5_dat/temps_df/processed/temps_df_cleaned.zarr
      Variables: u100, v100 (m/s); ssrd (J/m²/hour); t2m (K)

Outputs
-------
  ./../Results/CF_wind/CF_wind_hrly.zarr   — hourly wind CF  [0–1]
  ./../Results/CF_solar/CF_solar_hrly.zarr — hourly solar CF [0–~0.2]

Next step
---------
  02_calc_CF_daily.py  — daily aggregation of hourly CFs

References
----------
  Brown et al. (2021) Renewable Energy Resources, Technology and Data: A
      review of the literature, doi:10.1016/j.energy.2021.120034
  Bett & Thornton (2016) doi:10.1002/met.1548

@author: afer
"""

#%%
# Imports

import os
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#%%
# ─── Physical / model constants ───────────────────────────────────────────────

# ── Vestas V90-2.0 MW wind turbine (Brown et al. 2021) ──
P_RATED   = 2000.0   # Rated power [kW]
WS_CUTIN  =    3.0   # Cut-in wind speed [m/s]
WS_RATED  =   13.0   # Rated wind speed [m/s]
WS_CUTOUT =   25.0   # Cut-out wind speed [m/s]

# ── Solar PV model (Brown et al. 2021 / Bett & Thornton 2016) ──
ALPHA   =  1.2e-13   # Efficiency temperature coefficient (K⁻¹) — negligible
BETA    = -4.6e-3    # Efficiency temperature coefficient (K⁻¹) — dominant term
C1      =  0.033     # Irradiance log term
C2      = -0.0092    # Irradiance log-squared term
T_NOCT  =   48.0     # Normal operating cell temperature [°C]
T_0     =   25.0     # Ambient reference temperature for NOCT [°C]
T_STC   =   25.0     # Cell temperature at standard test conditions [°C]
G_0     =  800.0     # Reference irradiance for NOCT definition [W/m²]
G_STC   = 1000.0     # Irradiance at standard test conditions [W/m²]
G_MIN   =    1.0     # Minimum irradiance threshold for active generation [W/m²]

# ── Zarr output chunking ──
# Full time axis per (lat, lon) tile → optimal for time-series reads in
# subsequent scripts (SWDI/SSDI calculation, event detection).
LAT_CHUNK = 10
LON_CHUNK = 10


#%%
# ─── Load ERA5 Zarr store ─────────────────────────────────────────────────────
# Produced by 00b_ETL_nc_to_zarr.py; contains u100, v100, ssrd, t2m (and u10,
# v10) with short names, longitude wrapped to [−180, 180], and sorted coords.

era5_dat = os.environ.get("ERA5_dat")
if era5_dat is None:
    raise EnvironmentError(
        "Environment variable 'ERA5_dat' is not set.\n"
        "Add it to ~/.bashrc:  export ERA5_dat=/path/to/era5_data"
    )

zarr_path = Path(era5_dat) / "temps_df" / "processed" / "temps_df_cleaned.zarr"
print(f"Loading ERA5 Zarr store: {zarr_path}")

ds = xr.open_zarr(zarr_path, consolidated=True, chunks={})

print(f"Dataset loaded — variables : {list(ds.data_vars)}")
print(f"  time      : {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}"
      f"  ({ds.sizes['time']} steps)")
print(f"  latitude  : {float(ds.latitude.min()):.2f} → {float(ds.latitude.max()):.2f}"
      f"  ({ds.sizes['latitude']} cells)")
print(f"  longitude : {float(ds.longitude.min()):.2f} → {float(ds.longitude.max()):.2f}"
      f"  ({ds.sizes['longitude']} cells)")

# Pull out only the variables needed for CF calculation
u100 = ds['u100']   # 100 m zonal wind component [m/s]
v100 = ds['v100']   # 100 m meridional wind component [m/s]
ssrd = ds['ssrd']   # Surface solar radiation downwards [J/m²/hour, accumulated]
t2m  = ds['t2m']    # 2 m air temperature [K]


#%%
# ─── Wind capacity factor — Vestas V90-2.0 MW power curve ────────────────────
#
# Step 1: wind speed magnitude at 100 m hub height from u and v components.
# Step 2: piecewise power curve applied with xr.where (evaluated lazily):
#   ws < cut-in  → 0 kW   (below generation threshold)
#   cut-in ≤ ws ≤ rated  → 8th-degree polynomial fit to power curve
#   rated < ws ≤ cut-out → rated power (constant plateau)
#   ws > cut-out → 0 kW   (turbine shut-down for safety)
# Step 3: divide by rated power to get dimensionless CF in [0, 1].
#
# The polynomial coefficients reproduce the V90 power curve from Brown et al.
# (2021), Table 1, fitted to manufacturer data.

def power_curve_polynomial(ws):
    """
    8th-degree polynomial power curve for the Vestas V90-2.0 MW turbine.

    Valid between cut-in (3 m/s) and rated speed (13 m/s).  Outside this
    range the piecewise xr.where logic handles the constant regions.

    Parameters
    ----------
    ws : xr.DataArray
        Wind speed at 100 m hub height [m/s].

    Returns
    -------
    wp : xr.DataArray
        Power output [kW].

    Reference
    ---------
    Brown et al. (2021), doi:10.1016/j.energy.2021.120034
    """
    return (  634.228
            - 1248.5    * ws
            +  999.57   * ws**2
            -  426.224  * ws**3
            +  105.617  * ws**4
            -   15.4587 * ws**5
            +    1.3223 * ws**6
            -    0.0609186 * ws**7
            +    0.00116265 * ws**8)


# 100 m wind speed magnitude [m/s]
ws_100 = np.sqrt(u100**2 + v100**2)

# Piecewise power output [kW]
WP = xr.where(ws_100 > WS_CUTOUT,  0.0,
     xr.where(ws_100 > WS_RATED,   P_RATED,
     xr.where(ws_100 < WS_CUTIN,   0.0,
              power_curve_polynomial(ws_100))))

# Capacity factor [dimensionless, 0–1]
CF_wind = (WP / P_RATED).rename('CF_wind')

CF_wind.attrs.update({
    'long_name'   : 'Wind capacity factor — Vestas V90-2.0 MW at 100 m hub height',
    'units'       : 'dimensionless',
    'valid_range' : [0, 1],
    'turbine'     : 'Vestas V90-2.0 MW; P_rated=2000 kW; cut-in=3 m/s; '
                    'rated=13 m/s; cut-out=25 m/s',
    'reference'   : 'Brown et al. (2021), doi:10.1016/j.energy.2021.120034',
})

print(f"\nCF_wind computed (lazy) — shape: {CF_wind.shape}")


#%%
# ─── Solar capacity factor — temperature-corrected PV model ──────────────────
#
# The model follows Brown et al. (2021) / Bett & Thornton (2016) and consists
# of two parts:
#
#   1. Module temperature (NOCT approximation):
#        T_mod = t2m_C + (T_NOCT − T_0) × G / G_0
#
#   2. Relative efficiency accounting for temperature derating and irradiance
#      non-linearity:
#        η_rel = (1 + α·ΔT) × (1 + c1·ln(G') + c2·(ln G')² + β·ΔT)
#      where ΔT = T_mod − T_STC,  G' = G / G_STC.
#
#   3. Capacity factor:
#        CF_solar = η_rel × G / G_STC
#
# Hours with G ≤ G_MIN (1 W/m²) are set to CF_solar = 0 to avoid log(0)
# and to reflect the physical reality that panels are not generating at night.
#
# ssrd is stored in ARCO-ERA5 as accumulated J/m² per hourly step;
# dividing by 3600 gives the mean irradiance over that hour in W/m².

def calc_module_temperature(t2m_c, irr):
    """
    PV module temperature from ambient temperature and irradiance (NOCT model).

    Parameters
    ----------
    t2m_c : xr.DataArray   Ambient air temperature [°C]
    irr   : xr.DataArray   Plane-of-array irradiance [W/m²]

    Returns
    -------
    T_mod : xr.DataArray   Module temperature [°C]
    """
    return t2m_c + (T_NOCT - T_0) * (irr / G_0)


def calc_relative_efficiency(t2m_c, irr):
    """
    Relative PV efficiency accounting for temperature derating and irradiance
    non-linearity (Brown et al. 2021 / Bett & Thornton 2016).

    Only call this function for grid cells / hours where irr > G_MIN; passing
    irr ≤ 0 produces log(0) or log(negative) errors.

    Parameters
    ----------
    t2m_c : xr.DataArray   Ambient air temperature [°C]
    irr   : xr.DataArray   Plane-of-array irradiance [W/m²], must be > 0

    Returns
    -------
    eta_rel : xr.DataArray   Relative efficiency [dimensionless, ≈1 under STC]
    """
    T_mod   = calc_module_temperature(t2m_c, irr)
    dT      = T_mod - T_STC          # temperature deviation from STC [°C]
    G_prime = irr / G_STC            # normalised irradiance [dimensionless]
    log_G   = np.log(G_prime)         # natural log of normalised irradiance

    return (1 + ALPHA * dT) * (1 + C1 * log_G + C2 * log_G**2 + BETA * dT)


# Unit conversions
irr   = ssrd / 3600.0     # J/m² (accumulated per hour) → W/m² (hourly mean)
t2m_c = t2m - 273.15      # K → °C

# Apply model; force CF = 0 at night / below detection threshold
CF_solar = xr.where(
    irr > G_MIN,
    calc_relative_efficiency(t2m_c, irr) * irr / G_STC,
    0.0
).rename('CF_solar')

CF_solar.attrs.update({
    'long_name'   : 'Solar PV capacity factor — temperature-corrected PV model',
    'units'       : 'dimensionless',
    'valid_range' : [0, 1],
    'model'       : ('Brown et al. 2021 / Bett & Thornton 2016; '
                     'G_STC=1000 W/m²; T_NOCT=48°C; G_MIN=1 W/m²'),
    'reference'   : ('Brown et al. (2021), doi:10.1016/j.energy.2021.120034; '
                     'Bett & Thornton (2016), doi:10.1002/met.1548'),
    'ssrd_note'   : 'ssrd converted from J/m²/hour to W/m² by dividing by 3600',
})

print(f"CF_solar computed (lazy) — shape: {CF_solar.shape}")


#%%
# ─── Diagnostic: spatial maps of time-mean CF ────────────────────────────────
# Compute the time-averaged CF at every grid cell (triggers Dask computation).
# Expected spatial patterns:
#   Wind  — higher CF offshore and at high latitudes; local minima over land
#   Solar — monotonically increasing southward; near-zero above ~55°N in winter
# These maps serve as a basic sanity check before running the full pipeline.

print("\nComputing spatial mean CFs (triggers Dask — may take a moment)...")
CF_wind_mean  = CF_wind.mean(dim='time').compute()
CF_solar_mean = CF_solar.mean(dim='time').compute()

lat_min = float(ds.latitude.min())
lat_max = float(ds.latitude.max())
lon_min = float(ds.longitude.min())
lon_max = float(ds.longitude.max())
plot_region = [lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5]

fig, axes = plt.subplots(
    1, 2, figsize=(16, 7),
    subplot_kw={'projection': ccrs.LambertConformal(central_longitude=5, central_latitude=48)}
)

for ax, data, title, cmap, cbar_label in zip(
    axes,
    [CF_wind_mean,  CF_solar_mean],
    ['Time-mean hourly CF_wind (Vestas V90 @ 100 m)',
     'Time-mean hourly CF_solar (PV model)'],
    ['Blues',        'YlOrRd'],
    ['CF_wind [ ]',  'CF_solar [ ]'],
):
    ax.set_extent(plot_region, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle='--', edgecolor='gray')
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', alpha=0.4)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightblue', alpha=0.4)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
              cbar_kwargs={'label': cbar_label, 'shrink': 0.75})
    ax.set_title(title, fontsize=11, fontweight='bold')

plt.suptitle('ERA5 hourly capacity factors — time mean over full record',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


#%%
# ─── Diagnostic: sample grid cell distributions ──────────────────────────────
# Box-and-whisker plot and empirical distribution at the central grid cell.
# Wind: bimodal distribution (mode at zero = calm conditions + generating mode)
# Solar: tri-modal (night zeros dominant; low winter peak; high summer peak)

lat_idx = ds.sizes['latitude']  // 2
lon_idx = ds.sizes['longitude'] // 2
central_lat = float(ds.latitude.values[lat_idx])
central_lon = float(ds.longitude.values[lon_idx])

print(f"\nSample grid cell: ({central_lat:.2f}°N, {central_lon:.2f}°E)")

CF_wind_cell  = CF_wind.isel(latitude=lat_idx, longitude=lon_idx).compute()
CF_solar_cell = CF_solar.isel(latitude=lat_idx, longitude=lon_idx).compute()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f'Hourly CF distributions — sample grid cell ({central_lat:.2f}°N, {central_lon:.2f}°E)',
    fontsize=13, fontweight='bold'
)

for row, (da, name, color) in enumerate([
    (CF_wind_cell,  'CF_wind',  'steelblue'),
    (CF_solar_cell, 'CF_solar', 'goldenrod'),
]):
    vals = da.values.ravel()

    # Box and whisker
    axes[row, 0].boxplot(
        vals, vert=True, patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.7),
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
    )
    axes[row, 0].set_ylabel(f'{name} [ ]')
    axes[row, 0].set_title('Box & Whisker')

    # Histogram + KDE
    axes[row, 1].hist(vals, bins=50, density=True, alpha=0.6, color=color,
                      edgecolor='black', linewidth=0.3, label='Histogram')
    sns.kdeplot(vals, ax=axes[row, 1], color='black', linewidth=1.5, label='KDE')
    axes[row, 1].set_xlabel(f'{name} [ ]')
    axes[row, 1].set_ylabel('Density')
    axes[row, 1].set_title('Empirical Distribution')
    axes[row, 1].legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#%%
# ─── Diagnostic: mean seasonal cycle at central grid cell ────────────────────
# DOY mean ± 1 std envelope over the full record.
# Wind: modest winter peak due to stronger extratropical cyclone activity.
# Solar: strong summer peak; near-zero mean in winter months above ~50°N.

# Add DOY coordinate and compute DOY statistics for the central grid cell
CF_wind_doy  = CF_wind_cell.assign_coords(doy=CF_wind_cell.time.dt.dayofyear).groupby('doy')
CF_solar_doy = CF_solar_cell.assign_coords(doy=CF_solar_cell.time.dt.dayofyear).groupby('doy')

# Trim to DOY 1–365; DOY 366 exists in leap years but has fewer samples
wind_doy_mean  = CF_wind_doy.mean().values[:365]
wind_doy_std   = CF_wind_doy.std().values[:365]
solar_doy_mean = CF_solar_doy.mean().values[:365]
solar_doy_std  = CF_solar_doy.std().values[:365]

doy_x = np.arange(1, 366)
month_starts = [pd.Timestamp(year=2001, month=m, day=1).dayofyear for m in range(1, 13)]
month_names  = [calendar.month_abbr[m] for m in range(1, 13)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'Mean-year CF seasonal cycle (DOY climatology) — '
    f'({central_lat:.2f}°N, {central_lon:.2f}°E)',
    fontsize=13, fontweight='bold'
)

for ax, mean, std, name, color in zip(
    axes,
    [wind_doy_mean,  solar_doy_mean],
    [wind_doy_std,   solar_doy_std],
    ['CF_wind  (V90 @ 100 m)', 'CF_solar (PV model)'],
    ['steelblue',              'goldenrod'],
):
    ax.plot(doy_x, mean, color=color, linewidth=1.5, label='DOY mean')
    ax.fill_between(doy_x, mean - std, mean + std, color=color, alpha=0.3, label='± 1 std')
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_xlabel('Month')
    ax.set_ylabel('CF [ ]')
    ax.set_title(name)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%%
# ─── Save hourly CFs to Zarr ─────────────────────────────────────────────────
# Output Zarr stores: one per CF variable, consumed by 02_calc_CF_daily.py.
# Rechunk to full time axis per (lat, lon) tile before writing — this chunking
# layout is optimal for the time-series reads performed by downstream scripts.

results_base   = Path('./../Results')
out_wind_zarr  = results_base / 'CF_wind'  / 'CF_wind_hrly.zarr'
out_solar_zarr = results_base / 'CF_solar' / 'CF_solar_hrly.zarr'

out_wind_zarr.parent.mkdir(parents=True, exist_ok=True)
out_solar_zarr.parent.mkdir(parents=True, exist_ok=True)

# Convert to Dataset so to_zarr works cleanly and produces a named variable
CF_wind_ds  = CF_wind.chunk({'time': -1, 'latitude': LAT_CHUNK, 'longitude': LON_CHUNK}).to_dataset()
CF_solar_ds = CF_solar.chunk({'time': -1, 'latitude': LAT_CHUNK, 'longitude': LON_CHUNK}).to_dataset()

print(f"\nSaving CF_wind  → {out_wind_zarr}  (may take several minutes…)")
CF_wind_ds.to_zarr(out_wind_zarr, mode='w', consolidated=True)
print(f"  Done.")

print(f"Saving CF_solar → {out_solar_zarr}  (may take several minutes…)")
CF_solar_ds.to_zarr(out_solar_zarr, mode='w', consolidated=True)
print(f"  Done.")

print("\nHourly CFs saved successfully.")
print(f"  CF_wind  : {out_wind_zarr}")
print(f"  CF_solar : {out_solar_zarr}")
print("\nNext step: run 02_calc_CF_daily.py")
