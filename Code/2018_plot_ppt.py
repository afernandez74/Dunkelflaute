"""
Representative 2018 normalised anomalies for NL.
For visual representation of case studies in a single critical year.
"""
#%%
# Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401 – registers the .rio accessor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d

#%%
# paths
ERA5_path = Path(os.environ["ERA5_dat"])
CF_path = Path("./../Results/CF_daily")
CDHW_path = Path("~/CDHW_ag/Results")

ssdi_path = CF_path / "CF_solar_24h_daily.zarr"
swdi_path = CF_path / "CF_wind_daily.zarr"

smdi_path = CDHW_path / "SMDI" / "smdi.zarr"
svdi_path = CDHW_path / "SVDI" / "svdi.zarr"

# Natural Earth shapefile
countries_path = Path("../../../CDHW_ag/Data/countries_shp/ne_110m_admin_0_countries.shp")

BASELINE_START = "1980"
BASELINE_END   = "2024"
YEAR           = "2018"

#%%
# Load full time series
print("Opening Zarr stores...")
ssdi = xr.open_zarr(ssdi_path, consolidated=True, chunks={})
swdi = xr.open_zarr(swdi_path, consolidated=True, chunks={})
smdi = xr.open_zarr(smdi_path, consolidated=True, chunks={})
svdi = xr.open_zarr(svdi_path, consolidated=True, chunks={})
countries_gdf = gpd.read_file(countries_path)

#%%

# Extract the data variable from each Dataset 
# Adjust variable names below if your Zarr stores use different keys
ssdi_da = ssdi[list(ssdi.data_vars)[0]]
swdi_da = swdi[list(swdi.data_vars)[0]]
smdi_da = smdi[list(smdi.data_vars)[0]]
svdi_da = svdi[list(svdi.data_vars)[0]]
 
#%%

# Clip to Netherlands and spatially average
nl_geom = (
    countries_gdf[countries_gdf["SOVEREIGNT"] == "Netherlands"]
    .dissolve()
)

def nl_mean(da: xr.DataArray) -> xr.DataArray:
    """Clip to NL boundary and return unweighted spatial mean (1-D time series)."""
    clipped = (
        da
        .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        .rio.write_crs("EPSG:4326")
        .rio.clip(nl_geom.geometry, nl_geom.crs, all_touched=True)
    )
    return clipped.mean(dim=["latitude", "longitude"])

print("Clipping and averaging over the Netherlands...")
ssdi_nl = nl_mean(ssdi_da)
swdi_nl = nl_mean(swdi_da)
smdi_nl = nl_mean(smdi_da)
svdi_nl = nl_mean(svdi_da)

#%%
# Select 2018 and compute 
print("Selecting 2018...")
ssdi_2018 = ssdi_nl.sel(time=YEAR).compute().values.astype(float)
swdi_2018 = swdi_nl.sel(time=YEAR).compute().values.astype(float)
smdi_2018 = smdi_nl.sel(time=YEAR).compute().values.astype(float)
svdi_2018 = svdi_nl.sel(time=YEAR).compute().values.astype(float)
#%%
# Shared daily datetime index (use any one variable — all share the same time axis)
dates = pd.DatetimeIndex(ssdi_nl.sel(time=YEAR).time.values)
 
# 30-day centred moving average 
WINDOW = 30   # days
 
def moving_avg(arr: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Centred uniform moving average; NaN-safe via pandas."""
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=window // 2)
        .mean()
        .values
    )

ssdi_ma = moving_avg(ssdi_2018)
swdi_ma = moving_avg(swdi_2018)
smdi_ma = moving_avg(smdi_2018)
svdi_ma = moving_avg(svdi_2018)
#%%
#  
#  Colours (matching slide deck) 
C_TEAL  = "#0A8A64"   # SMDI — soil moisture
C_CORAL = "#C14B2A"   # SVDI — temperature / VPD
C_WIND  = "#3A5FA0"   # SWDI — wind CF
C_SOLAR = "#C49A00"   # SSDI — solar CF
C_AMBER = "#C47D00"   # dunkelflaute window shading
C_NAVY  = "#1F3060"
C_GRAY  = "#888888"
 
ALPHA_DAILY = 0.18    # opacity for raw daily lines
ALPHA_SHADE = 0.15    # background window shading
 
# ── Highlight windows ─────────────────────────────────────────────────
jja_start  = pd.Timestamp("2018-06-01")
jja_end    = pd.Timestamp("2018-08-31")
dunk_start = pd.Timestamp("2018-04-23")
dunk_end   = pd.Timestamp("2018-05-07")
 
# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.2, 3.70), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
 
# Background event shading
ax.axvspan(jja_start,  jja_end,  alpha=ALPHA_SHADE,        color=C_TEAL,  zorder=0)
ax.axvspan(dunk_start, dunk_end, alpha=ALPHA_SHADE + 0.05, color=C_AMBER, zorder=0)
 
# Zero reference and ±1σ guides
ax.axhline(0,  color="#CCCCCC", linewidth=0.8, zorder=1)
ax.axhline( 1, color="#E2E2E2", linewidth=0.5, linestyle="--", zorder=1)
ax.axhline(-1, color="#E2E2E2", linewidth=0.5, linestyle="--", zorder=1)
 
# ── Daily lines (faint) ───────────────────────────────────────────────
ax.plot(dates, smdi_2018, color=C_TEAL,  linewidth=0.6, alpha=ALPHA_DAILY, zorder=2)
ax.plot(dates, svdi_2018, color=C_CORAL, linewidth=0.6, alpha=ALPHA_DAILY, zorder=2)
ax.plot(dates, swdi_2018, color=C_WIND,  linewidth=0.6, alpha=ALPHA_DAILY, zorder=2)
ax.plot(dates, ssdi_2018, color=C_SOLAR, linewidth=0.6, alpha=ALPHA_DAILY, zorder=2)
 
# ── 30-day moving averages (bold) ─────────────────────────────────────
ax.plot(dates, smdi_ma, color=C_TEAL,  linewidth=2.1, zorder=3)
ax.plot(dates, svdi_ma, color=C_CORAL, linewidth=2.1, zorder=3)
ax.plot(dates, swdi_ma, color=C_WIND,  linewidth=1.9, linestyle="--",        zorder=3)
ax.plot(dates, ssdi_ma, color=C_SOLAR, linewidth=1.9, linestyle=(0, (5, 2)), zorder=3)
 
# Directional fills under/over zero for moving averages
ax.fill_between(dates, smdi_ma, 0, where=(smdi_ma < 0), alpha=0.14, color=C_TEAL,  zorder=2)
ax.fill_between(dates, svdi_ma, 0, where=(svdi_ma > 0), alpha=0.10, color=C_CORAL, zorder=2)
 
# Compound fill: where BOTH wind and solar moving averages are negative
both_neg = (swdi_ma < 0) & (ssdi_ma < 0)
ax.fill_between(dates, np.fmin(swdi_ma, ssdi_ma), 0,
                where=both_neg, alpha=0.20, color=C_AMBER, zorder=2)
 
# ── Annotations ───────────────────────────────────────────────────────
ax.annotate(
    "Compound drought-heat\n(JJA — agricultural window)",
    xy=(pd.Timestamp("2018-07-20"), ax.get_ylim()[1] * 0.88),
    fontsize=7.5, color=C_TEAL, fontweight="bold",
    ha="center", va="top",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_TEAL, lw=0.7, alpha=0.93)
)
 
ax.annotate(
    "Dunkelflaute — wind AND\nsolar collapse together\n(30 Apr 2018)",
    xy=(pd.Timestamp("2018-04-30"), ax.get_ylim()[0] * 0.82),
    fontsize=7.2, color=C_AMBER, fontweight="bold",
    ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_AMBER, lw=0.7, alpha=0.93)
)
 
# ── ±1σ labels ────────────────────────────────────────────────────────
ax.text(dates[-1] + pd.Timedelta(days=3),  1.02, "±1σ",
        fontsize=6.5, color="#BBBBBB", va="bottom")
ax.text(dates[-1] + pd.Timedelta(days=3), -1.08, "±1σ",
        fontsize=6.5, color="#BBBBBB", va="top")
 
# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xlim(dates[0] - pd.Timedelta(days=4), dates[-1] + pd.Timedelta(days=14))
ax.set_ylabel("Normalised anomaly (σ)", fontsize=8.5, color=C_NAVY, labelpad=6)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.tick_params(axis="x", labelsize=8.5, colors=C_NAVY)
ax.tick_params(axis="y", labelsize=8,   colors=C_GRAY)
for spine in ax.spines.values():
    spine.set_color("#DDDDDD")
    spine.set_linewidth(0.6)
 
# ── Legend ────────────────────────────────────────────────────────────
agri_sm  = mpatches.Patch(color=C_TEAL,  label="Soil moisture anomaly (SMDI)")
agri_vpd = mpatches.Patch(color=C_CORAL, label="Temperature / VPD anomaly (SVDI)")
en_wind  = mlines.Line2D([], [], color=C_WIND,  linewidth=1.9,
                         linestyle="--",        label="Wind CF anomaly (SWDI)")
en_solar = mlines.Line2D([], [], color=C_SOLAR, linewidth=1.9,
                         linestyle=(0, (5, 2)), label="Solar CF anomaly (SSDI)")
sep      = mpatches.Patch(color="none", label=" ")
 
ax.legend(
    handles=[agri_sm, agri_vpd, sep, en_wind, en_solar],
    fontsize=7.5, loc="upper left", ncol=1,
    framealpha=0.93, edgecolor="#DDDDDD",
    handlelength=1.8, handletextpad=0.6, labelspacing=0.30,
    title="— bold: 30-day moving avg  · faint: daily",
    title_fontsize=6.5
)
 
# ── Source note ───────────────────────────────────────────────────────
ax.text(0.01, -0.13,
        "ERA5-derived daily anomalies, Netherlands 2018  "
        "· Bold lines = 30-day centred moving average  · Faint = daily values",
        transform=ax.transAxes, fontsize=5.8, color="#AAAAAA", style="italic")
 
plt.tight_layout(pad=0.4)
 
out_path = Path("./chart_2018_real.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Chart saved to {out_path}")
plt.show()
 
















# %%
