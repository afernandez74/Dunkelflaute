#%%
# 06_z500_composite.py — shared blocking composite for CDHW (JJA) and Dunkelflaute (NDJFM)
# Composites z500 anomaly over detected event days to demonstrate the shared synoptic driver.

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

G0 = 9.80665                                   # gravity, for geopotential -> gpm

# Euro-Atlantic sector for the composite maps (wide enough to see the ridge)

LON_W, LON_E = -14.0, 18.0
LAT_S, LAT_N =  42.0, 62.0                     

# event pooling — spatially-extensive event days in both cases
DF_MIN_UNITS  = 3        # DF: >= this many of 7 countries in RL event that day
CDHW_MIN_FRAC = 0.20     # CDHW: >= this fraction of domain cells compound that day


# consistency stippling
STIP_THRESH = 0.80        # stipple where >= 80% of events share the anomaly sign
STIP_STEP   = 3           # subsample stipple points (every Nth gridcell) to avoid clutter
SHARED_SCALE = False      # True -> one symmetric colour scale across both panels

# impact regions (for outlines + area-mean metrics)
CDHW_REGION = ['France', 'Belgium', 'Netherlands', 'Germany']
DF_REGION   = ['France', 'Belgium', 'Netherlands', 'Germany',
               'Denmark', 'United Kingdom', 'Ireland']

# I/O — local pre-processed ERA5 
ERA5_DAT    = Path("/Volumes/AlejoED/WELCOME/")
Z_STORE     = ERA5_DAT / "df_dat/processed/df_dat_cleaned.zarr"

RL_DIR      = Path('./../Results/DF_RL')
CDHW_DIR    = Path('./../Results/CDHW')                 
DF_EVENTS   = RL_DIR / 'rl_events_fully_built.parquet'  
COUNTRIES_SHP = Path('~/CDHW_ag/Data/countries/ne_10m_admin_0_countries.shp').expanduser()
OUT_DIR     = Path('./../Results/Composites')
Z500_CACHE  = OUT_DIR / 'z500_anom_daily.zarr'


# CDHW catalogue lives in a self-documenting param folder (idx branch only)
CDHW_PARAM  = 'sm20_vpd90_sev10_95_gap3_dur3_off0'
CDHW_EVENTS = Path('~/CE/ag/Results/CDHW_results') / CDHW_PARAM / 'events_idx.parquet'

OUT_DIR.mkdir(parents=True, exist_ok=True)
#%%
# Daily z500 anomaly from local pre-processed store (cached)

if Z500_CACHE.exists():
    z_anom = xr.open_zarr(Z500_CACHE, consolidated=True)['z_anom']
    print(f"Loaded cached z500 anomaly: {z_anom.time.size} days.")
else:
    print("Building daily z500 anomaly from local store (one-time)...")
    ds = xr.open_zarr(Z_STORE, consolidated=True)
    z_raw = ds['z']                                     # (time, lat, lon), hourly

    # --- units check: geopotential (m2/s2, ~53000) vs geopotential metres (~5500) ---
    z_probe = float(z_raw.isel(time=0).mean().compute())
    G_DIV = G0 if z_probe > 20000 else 1.0
    print(f"  z sample mean = {z_probe:.0f}  ->  dividing by {G_DIV:.5f} "
          f"({'m2/s2 -> gpm' if G_DIV != 1.0 else 'already gpm'})")
    # After first run, HARD-SET this to avoid re-inferring:  G_DIV = 9.80665   (or 1.0)

    z = (z_raw / G_DIV).rename('z')
    z_daily = z.coarsen(time=24, boundary='trim', coord_func='min').mean()

    # smoothed day-of-year climatology (wrap year so Dec/Jan smoothing is continuous)
    clim = z_daily.groupby('time.dayofyear').mean()
    clim = (clim.pad(dayofyear=15, mode='wrap')
                .rolling(dayofyear=31, center=True, min_periods=1).mean()
                .isel(dayofyear=slice(15, -15)))

    z_anom = (z_daily.groupby('time.dayofyear') - clim).rename('z_anom')
    z_anom = z_anom.chunk({'time': -1, 'latitude': -1, 'longitude': -1})
    z_anom.to_dataset().to_zarr(Z500_CACHE, mode='w', consolidated=True)
    z_anom = xr.open_zarr(Z500_CACHE, consolidated=True)['z_anom']
    print(f"Cached z500 anomaly: {z_anom.time.size} days "
          f"({str(z_anom.time.values[0])[:10]} -> {str(z_anom.time.values[-1])[:10]}).")

z_times = pd.DatetimeIndex(z_anom.time.values)

#%%
# Event-date sets: spatially-extensive event days from each catalogue

def active_day_counts(ev_df, months):
    """Daily count of simultaneously-active event units (interval difference-array)."""
    starts = pd.to_datetime(ev_df['start_date']).dt.normalize()
    ends   = pd.to_datetime(ev_df['end_date']).dt.normalize()
    idx    = pd.date_range(starts.min(), ends.max(), freq='D')
    delta  = (starts.value_counts().reindex(idx, fill_value=0)
              - (ends + pd.Timedelta(days=1)).value_counts().reindex(idx, fill_value=0))
    counts = delta.cumsum().astype(int)
    return counts[counts.index.month.isin(months)]

def threshold_days(counts, min_units):
    dates = counts.index[counts >= min_units]
    return dates[dates.isin(z_times)]

# --- Dunkelflaute: >= DF_MIN_UNITS of 7 countries ---
df_ev   = pd.read_parquet(DF_EVENTS)
df_ev   = df_ev[df_ev['track'] == 'RL_Relative_90pct']
df_cnt  = active_day_counts(df_ev, months=[11, 12, 1, 2, 3])
dates_df = threshold_days(df_cnt, DF_MIN_UNITS)

# --- CDHW: >= CDHW_MIN_FRAC of domain cells (per-cell catalogue) ---
cdhw_ev = pd.read_parquet(CDHW_EVENTS)                    # per (cell x event), has lat/lon
n_cells = cdhw_ev[['latitude', 'longitude']].drop_duplicates().shape[0]  # domain proxy
cdhw_min_cells = int(np.ceil(CDHW_MIN_FRAC * n_cells))
cdhw_cnt  = active_day_counts(cdhw_ev, months=[6, 7, 8])
dates_cdhw = threshold_days(cdhw_cnt, cdhw_min_cells)

print(f"CDHW domain cells (unique in catalogue): {n_cells}")
print(f"CDHW daily active-cell count — median {cdhw_cnt.median():.0f}, "
      f"p90 {cdhw_cnt.quantile(0.9):.0f}, max {cdhw_cnt.max():.0f}")
print(f"CDHW threshold: >= {cdhw_min_cells} cells ({CDHW_MIN_FRAC:.0%} of domain)")
print(f"\nDunkelflaute (NDJFM, >={DF_MIN_UNITS} countries): {len(dates_df)} event-days")
print(f"CDHW        (JJA,   >={cdhw_min_cells} cells):     {len(dates_cdhw)} event-days")
#%%
# Composite anomaly and event-wise sign consistency

def composite(dates):
    sub  = z_anom.sel(time=dates).load()
    comp = sub.mean('time')
    frac_pos = (sub > 0).mean('time')          # fraction of events with positive anomaly
    return comp, frac_pos, len(dates)

comp_cdhw, sign_cdhw, n_cdhw = composite(dates_cdhw)
comp_df,   sign_df,   n_df   = composite(dates_df)
print("Composites built.")
#%%
# Region masks + summary metrics for the text

def region_mask(template, names):
    land = gpd.read_file(str(COUNTRIES_SHP))
    lo, la = np.meshgrid(template.longitude.values, template.latitude.values)
    pts = shapely.points(lo.ravel(), la.ravel())
    geom = land.loc[land['ADMIN'].isin(names)].union_all()
    m = shapely.within(pts, geom).reshape(la.shape)
    return xr.DataArray(m, coords={'latitude': template.latitude,
                                   'longitude': template.longitude},
                        dims=['latitude', 'longitude'])

mask_cdhw = region_mask(comp_cdhw, CDHW_REGION)
mask_df   = region_mask(comp_df,   DF_REGION)

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlam/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def summarize(comp, sign, mask, n, label):
    w = np.cos(np.radians(comp.latitude))
    area_mean = float(comp.where(mask).weighted(w).mean(('latitude', 'longitude')))
    sign_mean = float(sign.where(mask).weighted(w).mean(('latitude', 'longitude')))
    pk = comp.where((comp.latitude >= 35) & (comp.latitude <= 70))       # peak within core sector
    j = pk.argmax(dim=['latitude', 'longitude'])
    plat = float(pk.latitude[j['latitude']]); plon = float(pk.longitude[j['longitude']])
    print(f"{label}: N={n} | region mean anomaly {area_mean:+.0f} gpm | "
          f"region sign-consistency {sign_mean:.0%} | ridge peak {plat:.0f}N {plon:.0f}E ({float(pk.max()):+.0f} gpm)")
    return plat, plon

p_cdhw = summarize(comp_cdhw, sign_cdhw, mask_cdhw, n_cdhw, "CDHW  (JJA)")
p_df   = summarize(comp_df,   sign_df,   mask_df,   n_df,   "Dunkelflaute (NDJFM)")
print(f"\nSeasonal ridge-centre separation: {haversine(*p_cdhw, *p_df):.0f} km")
#%%
# Two-panel composite figure

proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52)
pc   = ccrs.PlateCarree()

if SHARED_SCALE:
    vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [comp_cdhw.values.ravel(), comp_df.values.ravel()])), 99))
    lims = [vmax, vmax]
else:
    lims = [float(np.nanpercentile(np.abs(c.values), 99)) for c in (comp_cdhw, comp_df)]

def draw(ax, comp, sign, mask, names, vmax, title, n):
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=pc)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor='0.3')
    ax.add_feature(cfeature.BORDERS,   lw=0.3, edgecolor='0.6')

    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(comp.longitude, comp.latitude, comp, levels=levels,
                     cmap='RdBu_r', extend='both', transform=pc)

    # stipple robust-sign gridcells
    rob = ((sign >= STIP_THRESH) | (sign <= 1 - STIP_THRESH))
    lo, la = np.meshgrid(comp.longitude.values, comp.latitude.values)
    sel = rob.values & np.isfinite(comp.values)
    ax.scatter(lo[sel][::STIP_STEP], la[sel][::STIP_STEP], s=0.6,
               color='k', alpha=0.5, transform=pc)

    # outline the impact region
    land = gpd.read_file(str(COUNTRIES_SHP))
    geom = land.loc[land['ADMIN'].isin(names)].union_all()
    ax.add_geometries([geom], crs=pc, facecolor='none', edgecolor='#111', lw=1.4)

    ax.set_title(f"{title}\n(n = {n} event-days)", fontweight='bold', fontsize=12)
    return cf

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), subplot_kw={'projection': proj})
cf0 = draw(axes[0], comp_cdhw, sign_cdhw, mask_cdhw, CDHW_REGION, lims[0],
           "Summer compound drought–heat", n_cdhw)
cf1 = draw(axes[1], comp_df,   sign_df,   mask_df,   DF_REGION,   lims[1],
           "Winter Dunkelflaute", n_df)

for ax, cf in zip(axes, (cf0, cf1)):
    cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85)
    cb.set_label('z500 anomaly (gpm)')

fig.suptitle("One driver, two seasons, two receptors: anticyclonic blocking over detected events",
             fontsize=15, y=1.02)
plt.tight_layout()
# plt.savefig(OUT_DIR / 'z500_composite_cdhw_vs_df.png', dpi=200, bbox_inches='tight')
plt.show()
#%%
# OPTIONAL: Monte-Carlo field significance (set RUN_MC = True to use)

RUN_MC = False
if RUN_MC:
    def mc_pvalue(dates, months, n_iter=500, seed=0):
        rng = np.random.default_rng(seed)
        pool = z_times[np.isin(z_times.month, months)]
        obs  = z_anom.sel(time=dates).mean('time').load()
        exceed = xr.zeros_like(obs)
        for _ in range(n_iter):
            draw = pd.DatetimeIndex(rng.choice(pool, size=len(dates), replace=False))
            null = z_anom.sel(time=draw).mean('time').load()
            exceed = exceed + (np.abs(null) >= np.abs(obs))
        return (exceed / n_iter)                    # two-sided p per gridcell

    p_cdhw_mc = mc_pvalue(dates_cdhw, [6, 7, 8])
    p_df_mc   = mc_pvalue(dates_df,   [11, 12, 1, 2, 3])
    print("MC significance computed; stipple where p < 0.05 instead of sign-consistency.")
# %%
