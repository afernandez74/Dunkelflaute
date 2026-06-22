import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401 — registers .rio accessor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates

# %%
# ── Paths ─────────────────────────────────────────────────────────────
ERA5_path = Path(os.environ["ERA5_dat"])
DF_path   = Path("./../Results/")
CDHW_path = Path("~/CDHW_ag/Results")


ssdi_path = DF_path  / "SSDI" / "ssdi.zarr"
swdi_path = DF_path  / "SWDI" / "swdi.zarr"
smdi_path = CDHW_path / "SMDI" / "smdi.zarr"
svdi_path = CDHW_path / "SVDI" / "svdi.zarr"

cf_solar_path = DF_path / "CF_daily" / "CF_solar_24h_daily.zarr"
cf_wind_path = DF_path / "CF_daily" / "CF_wind_daily.zarr"
countries_path = Path("../../../CDHW_ag/Data/countries_shp/ne_110m_admin_0_countries.shp")

YEAR = "2018"

# Capacity-mix weights for the combined energy index (NL ~2018 fleet)
# Approximate split: wind ~60%, solar ~40% of combined installed capacity.
W_WIND  = 0.60
W_SOLAR = 0.40

# %%
# ── Load zarr stores ──────────────────────────────────────────────────
print("Opening Zarr stores...")
ssdi_ds = xr.open_zarr(ssdi_path, consolidated=True, chunks={})
swdi_ds = xr.open_zarr(swdi_path, consolidated=True, chunks={})
smdi_ds = xr.open_zarr(smdi_path, consolidated=True, chunks={})
svdi_ds = xr.open_zarr(svdi_path, consolidated=True, chunks={})
cf_solar_ds = xr.open_zarr(cf_solar_path, consolidated=True,chunks={})
cf_wind_ds = xr.open_zarr(cf_wind_path, consolidated=True,chunks={})
countries_gdf = gpd.read_file(countries_path)

# Extract primary DataArrays
ssdi_da = ssdi_ds[list(ssdi_ds.data_vars)[0]]
swdi_da = swdi_ds[list(swdi_ds.data_vars)[0]]
smdi_da = smdi_ds[list(smdi_ds.data_vars)[0]]
svdi_da = svdi_ds[list(svdi_ds.data_vars)[0]]
cf_solar_da = cf_solar_ds[list(cf_solar_ds.data_vars)[0]]
cf_wind_da = cf_wind_ds[list(cf_wind_ds.data_vars)[0]]

# %%
# clip to NL
# ── Clip to NL and spatially average ─────────────────────────────────
nl_geom = countries_gdf[countries_gdf["SOVEREIGNT"] == "Netherlands"].dissolve()

def nl_mean(da: xr.DataArray) -> xr.DataArray:
    """Clip to NL interior cells (all_touched=False) and return spatial mean."""
    clipped = (
        da
        .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        .rio.write_crs("EPSG:4326")
        .rio.clip(nl_geom.geometry, nl_geom.crs, all_touched=False)
    )
    return clipped.mean(dim=["latitude", "longitude"])

print("Clipping and averaging over the Netherlands...")
ssdi_nl = nl_mean(ssdi_da)
swdi_nl = nl_mean(swdi_da)
smdi_nl = nl_mean(smdi_da)
svdi_nl = nl_mean(svdi_da)
cf_solar_nl = nl_mean(cf_solar_da)
cf_wind_nl = nl_mean(cf_wind_da)
# %%
