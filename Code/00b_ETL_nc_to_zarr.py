"""
00b_ETL_nc_to_zarr.py
=====================
ERA5 ETL Pipeline: ARCO-ERA5 NetCDF → Zarr for the dunkelflaute subproject.

Purpose
-------
Converts the raw annual ERA5 NetCDF files produced by 00_dwnld_arco_era5.py
into a single, consolidated Zarr store optimised for time-series analysis of
wind and solar capacity factors.

Transformations applied
-----------------------
1. Longitude wrap     : [0, 360]  →  [−180, 180]
2. Coordinate sort    : time, latitude, longitude for contiguous disk access
3. Variable rename    : long ARCO-ERA5 names → CF-compliant short names
                        (uses the 'short_name' attribute stored in the file;
                         falls back to the lookup table below if absent)
4. Re-chunk           : full time axis (−1) × LAT_CHUNK × LON_CHUNK tiles
                        → optimised for reading all hours at a fixed grid cell,
                          which is the dominant access pattern in CF/event code.
5. Consolidated Zarr  : single metadata file for near-instant open()

Short-name lookup (fallback if 'short_name' attribute is missing)
-----------------------------------------------------------------
  10m_u_component_of_wind          → u10
  10m_v_component_of_wind          → v10
  100m_u_component_of_wind         → u100
  100m_v_component_of_wind         → v100
  surface_solar_radiation_downwards → ssrd
  2m_temperature                    → t2m

Environment
-----------
  ERA5_dat   — base directory; script reads from $ERA5_dat/df_dat/
               and writes to $ERA5_dat/df_dat/processed/

"""
#%%
import os
import logging
from pathlib import Path

import xarray as xr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Spatial tile size for time-series-focused chunking.
# Full time axis (−1) means one contiguous chunk per lat/lon tile,
# which avoids repeated index lookups when reading long time series.
LAT_CHUNK = 10
LON_CHUNK = 10

# Subfolder under ERA5_dat that holds the raw annual NetCDF files
# OUTPUT_DIR in 00_dwnld_arco_era5.py
INPUT_FOLDER = "df_dat"

# Fallback short-name map used when a variable's 'short_name' attribute is
# absent.  Keys are ARCO-ERA5 long names; values are CF short names.
SHORT_NAME_MAP: dict[str, str] = {
    "10m_u_component_of_wind":            "u10",
    "10m_v_component_of_wind":            "v10",
    "100m_u_component_of_wind":           "u100",
    "100m_v_component_of_wind":           "v100",
    "surface_solar_radiation_downwards":  "ssrd",
    "2m_temperature":                     "t2m",
    # Additional variables kept for completeness (not downloaded by default)
    "2m_dewpoint_temperature":            "d2m",
    "10m_wind_speed":                     "si10",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

#%%
# ─────────────────────────────────────────────────────────────────────────────
# ETL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def convert_nc_to_zarr(folder_name: str, base_dir: Path) -> None:
    """
    Read all annual NetCDF files apply coordinate standardisation 
    and variable renaming, re-chunk 
    and write a Zarr store

    Parameters
    ----------
    folder_name : str
        Subdirectory with the raw *.nc files.
    base_dir : Path
        Root data directory
    """
    input_path  = Path(base_dir) / folder_name
    output_dir  = input_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path   = output_dir / f"{folder_name}_cleaned.zarr"

    # read files
    files = sorted(input_path.glob("*.nc"))
    if not files:
        log.error("No .nc files found in %s", input_path)
        return

    log.info("Found %d NetCDF file(s) in %s", len(files), input_path)
    log.info("Output Zarr store: %s", zarr_path)

    # load lazily
    log.info("Opening files with xr.open_mfdataset (parallel=True, lazy) …")
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=False,
        chunks={},
    )
    log.info(
        "Dataset loaded — variables: %s | time: %d steps | lat: %d | lon: %d",
        list(ds.data_vars),
        ds.sizes.get("time", 0),
        ds.sizes.get("latitude", 0),
        ds.sizes.get("longitude", 0),
    )

    # Longitude wrap: [0, 360] → [−180, 180] 
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    ds = ds.sortby(["time", "latitude", "longitude"])
    log.info(
        "Longitude wrapped and sorted — lon range: [%.2f, %.2f]",
        float(ds.longitude.min()),
        float(ds.longitude.max()),
    )

    # Variable renaming
    rename_dict: dict[str, str] = {}
    for var in ds.data_vars:
        # Prefer the 'short_name' attribute embedded in the file metadata
        short = ds[var].attrs.get("short_name", None)
        if short is None:
            # Fall back to the static lookup table
            short = SHORT_NAME_MAP.get(var, var)
            if short == var:
                log.warning(
                    "No short name found for '%s' — keeping original name.", var
                )
            else:
                log.info("Renamed '%s' → '%s'  (lookup table)", var, short)
        else:
            log.info("Renamed '%s' → '%s'  (short_name attribute)", var, short)
        rename_dict[var] = short

    ds = ds.rename(rename_dict)

    # Attach provenance attributes
    for var in ds.data_vars:
        ds[var].attrs.update({
            "source":    "ARCO-ERA5 (gs://gcp-public-data-arco-era5)",
            "processed": "00b_ETL_nc_to_zarr.py — dunkelflaute subproject",
        })

    # Re-chunk for time-series access 
    # Full time axis per spatial tile: optimal for reading all hours at a
    # given grid cell (capacity-factor and event-detection workflows).
    ds = ds.chunk({"time": -1, "latitude": LAT_CHUNK, "longitude": LON_CHUNK})
    log.info(
        "Re-chunked: time=full, lat=%d, lon=%d",
        LAT_CHUNK, LON_CHUNK,
    )

    # Write Zarr
    log.info("Writing Zarr store (this may take several minutes) …")
    ds.to_zarr(zarr_path, mode="w", consolidated=True)
    log.info("Success ✓  Zarr store written to: %s", zarr_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _era5_dat = os.environ.get("ERA5_dat")
    if _era5_dat is None:
        raise EnvironmentError(
            "Environment variable 'ERA5_dat' is not set.\n"
            "Add it to ~/.bashrc, e.g.:\n"
            "  export ERA5_dat=/path/to/data/era5"
        )
    base_dir = Path(_era5_dat)
    convert_nc_to_zarr(INPUT_FOLDER, base_dir)


if __name__ == "__main__":
    main()
