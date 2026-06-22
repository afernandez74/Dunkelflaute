"""
00_dwnld_arco_era5.py
=====================
Download ERA5 variables needed for the dunkelflaute analysis from the
ARCO-ERA5 Google Cloud Public Dataset and save as yearly NetCDF files.

Variables downloaded:
  * 10m u- and v-wind components  (u10, v10)  — near-surface wind for onshore
  * 100m u- and v-wind components (u100, v100) — hub-height wind for offshore/onshore
  * surface solar radiation downwards (ssrd)   — solar PV capacity factor
  * 2 m temperature (t2m)                      — PV temperature correction

Geographic domain:
  The onshore domain mirrors the 'ag' subproject [N55, W5, S41, E16] and is
  extended by a 200 km offshore buffer in all coastal directions:
      200 km ≈ 1.8° latitude
      200 km ≈ 2.9° longitude (at mid-domain latitude ~48°N)
  Resulting domain: N57, W8, S39, E19  — covers the Netherlands, Belgium,
  Germany, France, Denmark, and adjacent North Sea / Atlantic / Mediterranean
  offshore zones required for offshore wind resource assessment.

Dataset reference:
  https://cloud.google.com/storage/docs/public-datasets/era5
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

Usage
-----
  # Single year (used by SLURM array):
  python 00_dwnld_arco_era5.py --year 2005

  # All years sequentially (local / interactive):
  python 00_dwnld_arco_era5.py

Environment
-----------
  ERA5_dat   — base directory for output files (set in ~/.bashrc on HPC)
               Downloaded NetCDF files are written to $ERA5_dat/df_dat/
"""

import argparse
import logging
import os

import gcsfs
import xarray as xr
from tqdm import tqdm

import dask
dask.config.set({"array.slicing.split_large_chunks": True})

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Geographic bounding box: [North, West, South, East] (degrees)
# + 200 km offshore buffer (~1.8° lat, ~2.9° lon at 48°N mean latitude)
AREA = [62, -14, 42, 18]

# ARCO-ERA5 variable names (long names as stored in the Zarr store).
# These map to the short names: u10, v10, u100, v100, ssrd, t2m.
# NOTE: 10m wind components are included alongside 100m winds so that
#       the ETL/CF pipeline can optionally compute a hub-height wind speed
#       by log-law extrapolation if needed; remove them to reduce file size.
VARIABLES = [
    # "10m_u_component_of_wind",        # u10  — near-surface wind (onshore reference)
    # "10m_v_component_of_wind",        # v10
    "100m_u_component_of_wind",       # u100 — hub-height wind (primary for CF)
    "100m_v_component_of_wind",       # v100
    "surface_solar_radiation_downwards",  # ssrd — accumulated J/m² per hour
    "2m_temperature",                 # t2m  — PV temperature correction
    "geopotential"  # on pressure levels — subset to 500 hPa below (Z500)
]

# Temporal domain
YEAR_START = 1980
YEAR_END   = 2024          # inclusive; script caps at last available date


# Pressure level (hPa) for level-based variables 
PRESSURE_LEVEL = 500

# Z = geopotential / g.  True → store as height `z` in metres; False → raw geopotential.
CONVERT_TO_GEOPOTENTIAL_HEIGHT = True
G0 = 9.80 

# Output directory — NetCDF files written here, one per year
# Uses the same ERA5_dat env var as the 'ag' subproject, but a separate
# subdirectory (df_dat) to avoid clashing with ag downloads.
_ERA5_DAT = os.environ.get("ERA5_dat")
if _ERA5_DAT is None:
    raise EnvironmentError(
        "Environment variable 'ERA5_dat' is not set.\n"
        "Add it to ~/.bashrc, e.g.:\n"
        "  export ERA5_dat=/path/to/data/era5"
    )
OUTPUT_DIR = os.path.join(_ERA5_DAT, "df_dat")

# ARCO-ERA5 Zarr store on Google Cloud Storage (public, no auth needed)
ZARR_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# NetCDF encoding — compression settings applied to every variable
NC_ENCODING_DEFAULTS = {
    "zlib": True,
    "complevel": 4,
    "dtype": "float32",
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def open_arco_era5(zarr_store: str) -> xr.Dataset:
    """Open the ARCO-ERA5 Zarr store anonymously (no credentials required)."""
    log.info("Opening ARCO-ERA5 Zarr store …")
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(zarr_store, gcs=fs, check=False)
    ds    = xr.open_zarr(store, consolidated=True)
    log.info(
        "Store opened — available time range: %s → %s",
        str(ds.time.values[0])[:10],
        str(ds.time.values[-1])[:10],
    )
    return ds


def select_variables(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Keep only the requested variables; raise a clear error if any are missing."""
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        available = sorted(ds.data_vars)
        raise ValueError(
            f"Variable(s) not found in ARCO-ERA5 dataset: {missing}\n"
            f"Available variables:\n  " + "\n  ".join(available)
        )
    return ds[variables]

def select_pressure_level(ds: xr.Dataset, level_hpa: int) -> xr.Dataset:
    """
    Subset level-based variables (e.g. geopotential) to a single pressure level.
    Surface variables (no 'level' dim) pass through unchanged; the scalar `level`
    coordinate is retained for provenance.
    """
    if "level" not in ds.dims and "level" not in ds.coords:
        return ds
    ds = ds.sel(level=level_hpa)
    log.info("Selected pressure level: %d hPa", level_hpa)
    return ds


def select_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    north, west, south, east = area
    lat_slice = slice(north, south)  

    # Convert to 0–360
    west_360 = west % 360
    east_360 = east % 360

    if west_360 <= east_360:
        # Domain does not cross Greenwich
        ds = ds.sel(
            latitude=lat_slice,
            longitude=slice(west_360, east_360)
        )
    else:
        # Domain crosses Greenwich — concatenate west and east chunks,
        # then reassign longitudes to a continuous -180→180 range
        ds_e = ds.sel(latitude=lat_slice, longitude=slice(0, east_360))
        ds_w = ds.sel(latitude=lat_slice, longitude=slice(west_360, 360))
        # Shift west chunk longitudes to negative so concat is monotonic
        ds_w = ds_w.assign_coords(longitude=ds_w.longitude.values - 360)
        ds = xr.concat([ds_w, ds_e], dim="longitude")

    log.info(
        "Spatial subset: lat [%.2f → %.2f], lon [%.2f → %.2f]  (%d × %d grid cells)",
        float(ds.latitude.max()),
        float(ds.latitude.min()),
        float(ds.longitude.min()),
        float(ds.longitude.max()),
        ds.sizes["latitude"],
        ds.sizes["longitude"],
    )
    return ds



def to_geopotential_height(ds: xr.Dataset) -> xr.Dataset:
    """Convert `geopotential` (m² s⁻²) → geopotential height `z` (m)."""
    if "geopotential" not in ds.data_vars:
        return ds
    ds = ds.rename({"geopotential": "z"})
    ds["z"] = ds["z"] / G0
    ds["z"].attrs.update(
        units="m",
        long_name="Geopotential height",
        standard_name="geopotential_height",
    )
    return ds

def build_encoding(ds: xr.Dataset) -> dict:
    """Build per-variable NetCDF encoding dictionary."""
    return {var: NC_ENCODING_DEFAULTS.copy() for var in ds.data_vars}


def download_year(ds: xr.Dataset, year: int, output_dir: str) -> None:
    """
    Select one calendar year from the remote Zarr store, stream it to disk
    as a compressed NetCDF file.  Skips years already downloaded.
    """
    out_path = os.path.join(output_dir, f"era5_{year}.nc")

    if os.path.exists(out_path):
        log.info("Year %d — file already exists, skipping: %s", year, out_path)
        return

    t_min = str(ds.time.values[0])[:10]
    t_max = str(ds.time.values[-1])[:10]
    year_start = f"{year}-01-01"
    year_end   = f"{year}-12-31"

    if year_start > t_max or year_end < t_min:
        log.warning("Year %d — no data available in store, skipping.", year)
        return

    t0 = max(year_start, t_min)
    t1 = min(year_end,   t_max)

    log.info("Year %d — selecting %s → %s …", year, t0, t1)
    ds_year = ds.sel(time=slice(t0, t1))

    log.info(
        "Year %d — writing %d timesteps to disk (streaming) …",
        year, ds_year.sizes["time"],
    )

    ds_year.to_netcdf(out_path, encoding=build_encoding(ds_year), compute=True)
    ds_year.close()
    log.info("Year %d — done ✓  →  %s", year, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ARCO-ERA5 data for the dunkelflaute subproject.\n"
            "Saves one compressed NetCDF file per year to $ERA5_dat/df_dat/."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Calendar year to download (e.g. 2005).  "
            "If omitted, all years from YEAR_START to YEAR_END are downloaded "
            "sequentially.  When submitted as a SLURM array job, pass "
            "$SLURM_ARRAY_TASK_ID here."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Output directory: %s", os.path.abspath(OUTPUT_DIR))
    log.info("Variables: %s", VARIABLES)
    log.info("Domain (N/W/S/E): %s", AREA)

    # ── Open remote dataset ──────────────────────────────────────────────────
    ds_full = open_arco_era5(ZARR_STORE)

    # ── Variable selection ───────────────────────────────────────────────────
    ds = select_variables(ds_full, VARIABLES)

    # pressure-level subset (geopotential height single level z500)
    ds = select_pressure_level(ds, PRESSURE_LEVEL)
    if CONVERT_TO_GEOPOTENTIAL_HEIGHT:
        ds = to_geopotential_height(ds)
    
    # ── Spatial subsetting ───────────────────────────────────────────────────
    ds = select_area(ds, AREA)

    log.info(
        "Subset size: %d lat × %d lon × %d time (full store)",
        ds.sizes["latitude"],
        ds.sizes["longitude"],
        ds.sizes["time"],
    )

    # Chunk along time for efficient streaming reads from GCS
    ds = ds.chunk({"time": 24})

    # ── Download ─────────────────────────────────────────────────────────────
    if args.year is not None:
        # Single-year mode — used by SLURM array tasks
        log.info("Single-year mode: year = %d", args.year)
        download_year(ds, args.year, OUTPUT_DIR)
    else:
        # Sequential mode — local / interactive use
        last_available_year = int(str(ds.time.values[-1])[:4])
        year_end_eff = min(YEAR_END, last_available_year)
        years = list(range(YEAR_START, year_end_eff + 1))
        log.info(
            "Sequential mode: downloading %d years (%d → %d) …",
            len(years), years[0], years[-1],
        )
        for year in tqdm(years, desc="Years", unit="yr"):
            try:
                download_year(ds, year, OUTPUT_DIR)
            except Exception as exc:
                log.error("Year %d — FAILED: %s", year, exc)

    log.info("Finished.  Output directory: %s", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
