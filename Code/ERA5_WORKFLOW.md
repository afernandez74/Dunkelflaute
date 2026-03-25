# ERA5 Data Download and Preprocessing Workflow

## Overview

This document describes the refactored ERA5 download and preprocessing workflow. The code is organized into reusable library modules (`src/`) and executable scripts (`scripts/`).

## File Structure

```
Code/
├── src/
│   ├── era5_download.py          # Functions for downloading ERA5 data
│   └── era5_preprocessing.py     # Functions for merging/preprocessing ERA5 files
│
└── scripts/
    ├── download_era5.py           # Script to download ERA5 data
    └── merge_era5.py              # Script to merge downloaded files
```

## Workflow

### Step 1: Download ERA5 Data

Download data from the Copernicus Climate Data Store (CDS):

```bash
python Code/scripts/download_era5.py \
    --variables 2m_temperature 100m_u_component_of_wind 100m_v_component_of_wind surface_solar_radiation_downwards \
    --start-year 2020 \
    --end-year 2023 \
    --download-dir Data/ERA5/downloads/
```

**Options:**
- `--variables`: List of variables to download (required)
- `--start-year`: First year (required)
- `--end-year`: Last year, inclusive (required)
- `--download-dir`: Where to save files (required)
- `--area`: Geographic area [N, W, S, E] (optional, defaults to Netherlands region)
- `--dataset`: CDS dataset name (optional, defaults to "reanalysis-era5-single-levels")

**Note:** You need CDS API credentials set up. See: https://cds.climate.copernicus.eu/api-how-to

### Step 2: Merge Downloaded Files (Optional)

After downloading, you can merge all files into a single NetCDF:

```bash
python Code/scripts/merge_era5.py \
    --download-dir Data/ERA5/downloads/ \
    --output-dir Data/ERA5/
```

**Options:**
- `--download-dir`: Directory with downloaded files (required)
- `--output-dir`: Where to save merged file (required)
- `--pattern`: Optional filename pattern (e.g., "ERA5_*_2020_*.nc")
- `--output-filename`: Custom output filename (optional, auto-generated if not provided)
- `--lazy`: Use lazy loading (faster but may not write complete file immediately)
- `--dry-run`: Preview what would be merged without actually merging

## Recommendations: To Merge or Not to Merge?

### Option A: Keep Individual Files (Recommended for Most Cases)

**Pros:**
- ✅ Easier to update incrementally (add new years without re-merging)
- ✅ More flexible (can merge subsets on-the-fly)
- ✅ Less risk (if one file corrupts, others are safe)
- ✅ Can use xarray's lazy loading: `xr.open_mfdataset("Data/ERA5/downloads/*.nc")`

**Cons:**
- ❌ Slower to open many files (but xarray handles this well with lazy loading)

**When to use:** Default approach. Use xarray's `open_mfdataset()` for analysis.

### Option B: Merge into Single File

**Pros:**
- ✅ Single file is easier to manage
- ✅ Faster to open (no merging overhead)
- ✅ Good for sharing complete datasets

**Cons:**
- ❌ Very large files (10s-100s of GB)
- ❌ Hard to update (must re-merge to add data)
- ❌ If corrupted, lose everything
- ❌ Slow initial write

**When to use:** 
- When you have a complete, final dataset
- When sharing with collaborators
- When you need maximum read performance

### Option C: Merge by Variable Groups

**Pros:**
- ✅ Balance between flexibility and performance
- ✅ Can update individual variable groups
- ✅ Smaller files than full merge

**Example:**
```bash
# Merge all wind variables
python Code/scripts/merge_era5.py --download-dir Data/ERA5/downloads/ --output-dir Data/ERA5/ --pattern "*u100*v100*" --output-filename ERA5_wind_merged.nc

# Merge all solar/temperature variables
python Code/scripts/merge_era5.py --download-dir Data/ERA5/downloads/ --output-dir Data/ERA5/ --pattern "*t2m*ssrd*" --output-filename ERA5_solar_temp_merged.nc
```

## Example: Complete Workflow

```bash
# 1. Download ERA5 data for 2020-2023
python Code/scripts/download_era5.py \
    --variables 2m_temperature 100m_u_component_of_wind 100m_v_component_of_wind surface_solar_radiation_downwards \
    --start-year 2020 \
    --end-year 2023 \
    --download-dir Data/ERA5/downloads/

# 2. Preview what would be merged (optional)
python Code/scripts/merge_era5.py \
    --download-dir Data/ERA5/downloads/ \
    --output-dir Data/ERA5/ \
    --dry-run

# 3. Merge files (optional - only if you want a single file)
python Code/scripts/merge_era5.py \
    --download-dir Data/ERA5/downloads/ \
    --output-dir Data/ERA5/

# 4. Use merged file (or individual files) for analysis
python Code/scripts/compute_capacity_factors.py \
    --era5-data Data/ERA5/ERA5_merged_*.nc \
    --output-dir Results/ \
    --save
```

## Using Individual Files Without Merging

You can use xarray's `open_mfdataset()` to work with multiple files without merging:

```python
import xarray as xr

# Lazy loading - doesn't load all data into memory
ds = xr.open_mfdataset("Data/ERA5/downloads/*.nc", combine="by_coords")

# Now use ds as if it were a single file
cf_solar = compute_solar_cf(ds.ssrd, ds.t2m)
```

This is often the best approach - you get the convenience of a single dataset without the overhead of merging!
