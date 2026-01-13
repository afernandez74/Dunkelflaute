# Climate-Driven Risk of Dunkelflaute Events in the Netherlands

This repository analyzes the joint scarcity of wind and solar generation (“dunkelflaute”) over the Netherlands using ERA5 and related datasets. It converts meteorological variables into capacity factors, combines them into system-level metrics, and assesses the frequency, duration, and return intervals of low-generation events. The framework is designed to extend to future climate projections and different wind/solar mixes.

## Objectives
- Quantify historical dunkelflaute risk: frequency, duration, return intervals.
- Link climate variability to energy impacts via capacity factors and system mixes.
- Assess spatial heterogeneity of risk across the Netherlands and nearby regions.
- Provide a baseline that can extend to future climate scenarios and alternative wind/solar deployments.

## Data sources
- ERA5 hourly reanalysis (wind components at 100 m, surface solar radiation, 2 m temperature).
- Optional: CMIP6/ERA5-consistent projections for future scenarios.
- Optional: Renewable Ninja national wind/solar capacity factors for comparison.

## Recommended project layout

```
Code/                     # repository root (this folder)
├─ README.md              # this file
├─ requirements.txt       # Python dependencies
├─ src/dunkelflaute/      # reusable library code (to create)
│  ├─ __init__.py
│  ├─ capacity_factors.py     # wind/solar CF calculations
│  ├─ events.py               # declustering, return intervals
│  ├─ dependence.py           # tail dependence metrics
│  ├─ era5_download.py        # CDS download helpers
│  └─ era5_preprocessing.py   # merge downloads into unified datasets
├─ scripts/               # thin command-line entrypoints (to create)
│  ├─ compute_capacity_factors.py
│  ├─ analyze_dunkelflaute.py
│  └─ tail_dependence.py
├─ Data/                  # input data (keep out of git if large)
│  ├─ ERA5/
│  └─ ninja_NL/
├─ Results/               # intermediate outputs (e.g., CF NetCDFs)
├─ Figures/               # plots/maps
└─ notebooks/             # optional exploratory notebooks
```

Your current scripts can be migrated gradually:
- Move reusable logic into `src/dunkelflaute/` functions.
- Keep the existing scripts as thin wrappers in `scripts/` that call those functions.
- Avoid interactive `input()` prompts; prefer command-line flags or config variables.

## Setup
1) Create a virtual environment (example with `venv`):
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
If you prefer Conda, create an `environment.yml` later and use `conda env create -f environment.yml`.

2) (Optional) Install development extras for testing/formatting:
```
pip install pytest black isort
```

## Quickstart workflows

### Merge ERA5 monthly downloads into a single dataset
Adapt from your existing `read_downloads_save_dataset.py`:
```
python scripts/merge_era5_downloads.py \
  --input-dir Data/ERA5/downloads \
  --output Data/ERA5/ERA5_hrly_vars.nc
```
*(We will create this script after refactoring.)*

### Compute capacity factors
Refactor `calc_CF_wind.py` and `calc_CF_solar.py` into `src/dunkelflaute/capacity_factors.py`, then run a thin wrapper:
```
python scripts/compute_capacity_factors.py \
  --era5 Data/ERA5/ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8 \
  --out Results/CF_wind/CF_wind.nc \
  --out-solar Results/CF_solar/CF_solar.nc
```

### Analyze dunkelflaute events
Refactor your declustering logic (`CF_analysis.py` / `DF_eventCount_ReturnInterval.py`) into `src/dunkelflaute/events.py`:
```
python scripts/analyze_dunkelflaute.py \
  --cf-wind Results/CF_wind/CF_wind.nc \
  --cf-solar Results/CF_solar/CF_solar.nc \
  --delta-solar 0.25 \
  --threshold 0.05 \
  --min-duration 3 \
  --min-gap 1 \
  --out Results/DF/df_events.nc
```

### Tail dependence / joint behavior
Move the lower-tail dependence code (`DF_delta_tail_dependence.py`) into `src/dunkelflaute/dependence.py` and call via `scripts/tail_dependence.py`.

## Coding standards to adopt
- Prefer functions and modules over monolithic scripts.
- Add docstrings with inputs/outputs/units and brief references (e.g., Brown et al. 2021 for power curve, Zscheischler et al. 2020 for compound events).
- Replace `input()` prompts with command-line flags (`argparse`) or a small config file.
- Keep paths and key parameters (thresholds, durations, weights) in a central config module or CLI options.
- Use reproducible environments (`requirements.txt` or `environment.yml`).
- Consider light tests with `pytest` for core routines (declustering, CF bounds).

## Next steps (suggested order)
1) Create `src/dunkelflaute/` and add `__init__.py`.
2) Refactor solar and wind CF logic into `capacity_factors.py` with clean functions.
3) Add thin driver scripts in `scripts/` that call those functions (no interactive prompts).
4) Refactor declustering/return-interval logic into `events.py` and add a script to run it.
5) Add minimal tests (e.g., synthetic series) to validate declustering and CF bounds.
6) Optionally add `environment.yml` if you prefer Conda, and a `.gitignore` to exclude large data/outputs.
