#!/bin/bash
#SBATCH --job-name=era5_df_dwnld
#SBATCH --array=1980-2024             # one task per year (adjust range as needed)
#SBATCH --output=logs/era5_df_%a.out  # relative to submission directory (dunkelflaute/Code/)
#SBATCH --error=logs/era5_df_%a.err
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                     # 6 variables × broader domain → ~2× memory vs ag
#SBATCH --time=03:00:00               # hourly, 6 vars, wider domain — allow extra time
#SBATCH --export=ALL                  # inherits ERA5_dat, MAMBA_EXE, MAMBA_ROOT_PREFIX

# ─────────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────────
# This SLURM array job mirrors SLURM_dwnld_arco_era5.sh from the 'ag'
# subproject but targets the dunkelflaute download script (00_dwnld_arco_era5.py
# in this directory).
#
# Each array task downloads one calendar year of:
#   u10, v10, u100, v100, ssrd, t2m
# for the extended European domain [N57, W8, S39, E19] and saves a compressed
# NetCDF file to $ERA5_dat/df_dat/era5_<YEAR>.nc
#
# Memory note: 6 variables × ~96 lon pts × ~73 lat pts × 8760 h ≈ 2.4 GB raw
# float32 per year; 64 GB allows comfortable dask overhead and multi-threading.
#
# Submit with:
#   cd dunkelflaute/Code
#   mkdir -p logs
#   sbatch SLURM_dwnld_arco_era5.sh
#
# To download a subset of years, adjust the --array range, e.g.:
#   #SBATCH --array=2000-2024

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

# SLURM_SUBMIT_DIR is always the directory where sbatch was invoked
SCRIPT_DIR=$SLURM_SUBMIT_DIR

# Initialise mamba using MAMBA_EXE and MAMBA_ROOT_PREFIX inherited from ~/.bashrc
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
mamba activate CE_env

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# ERA5_dat is inherited from ~/.bashrc via --export=ALL
# The Python script will create $ERA5_dat/df_dat/ automatically.

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$ERA5_dat"

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================"
echo "Project:      dunkelflaute"
echo "SLURM job:    $SLURM_JOB_ID"
echo "Array task:   $SLURM_ARRAY_TASK_ID  (year)"
echo "Node:         $SLURMD_NODENAME"
echo "Script dir:   $SCRIPT_DIR"
echo "Output dir:   $ERA5_dat/df_dat"
echo "Start:        $(date)"
echo "========================================"

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

python "$SCRIPT_DIR/00_dwnld_arco_era5.py" --year "$SLURM_ARRAY_TASK_ID"

EXIT_CODE=$?

echo "========================================"
echo "End:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
