#!/usr/bin/env python3
"""
Compute wind and solar capacity factors from ERA5 data.

Usage:
    python scripts/compute_capacity_factors.py --era5-data Data/ERA5/file.nc --save
"""

import argparse
import sys
from pathlib import Path
import xarray as xr

# Import our library functions
# Note: We need to add the src directory to Python's path so it can find our module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dunkelflaute.capacity_factors import compute_solar_cf, compute_wind_cf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute capacity factors from ERA5 data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--era5-data",
        type=str,
        required=True,
        help="ERA5 NetCDF input file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Results/",
        help="Output directory (default: Results/)"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to disk"
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    # Load ERA5 data
    print(f"Loading ERA5 data from {args.era5_data}...")
    ds = xr.open_mfdataset(args.era5_data)
    # DEBUG: Pause here to inspect 'ds' variable
    breakpoint()  # Uncomment this line to enable debugging

    # Extract variables
    print("Extracting variables...")
    ssrd = ds.ssrd  # Will be converted inside compute_solar_cf if needed
    t2m = ds.t2m    # Will be converted inside compute_solar_cf if needed
    u100 = ds.u100
    v100 = ds.v100
    
    # Compute capacity factors
    print("Computing solar capacity factor...")
    cf_solar = compute_solar_cf(ssrd, t2m)
    
    print("Computing wind capacity factor...")
    cf_wind = compute_wind_cf(u100, v100)
    
    # Save if requested
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        input_name = Path(args.era5_data).stem
        solar_path = output_dir / f"CF_solar_{input_name}.nc"
        wind_path = output_dir / f"CF_wind_{input_name}.nc"
        
        print(f"Saving solar CF to {solar_path}...")
        cf_solar.to_netcdf(solar_path)
        
        print(f"Saving wind CF to {wind_path}...")
        cf_wind.to_netcdf(wind_path)
        
        print("Done!")
    else:
        print("Computation complete. Use --save to write results to disk.")
        print(f"Solar CF shape: {cf_solar.shape}")
        print(f"Wind CF shape: {cf_wind.shape}")


if __name__ == "__main__":
    main()
