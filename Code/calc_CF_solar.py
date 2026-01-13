#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:35:36 2025

@author: afer
"""

import os 
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#%% load 2 datasets: wind and temp files AND solar rad files

dat_path = os.path.expanduser('./../Data/ERA5/')

files = os.listdir(dat_path)

name_file = 'ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8'

file_path = dat_path + name_file

ds = xr.open_mfdataset(file_path)

ssrd = ds.ssrd / 3600 # J/m2 -> W/m2
t2m = ds.t2m - 273.15 # K to C 

#%% maximum coordinates in dataset

lat_min = ds['latitude'].min().item()
lat_max = ds['latitude'].max().item()
lon_min = ds['longitude'].min().item()
lon_max = ds['longitude'].max().item()

plot_region = [lon_min-1, lon_max+1, lat_min-1, lat_max+1]

#%% Calculate solar power potential 
 
# from Brown et al., 2021 (Bett and Thornton, 2016)

# constants: 
alpha = 1.2e-13 # K-1
beta = -4.6e-3 # K-1
c1 = 0.033
c2 = -0.0092

T_NOCT = 48 # C
T_0 = 25 # C
T_STC = 25 # C

G_0 = 800 # W*m-2
G_STC = 1000 # W*m-2


def T_mod_calc (t2m, ssrd, T_NOCT=T_NOCT, T_0=T_0, G_0=G_0):
    """
    Module temperature [C]
    """
    
    return t2m + (T_NOCT - T_0) * (ssrd / G_0)

def eta_rel_calc(t2m, ssrd, 
            alpha=alpha, beta=beta, c1=c1, c2=c2,
            T_STC=T_STC, G_STC=G_STC):
    """ 
    Relative efficiency of PV generation given weather conditions
    """
    
    T_mod = T_mod_calc (t2m, ssrd)
    delta_T_mod = T_mod - T_STC
    G_prime = ssrd/G_STC
    log_G_prime = np.log(G_prime)
    
    return (1 + alpha * delta_T_mod) *\
        (1 + c1 * log_G_prime + c2 * (log_G_prime ** 2) + beta * delta_T_mod)

# Calculate solar relative efficiency
eta_rel = xr.where(ssrd > 1, eta_rel_calc(t2m, ssrd), 0)

# Calculate Capacity Factor 
CF_solar = eta_rel * ssrd / G_STC

#%% save CF_solar

path_save = os.path.expanduser('./../Results/CF_solar/')
name = 'CF_solar_time_1980_2023_latlon_50_54_3_8.nc'
save_toggle = input("Save CF_solar results? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    CF_solar.to_netcdf(path_save+name)
