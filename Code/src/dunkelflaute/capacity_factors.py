"""
Solar and wind capacity factor calculations.

Based on Brown et al. (2021) and Bett & Thornton (2016).
"""

import xarray as xr
import numpy as np

# Constants from Brown et al. (2021) and Bett & Thornton (2016)
# Solar PV parameters
ALPHA = 1.2e-13  # K^-1
BETA = -4.6e-3   # K^-1
C1 = 0.033
C2 = -0.0092
T_NOCT = 48      # °C - Nominal Operating Cell Temperature
T_0 = 25         # °C - Reference temperature
T_STC = 25       # °C - Standard Test Conditions temperature
G_0 = 800        # W/m² - Reference irradiance
G_STC = 1000     # W/m² - Standard Test Conditions irradiance

# Wind turbine parameters
P_R = 2000       # kW - Rated power (Vestas V90-2.0MW)


def compute_module_temperature(t2m: xr.DataArray, ssrd: xr.DataArray) -> xr.DataArray:
    """
    Calculate PV module temperature from ambient temperature and solar radiation.
    
    Parameters
    ----------
    t2m : xr.DataArray
        2-meter air temperature [°C]
    ssrd : xr.DataArray
        Surface solar radiation downwards [W/m²]
    
    Returns
    -------
    xr.DataArray
        Module temperature [°C]
    
    References
    ----------
    Brown et al. (2021), Bett & Thornton (2016)
    """
    return t2m + (T_NOCT - T_0) * (ssrd / G_0)


def compute_relative_efficiency(t2m: xr.DataArray, ssrd: xr.DataArray) -> xr.DataArray:
    """
    Calculate relative PV efficiency accounting for temperature and irradiance effects.
    
    Parameters
    ----------
    t2m : xr.DataArray
        2-meter air temperature [°C]
    ssrd : xr.DataArray
        Surface solar radiation downwards [W/m²]
    
    Returns
    -------
    xr.DataArray
        Relative efficiency (dimensionless)
    
    References
    ----------
    Brown et al. (2021), Bett & Thornton (2016)
    """
    T_mod = compute_module_temperature(t2m, ssrd)
    delta_T_mod = T_mod - T_STC
    G_prime = ssrd / G_STC
    log_G_prime = np.log(G_prime)
    
    return (1 + ALPHA * delta_T_mod) * \
           (1 + C1 * log_G_prime + C2 * (log_G_prime ** 2) + BETA * delta_T_mod)


def compute_solar_cf(ssrd: xr.DataArray, t2m: xr.DataArray) -> xr.DataArray:
    """
    Compute solar capacity factor from surface solar radiation and temperature.
    
    This function handles unit conversion and applies the PV efficiency model
    from Brown et al. (2021).
    
    Parameters
    ----------
    ssrd : xr.DataArray
        Surface solar radiation downwards. If in J/m², will be converted to W/m².
    t2m : xr.DataArray
        2-meter air temperature. If in Kelvin, will be converted to °C.
    
    Returns
    -------
    xr.DataArray
        Solar capacity factor (0-1)
    
    References
    ----------
    Brown et al. (2021), Bett & Thornton (2016)
    """
    # Convert units if needed (detect if ssrd is in J/m²)
    if ssrd.max() > 1000:  # Likely J/m² if max > 1000
        ssrd = ssrd / 3600  # Convert J/m² to W/m²
    
    # Convert temperature if needed (detect if in Kelvin)
    if t2m.min() > 200:  # Likely Kelvin if min > 200
        t2m = t2m - 273.15  # Convert K to °C
    
    # Compute relative efficiency
    eta_rel = xr.where(ssrd > 1, compute_relative_efficiency(t2m, ssrd), 0)
    
    # Compute capacity factor
    CF_solar = eta_rel * ssrd / G_STC
    
    return CF_solar


def power_curve(ws: xr.DataArray) -> xr.DataArray:
    """
    Vestas V90-2.0MW turbine power curve.
    
    Parameters
    ----------
    ws : xr.DataArray
        Wind speed [m/s]
    
    Returns
    -------
    xr.DataArray
        Power output [kW]
    
    References
    ----------
    Brown et al. (2021)
    """
    return (634.228 - 1248.5*ws + 999.57*(ws**2) - 426.224*(ws**3) +
            105.617*(ws**4) - 15.4587*(ws**5) + 1.3223*(ws**6) -
            0.0609186*(ws**7) + 0.00116265*(ws**8))


def compute_wind_cf(u100: xr.DataArray, v100: xr.DataArray) -> xr.DataArray:
    """
    Compute wind capacity factor from u and v wind components at 100m.
    
    Parameters
    ----------
    u100 : xr.DataArray
        U-component of wind at 100m [m/s]
    v100 : xr.DataArray
        V-component of wind at 100m [m/s]
    
    Returns
    -------
    xr.DataArray
        Wind capacity factor (0-1)
    
    References
    ----------
    Brown et al. (2021)
    """
    # Compute wind speed
    wind_100 = np.sqrt(u100**2 + v100**2)
    
    # Apply power curve with cut-in/cut-out limits
    WP = xr.where(wind_100 > 25, 0,
           xr.where(wind_100 > 13, P_R,
             xr.where(wind_100 < 3, 0,
               power_curve(wind_100))))
    
    # Convert to capacity factor
    CF_wind = WP / P_R
    
    return CF_wind
