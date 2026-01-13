# Climate-Driven Risk of Dunkelflaute Events in the Netherlands

This repository analyzes the risk and reccurrenceof wind and solar generation droughts (“dunkelflaute”) over the Netherlands using climate datasets (e.g. ERA5). It converts meteorological variables into capacity factors, combines them into system-level metrics, and assesses the frequency, duration, and return intervals of low-generation events in daily timescales. The framework is designed to extend to future climate projections and different wind/solar mixes.

## Objectives
- Quantify historical dunkelflaute risk: frequency, duration, return intervals.
- Link climate variability to energy impacts via capacity factors and system mixes.
- Assess spatial heterogeneity of risk across the Netherlands and nearby regions.
- Explore optimization of location and/or proportions of solar/wind resources within the country to minimize energy shortfalls.
- Provide a baseline that can extend to future climate scenarios and alternative wind/solar deployments.

## Data sources
- ERA5 hourly reanalysis (wind components at 100 m, surface solar radiation, 2 m temperature).
- Future: CMIP6/ERA5-consistent projections for future scenarios.