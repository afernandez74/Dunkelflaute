#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_DF_RL.py — Residual-Load Dunkelflaute Detection
Detects winter energy compound events based on residual load (Demand - Generation).
Tracks:
  T1 (Relative): RL > 90th percentile of winter distribution.
  T2 (Absolute): RL > 0 MW (Unserved energy).
"""
#%%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import kendalltau
from pathlib import Path

#%% 1. Configuration & Constants
WINTER_MONTHS = [11, 12, 1, 2, 3]  # NDJFM
THRESHOLD_PCT = 90                 # Upper tail for RL
MAX_GAP       = 1                  # Days to pool (Otero et al. 2022)
MIN_DURATION  = 3                  # Minimum event duration in days
COUNTRIES     = ['FR', 'BE', 'NL', 'DE', 'DK', 'GB', 'IE']

# Define Paths
IN_DIR = Path("./../Results")
OUT_DIR = Path("./../Results/DF_RL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#%% 2. Data Loading & Verification
print("Loading upstream data...")
capgen_df = pd.read_parquet(IN_DIR / "DF_supply/capgen_pot_absolute.parquet")
demand_df = pd.read_parquet(IN_DIR / "demand/demand_weather.parquet")
supply_events_df = pd.read_parquet(IN_DIR / "DF_supply/df_supply_events.parquet")


# Ensure exact index alignment (crucial for time series math)
assert demand_df.index.equals(capgen_df.index), "Time indices of Demand and Generation do not match!"
assert set(COUNTRIES).issubset(demand_df.columns), "Missing countries in demand data."

# Calculate Residual Load (RL = Demand - Generation)
rl_df = demand_df[COUNTRIES] - capgen_df[COUNTRIES]

# First Action: Print Summary Stats and Verify Zero-Crossing
print("\n--- Residual Load (RL) Summary Stats (MW) ---")
is_winter = rl_df.index.month.isin(WINTER_MONTHS)
rl_winter = rl_df[is_winter]

for country in COUNTRIES:
    mean_rl = rl_winter[country].mean()
    min_rl = rl_winter[country].min()
    max_rl = rl_winter[country].max()
    pct_positive = (rl_winter[country] > 0).mean() * 100
    print(f"{country}: Mean={mean_rl:,.0f}, Min={min_rl:,.0f}, Max={max_rl:,.0f}, Days > 0={pct_positive:.1f}%")
    
    # Sanity Check
    if pct_positive == 0:
        print(f"  [WARNING] {country} never crosses RL > 0. Track 2 will be empty.")
#%%
# =================================
# Event Detection and characterization functions 
# =================================
def merge_and_filter_runs_1d(is_event, max_gap=MAX_GAP, min_duration=MIN_DURATION):
    """
    Pools events separated by max_gap and filters by min_duration (Otero et al. 2022).
    """
    # Identify distinct runs
    run_starts = (is_event & ~is_event.shift(1, fill_value=False))
    run_ids = run_starts.cumsum()
    run_ids[~is_event] = 0
    
    # Pool gaps <= max_gap
    # (Simplified logic: if gap between valid run_ids is <= max_gap, bridge it)
    pooled_is_event = is_event.copy()
    gap_counter = 0
    for i in range(1, len(pooled_is_event)):
        if not pooled_is_event.iloc[i]:
            gap_counter += 1
        else:
            if 0 < gap_counter <= max_gap and pooled_is_event.iloc[i-gap_counter-1]:
                pooled_is_event.iloc[i-gap_counter:i] = True
            gap_counter = 0
            
    # Filter by duration
    pooled_starts = (pooled_is_event & ~pooled_is_event.shift(1, fill_value=False))
    pooled_ids = pooled_starts.cumsum()
    pooled_ids[~pooled_is_event] = 0
    
    run_lengths = pooled_ids.value_counts()
    valid_runs = run_lengths[run_lengths >= min_duration].index
    
    final_event_mask = pooled_ids.isin(valid_runs) & (pooled_ids > 0)
    return final_event_mask, pooled_ids

def characterize_events_1d(rl_series, dem_series, gen_series, event_mask, event_ids, threshold, winter_id):
    """Calculates severity, deficit, and ENS_GWh per event."""
    events = []
    unique_ids = event_ids[event_mask].unique()
    
    for eid in unique_ids:
        idx = event_ids == eid
        duration = idx.sum()
        rl_slice = rl_series[idx]
        
        # Calculate severity and metrics
        min_val = rl_slice.min()
        max_val = rl_slice.max() # Most extreme deficit
        mean_val = rl_slice.mean()
        
        # Severity = Summed shortfall relative to threshold
        severity_S = (rl_slice - threshold).clip(lower=0).sum()
        
        # Energy Not Served (ENS) in GWh: sum(max(Demand - Gen, 0)) * 24h / 1000
        # This maps directly to required storage/backup (Kittel & Schill 2024)
        dem_slice = dem_series[idx]
        gen_slice = gen_series[idx]
        ens_gwh = (dem_slice - gen_slice).clip(lower=0).sum() * 24 / 1000
        
        # Season attribution (majority rule)
        w_id = winter_id[idx].mode()[0]
        
        events.append({
            'start_date': rl_slice.index[0],
            'end_date': rl_slice.index[-1],
            'winter': w_id,
            'duration': duration,
            'max_rl_MW': max_val,
            'mean_rl_MW': mean_val,
            'severity_S': severity_S,
            'ENS_GWh': ens_gwh
        })
    return events

def detect_events_1d(rl_series, dem_series, gen_series, winter_mask, winter_id, track, tail='upper'):
    """Detects events based on track logic."""
    if track == 'T1':
        # Track 1: > 90th percentile of strictly winter days
        threshold = np.percentile(rl_series[winter_mask], THRESHOLD_PCT)
        raw_flags = (rl_series > threshold) & winter_mask
    elif track == 'T2':
        # Track 2: > 0 MW (absolute unserved energy)
        threshold = 0
        raw_flags = (rl_series > threshold) & winter_mask
        
    event_mask, event_ids = merge_and_filter_runs_1d(raw_flags)
    events = characterize_events_1d(rl_series, dem_series, gen_series, event_mask, event_ids, threshold, winter_id)
    return events, event_mask

#%%
# =================================
#  Execute Detection (Per Country & Copper-Plate)
# =================================

# Define Winter Mask and IDs
# Nov/Dec get pushed to the NEXT year's integer to form a continuous winter season
winter_id = pd.Series(rl_df.index.year, index=rl_df.index)
winter_id[rl_df.index.month.isin([11, 12])] += 1
is_winter = rl_df.index.month.isin(WINTER_MONTHS)

# To ensure we don't count incomplete edge winters (must have both Nov and Mar)
valid_winters = [y for y in winter_id.unique() 
                 if (rl_df[(winter_id == y) & (rl_df.index.month == 11)].shape[0] > 0 and 
                     rl_df[(winter_id == y) & (rl_df.index.month == 3)].shape[0] > 0)]

is_valid_winter_day = is_winter & winter_id.isin(valid_winters)

# Data structures
all_events = []
daily_event_masks = {'T1': pd.DataFrame(index=rl_df.index), 
                     'T2': pd.DataFrame(index=rl_df.index)}

# Per-Country Detection
for country in COUNTRIES:
    for track in ['T1', 'T2']:
        events, mask = detect_events_1d(
            rl_series=rl_df[country],
            dem_series=demand_df[country],
            gen_series=capgen_df[country],
            winter_mask=is_valid_winter_day,
            winter_id=winter_id,
            track=track,
            tail='upper'
        )
        
        # Append metadata
        for e in events:
            e['country'] = country
            e['track'] = track
        all_events.extend(events)
        daily_event_masks[track][country] = mask

# Copper-Plate Aggregate (Continental integration)
rl_eu = rl_df[COUNTRIES].sum(axis=1)
dem_eu = demand_df[COUNTRIES].sum(axis=1)
gen_eu = capgen_df[COUNTRIES].sum(axis=1)

for track in ['T1', 'T2']:
    events, mask = detect_events_1d(
        rl_series=rl_eu, dem_series=dem_eu, gen_series=gen_eu,
        winter_mask=is_valid_winter_day, winter_id=winter_id, track=track
    )
    for e in events:
        e['country'] = 'EU_AGG'
        e['track'] = track
    all_events.extend(events)
    daily_event_masks[track]['EU_AGG'] = mask

events_df = pd.DataFrame(all_events)
#%%
# =================================
# Metrics: Overlap and Concurrence
# =================================

# Jaccard Index: (Supply AND RL) / (Supply OR RL)
print("\n--- Event Overlap (Jaccard Index, Track 1 vs Supply Track 1) ---")
supply_t1 = supply_events_df[supply_events_df['track'] == 'T1'] # Assuming T1 exists in supply

overlap_data = []
for country in COUNTRIES:
    # Reconstruct daily binary masks from event catalogues
    sup_days = set()
    for _, row in supply_t1[supply_t1['country'] == country].iterrows():
        sup_days.update(pd.date_range(row['start_date'], row['end_date']).date)
        
    rl_days = set()
    rl_t1 = events_df[(events_df['country'] == country) & (events_df['track'] == 'T1')]
    for _, row in rl_t1.iterrows():
        rl_days.update(pd.date_range(row['start_date'], row['end_date']).date)
    
    intersection = len(sup_days.intersection(rl_days))
    union = len(sup_days.union(rl_days))
    jaccard = intersection / union if union > 0 else 0
    
    overlap_data.append({'country': country, 'jaccard': jaccard})
    print(f"{country}: Jaccard = {jaccard:.2f}")

# Concurrence: Fraction of countries experiencing an RL event per day
concurrence_t1 = daily_event_masks['T1'][COUNTRIES].sum(axis=1)
#%% 
# Annual Aggregation and Trends
# =================================

annual_metrics = []

for track in ['T1', 'T2']:
    for country in COUNTRIES + ['EU_AGG']:
        country_events = events_df[(events_df['country'] == country) & (events_df['track'] == track)]
        
        for w_id in valid_winters:
            w_events = country_events[country_events['winter'] == w_id]
            n_events = len(w_events)
            
            annual_metrics.append({
                'country': country,
                'winter': w_id,
                'track': track,
                'n_events': n_events,
                'total_dur': w_events['duration'].sum() if n_events > 0 else 0,
                'mean_dur': w_events['duration'].mean() if n_events > 0 else np.nan,
                'max_dur': w_events['duration'].max() if n_events > 0 else np.nan,
                'total_S': w_events['severity_S'].sum() if n_events > 0 else 0,
                'ENS_GWh': w_events['ENS_GWh'].sum() if n_events > 0 else 0
            })

annual_df = pd.DataFrame(annual_metrics)

# Trend Analysis (Mann-Kendall Tau Proxy)
print("\n--- Long-term Trends (Kendall's Tau) ---")
for country in COUNTRIES:
    # Analyzing Track 1 Intensity
    ts = annual_df[(annual_df['country'] == country) & (annual_df['track'] == 'T1')].sort_values('winter')
    tau_occ, p_occ = kendalltau(ts['winter'], ts['n_events'])
    tau_int, p_int = kendalltau(ts['winter'], ts['ENS_GWh'])
    
    print(f"{country} T1 - Occurrences: τ={tau_occ:.2f} (p={p_occ:.2f}), Intensity (ENS): τ={tau_int:.2f} (p={p_int:.2f})")
# %%
#%% 
# Figures
# =================================

# Fig 1: RL Time Series (DE vs DK) - Sample Winter (e.g., 2018)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
sample_winter = is_valid_winter_day & (winter_id == 2018)

for ax, country in zip(axes, ['DE', 'DK']):
    ax.plot(rl_df.index[sample_winter], rl_df.loc[sample_winter, country], color='black', lw=1)
    
    # 90th percentile line and Zero line
    thresh = np.percentile(rl_df.loc[is_valid_winter_day, country], THRESHOLD_PCT)
    ax.axhline(thresh, color='red', linestyle='--', label=f'T1 Threshold (90th Pct: {thresh:.0f} MW)')
    ax.axhline(0, color='blue', linestyle='-', label='T2 Threshold (0 MW / Unserved)')
    
    # Shade T1 Events
    mask_t1 = daily_event_masks['T1'].loc[sample_winter, country]
    ax.fill_between(rl_df.index[sample_winter], rl_df.loc[sample_winter, country], thresh, 
                    where=mask_t1, color='red', alpha=0.3, label='T1 Event')
                    
    ax.set_title(f"{country} Residual Load (Winter 2018)")
    ax.set_ylabel("Residual Load (MW)")
    ax.legend(loc='lower left')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
# plt.savefig(OUT_DIR / "fig_timeseries_DE_DK.svg")
plt.show()
plt.close()

# Fig 2: Duration vs Severity Scatter (Otero'22 copula setup)
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(COUNTRIES)))

for (country, color) in zip(COUNTRIES, colors):
    dat = events_df[(events_df['country'] == country) & (events_df['track'] == 'T1')]
    ax.scatter(dat['duration'], dat['severity_S'], alpha=0.6, label=country, color=color, edgecolors='k')

ax.set_title("Track 1: Duration vs Severity (All Winters)")
ax.set_xlabel("Duration (days)")
ax.set_ylabel("Severity (Summed Excess RL, MW)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
# plt.savefig(OUT_DIR / "fig_duration_severity_scatter.svg")
plt.show()

plt.close()

#%%
#  Outputs & Run Config
# =================================


# Export parquets
rl_df.to_parquet(OUT_DIR / "rl_country.parquet")
events_df.to_parquet(OUT_DIR / "df_rl_events.parquet")
annual_df.to_parquet(OUT_DIR / "df_rl_annual.parquet")

# JSON Configuration & Caveat manifest
run_config = {
    "parameters": {
        "WINTER_MONTHS": WINTER_MONTHS,
        "THRESHOLD_PCT": THRESHOLD_PCT,
        "MAX_GAP": MAX_GAP,
        "MIN_DURATION": MIN_DURATION
    },
    "provenance": {
        "demand": "demand_weather.parquet (vintage 2017-07-01)",
        "generation": "capgen_pot_df.parquet (Hu et al. 2023 maximal potential)"
    },
    "caveats": [
        "Vintage mismatch: Demand is frozen at REF, capacity is max technical potential. Implies structural surplus.",
        "Cold-extreme extrapolation: HDD response uses linear fit on 2015-2019; underestimates extreme freezing loads.",
        "Offshore CF: Assumes onshore Vestas V90 @100m curve; understates real offshore output.",
        "Spatial bounds: Hu dataset truncates at -10.5E, under-representing Irish EEZ offshore wind.",
        "Zone mismatch: Northern Ireland mapped to GB spatial mask but IE_SEM electricity market."
    ]
}

with open(OUT_DIR / "df_rl_runconfig.json", "w") as f:
    json.dump(run_config, f, indent=4)

print("\nPipeline complete. Assets written to:", OUT_DIR)
# %%
