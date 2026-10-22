#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_DF_RL_ID.py — Winter residual-load event analysis
===================================================

Inputs
------
Step 03:
    generation_irena_hu.parquet
    generation_max_hu.parquet
    DF_supply_events_all.csv

Step 04:
    demand_weather.parquet

Residual load:
    RL = demand - wind and solar generation

Deployment fleets:
    1. IRENA Hu: IRENA capacities distributed using Hu spatial weights.
    2. Max Hu: hypothetical maximum Hu deployment.

Event definitions:
    1. RL above a country- and fleet-specific winter percentile.
    2. RL above zero MW.

Positive residual load represents demand remaining after wind and solar
generation. It is not an estimate of unserved energy.

All input series represent daily mean power in MW.

Thresholds and standard deviations currently use all valid analysis
winters. A separate fixed reference period is not implemented here.
"""

#%%
# Imports
# =======

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.ndimage import label
from scipy.stats import kendalltau


#%%
# Config and params
# =================

WINTER_MONTHS = [11, 12, 1, 2, 3]

RL_PCT = 90
DEMAND_PCT = 90
RL_ABS_FLOOR = 0.0

MAX_GAP = 1
MIN_DURATION = 3

CAP_VINTAGE = 2024
DEMAND_REF = '2017-07-01'       # must match REF in step 04

CONCURRENCE_K = 5

SAMPLE_COUNTRY = 'FR'
SAMPLE_START = '2016-10-01'
SAMPLE_END = '2019-04-01'

COUNTRIES = {
    'France': 'FR',
    'Belgium': 'BE',
    'Netherlands': 'NL',
    'Germany': 'DE',
    'Denmark': 'DK',
    'United Kingdom': 'GB',
    'Ireland': 'IE',
}

# output_id preserves filenames currently expected by step 07.
FLEETS = {
    'IRENA Hu': {
        'generation_file': 'generation_irena_hu.parquet',
        'output_id': 'present_day',
        'label': f'IRENA Hu ({CAP_VINTAGE} capacities)',
    },
    'Max Hu': {
        'generation_file': 'generation_max_hu.parquet',
        'output_id': 'fully_built',
        'label': 'Max Hu (hypothetical maximum deployment)',
    },
}

# Supply definition used for comparison with RL episodes.
SUPPLY_THRESHOLD_TYPE = 'percentile'
SUPPLY_THRESHOLD_VALUE = 10.0

# These settings mirror the supplied step 03.
SUPPORTED_SUPPLY_THRESHOLDS = {
    'percentile': [1, 5, 10],
    'absolute': [0.01, 0.05, 0.10, 0.15, 0.20],
}

# Retain the relative-event identifier expected by step 07.
RL_REL_TRACK = f'RL_Relative_{RL_PCT:g}pct'
RL_ABS_TRACK = 'RL_Absolute_positive'
DEMAND_TRACK = f'DEM_Relative_{DEMAND_PCT:g}pct'

# Input / output: run with Code as the working directory.
SUPPLY_DIR = Path('./../Results/DF_supply')
DEMAND_PATH = Path('./../Results/demand/demand_weather.parquet')
SUPPLY_EVENTS_PATH = SUPPLY_DIR / 'DF_supply_events_all.csv'

OUT_DIR = Path('./../Results/DF_RL')
OUT_DIR.mkdir(parents=True, exist_ok=True)


#%%
# Input preparation
# =================

def prepare_daily_input(df, input_name):
    """Validate daily timestamps and select the study countries."""

    df = df.copy()
    df.index = pd.DatetimeIndex(df.index, name='date')

    if df.empty:
        raise ValueError(f'{input_name}: empty input.')

    if df.index.hasnans:
        raise ValueError(f'{input_name}: missing timestamps.')

    if not df.index.is_unique:
        raise ValueError(f'{input_name}: duplicate timestamps.')

    if not df.index.is_monotonic_increasing:
        raise ValueError(f'{input_name}: timestamps are not sorted.')

    if not df.columns.is_unique:
        raise ValueError(f'{input_name}: duplicate country columns.')

    country_codes = list(COUNTRIES.values())
    missing = set(country_codes) - set(df.columns)

    if missing:
        raise ValueError(
            f'{input_name}: missing countries {sorted(missing)}.'
        )

    time_steps = df.index.to_series().diff().dropna()

    if not time_steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            f'{input_name}: expected a continuous daily time series.'
        )

    df = df.loc[:, country_codes].astype(float)

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            f'{input_name}: missing or non-finite values.'
        )

    return df


#%%
# Load inputs and calculate residual load for both fleets
# =======================================================

demand_df = prepare_daily_input(
    pd.read_parquet(DEMAND_PATH),
    'Weather-driven demand'
)

supply_events = pd.read_csv(
    SUPPLY_EVENTS_PATH,
    parse_dates=['start', 'end']
)

required_supply_columns = {
    'country', 'track', 'source',
    'threshold_type', 'threshold_value',
    'start', 'end'
}

missing_columns = required_supply_columns - set(supply_events.columns)

if missing_columns:
    raise ValueError(
        'Supply catalogue is missing columns: '
        f'{sorted(missing_columns)}'
    )

if SUPPLY_THRESHOLD_TYPE not in SUPPORTED_SUPPLY_THRESHOLDS:
    raise ValueError('Unsupported supply threshold type.')

if not np.isclose(
    SUPPLY_THRESHOLD_VALUE,
    SUPPORTED_SUPPLY_THRESHOLDS[SUPPLY_THRESHOLD_TYPE]
).any():
    raise ValueError(
        'Selected supply threshold is not included in step 03.'
    )

generation = {}
residual_load = {}

for fleet_name, fleet_config in FLEETS.items():

    generation_path = (
        SUPPLY_DIR / fleet_config['generation_file']
    )

    generation_df = prepare_daily_input(
        pd.read_parquet(generation_path),
        f'{fleet_name} generation'
    )

    pd.testing.assert_index_equal(
        demand_df.index,
        generation_df.index,
        obj=f'Demand and {fleet_name} generation dates'
    )

    generation[fleet_name] = generation_df
    residual_load[fleet_name] = demand_df - generation_df

    print(f"\n{fleet_config['label']}")
    print(f'Loaded generation → {generation_path}')
    print('Mean residual load (MW):')
    print(
        residual_load[fleet_name]
        .mean()
        .round(0)
        .to_string()
    )

print(
    f'\nInputs: {len(demand_df)} days | '
    f'{len(COUNTRIES)} countries | {len(FLEETS)} fleets'
)
print(f'Demand background reference: {DEMAND_REF}')


#%%
# Event detection functions
# =========================

EVENT_COLUMNS = [
    'country',
    'track',
    'start_date',
    'end_date',
    'winter',
    'threshold',
    'duration',
    'peak_val',
    'mean_val',
    'severity_S',
    'excess_sum',
]


def merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """Pool nearby True runs and retain qualifying episode spans."""

    flag = np.asarray(flag, dtype=bool)

    if not flag.any():
        return np.zeros_like(flag, dtype=bool)

    runs = []
    in_run = False
    start = None

    for t in range(len(flag)):

        if flag[t] and not in_run:
            start = t
            in_run = True

        elif not flag[t] and in_run:
            runs.append([start, t - 1])
            in_run = False

    if in_run:
        runs.append([start, len(flag) - 1])

    merged = [runs[0]]

    for start, end in runs[1:]:

        gap = start - merged[-1][1] - 1

        if gap <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    out = np.zeros_like(flag, dtype=bool)

    for start, end in merged:

        if end - start + 1 >= min_duration:
            out[start:end + 1] = True

    return out


def detect_events_1d(
        x,
        mask_valid,
        thr_pct=None,
        abs_thr=None,
        max_gap=1,
        min_duration=3):
    """Detect upper-tail episodes using valid winter observations."""

    x = np.asarray(x, dtype=float)
    mask_valid = np.asarray(mask_valid, dtype=bool)

    if len(x) != len(mask_valid):
        raise ValueError('Series and winter mask lengths differ.')

    if not mask_valid.any():
        raise ValueError('No valid observations for event detection.')

    winter_values = x[mask_valid]

    if not np.isfinite(winter_values).all():
        raise ValueError(
            'Missing or non-finite values within analysis winters.'
        )

    sigma = float(np.std(winter_values))

    if abs_thr is not None:
        threshold = float(abs_thr)

    elif thr_pct is not None:
        threshold = float(
            np.percentile(winter_values, thr_pct)
        )

    else:
        raise ValueError('Specify a percentile or absolute threshold.')

    raw_flag = (x > threshold) & mask_valid

    event_mask = merge_and_filter_runs_1d(
        raw_flag,
        max_gap,
        min_duration
    )

    return event_mask, threshold, sigma


def characterize_events_1d(event_mask, x, threshold, sigma):
    """Calculate characteristics of upper-tail episode spans."""

    rows = []
    labels, n_events = label(event_mask)

    for event_id in range(1, n_events + 1):

        idx = np.flatnonzero(labels == event_id)

        start_idx = int(idx[0])
        end_idx = int(idx[-1])

        segment = x[start_idx:end_idx + 1]
        exceedance = np.clip(segment - threshold, 0, None)

        excess_sum = float(exceedance.sum())

        rows.append(dict(
            start_idx=start_idx,
            end_idx=end_idx,
            duration=int(len(segment)),
            peak_val=float(np.max(segment)),
            mean_val=float(np.mean(segment)),
            severity_S=(
                excess_sum / sigma if sigma > 0 else np.nan
            ),
            excess_sum=excess_sum
        ))

    return rows


def event_day_bool(events_df, index, countries):
    """Mark episode spans, including pooled gap days."""

    out = pd.DataFrame(
        False,
        index=index,
        columns=countries
    )

    for _, row in events_df.iterrows():

        if row['country'] in out.columns:

            span = (
                (index >= row['start_date'])
                & (index <= row['end_date'])
            )

            out.loc[span, row['country']] = True

    return out


def pairwise_jaccard(bool_df):
    """Calculate pairwise overlap; empty unions are undefined."""

    countries = list(bool_df.columns)

    matrix = pd.DataFrame(
        np.nan,
        index=countries,
        columns=countries
    )

    for country_a in countries:
        for country_b in countries:

            a = bool_df[country_a].to_numpy()
            b = bool_df[country_b].to_numpy()

            union = (a | b).sum()

            if union:
                matrix.loc[country_a, country_b] = (
                    (a & b).sum() / union
                )

    return matrix


def safe_correlation(a, b):
    """Return NaN when either input has no variability."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]

    if len(a) < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


#%%
# Analyse one deployment fleet
# ============================

def analyse_fleet(
        fleet_name,
        fleet_config,
        supply_df,
        demand_df,
        rl_df,
        supply_events):
    """Detect, characterize, plot and save results for one fleet."""

    output_id = fleet_config['output_id']
    fleet_label = fleet_config['label']
    countries = list(COUNTRIES.values())

    print(f'\nAnalysing {fleet_label}')

    # Winter masks
    # ------------

    times = rl_df.index
    months = times.month.to_numpy()
    years = times.year.to_numpy()

    winter_id = np.where(months >= 11, years + 1, years)
    is_ndjfm = np.isin(months, WINTER_MONTHS)

    # Retain winters covering the full November–March interval.
    # Input validation already requires continuous daily timestamps.
    valid_winters = []

    for winter in np.unique(winter_id[is_ndjfm]):

        first_day = pd.Timestamp(
            year=int(winter) - 1, month=11, day=1
        )
        last_day = pd.Timestamp(
            year=int(winter), month=3, day=31
        )

        dates = times[
            is_ndjfm & (winter_id == winter)
        ]

        if len(dates) == 0:
            continue

        expected_days = (last_day - first_day).days + 1

        if (
            dates[0].normalize() == first_day
            and dates[-1].normalize() == last_day
            and len(dates) == expected_days
        ):
            valid_winters.append(int(winter))

    if not valid_winters:
        raise ValueError('No complete analysis winters were found.')

    winter_ok = (
        is_ndjfm
        & np.isin(winter_id, valid_winters)
    )

    winter_dates = times[winter_ok]
    n_winters = len(valid_winters)

    print(
        f'Complete winters: {n_winters} '
        f'({valid_winters[0]} → {valid_winters[-1]})'
    )

    # Detection for a country dataframe
    # ---------------------------------

    def run_detection(df_series, track_name, threshold_type, value):

        all_events = []

        for country in countries:

            x = df_series[country].to_numpy(dtype=float)

            if threshold_type == 'percentile':

                event_mask, threshold, sigma = detect_events_1d(
                    x,
                    winter_ok,
                    thr_pct=value,
                    max_gap=MAX_GAP,
                    min_duration=MIN_DURATION
                )

            elif threshold_type == 'absolute':

                event_mask, threshold, sigma = detect_events_1d(
                    x,
                    winter_ok,
                    abs_thr=value,
                    max_gap=MAX_GAP,
                    min_duration=MIN_DURATION
                )

            else:
                raise ValueError('Unknown threshold type.')

            rows = characterize_events_1d(
                event_mask, x, threshold, sigma
            )

            for row in rows:

                start_idx = row.pop('start_idx')
                end_idx = row.pop('end_idx')

                all_events.append(dict(
                    country=country,
                    track=track_name,
                    start_date=times[start_idx],
                    end_date=times[end_idx],
                    winter=int(winter_id[start_idx]),
                    threshold=threshold,
                    **row
                ))

        return pd.DataFrame(
            all_events,
            columns=EVENT_COLUMNS
        )

    # Detect both RL definitions
    # --------------------------

    events_rl_rel = run_detection(
        rl_df,
        RL_REL_TRACK,
        'percentile',
        RL_PCT
    )

    events_rl_abs = run_detection(
        rl_df,
        RL_ABS_TRACK,
        'absolute',
        RL_ABS_FLOOR
    )

    master_rl_df = pd.concat(
        [events_rl_rel, events_rl_abs],
        ignore_index=True
    )

    print(
        master_rl_df.groupby('track').agg(
            n_events=('duration', 'size'),
            mean_duration=('duration', 'mean')
        ).to_string()
    )

    # Annual metrics, including event-free winters
    # -------------------------------------------

    annual_records = []

    full_index = pd.MultiIndex.from_product(
        [countries, valid_winters],
        names=['country', 'winter']
    )

    total_columns = [
        'n_events', 'total_dur', 'total_S', 'total_excess'
    ]

    for track_name in [RL_REL_TRACK, RL_ABS_TRACK]:

        selected = master_rl_df.loc[
            master_rl_df['track'].eq(track_name)
        ]

        if selected.empty:

            annual = pd.DataFrame(
                index=full_index,
                columns=[
                    'n_events', 'total_dur',
                    'mean_dur', 'max_dur',
                    'total_S', 'total_excess'
                ],
                dtype=float
            )

        else:

            annual = (
                selected.groupby(['country', 'winter'])
                .agg(
                    n_events=('duration', 'size'),
                    total_dur=('duration', 'sum'),
                    mean_dur=('duration', 'mean'),
                    max_dur=('duration', 'max'),
                    total_S=(
                        'severity_S',
                        lambda values: values.sum(min_count=1)
                    ),
                    total_excess=('excess_sum', 'sum')
                )
                .reindex(full_index)
            )

        no_events = annual['n_events'].isna()

        annual.loc[no_events, total_columns] = 0.0
        annual.loc[no_events, ['mean_dur', 'max_dur']] = np.nan

        annual['n_events'] = annual['n_events'].astype(int)
        annual['total_dur'] = annual['total_dur'].astype(int)
        annual['track'] = track_name

        annual_records.append(annual.reset_index())

    annual_df = pd.concat(
        annual_records,
        ignore_index=True
    )

    # Select the matching supply catalogue
    # -----------------------------------

    selection = (
        supply_events['track'].eq(fleet_name)
        & supply_events['source'].eq('CF')
        & supply_events['threshold_type'].eq(
            SUPPLY_THRESHOLD_TYPE
        )
        & np.isclose(
            supply_events['threshold_value'],
            SUPPLY_THRESHOLD_VALUE
        )
    )

    selected_supply = (
        supply_events.loc[selection]
        .rename(columns={
            'start': 'start_date',
            'end': 'end_date'
        })
        .copy()
    )

    print(
        f'Supply comparison: {fleet_name} | '
        f'{SUPPLY_THRESHOLD_TYPE} = '
        f'{SUPPLY_THRESHOLD_VALUE:g}'
    )
    print(f'Selected supply episodes: {len(selected_supply)}')

    sup_bool = event_day_bool(
        selected_supply,
        winter_dates,
        countries
    )

    rl_bool = event_day_bool(
        events_rl_rel,
        winter_dates,
        countries
    )

    # Supply–RL episode overlap
    # -------------------------

    overlap_rows = []

    for country in countries:

        a = sup_bool[country].to_numpy()
        b = rl_bool[country].to_numpy()

        both = int((a & b).sum())
        union = int((a | b).sum())

        overlap_rows.append(dict(
            country=country,
            supply_days=int(a.sum()),
            rl_days=int(b.sum()),
            both=both,
            supply_only=int((a & ~b).sum()),
            rl_only=int((b & ~a).sum()),
            jaccard=both / union if union else np.nan,
            frac_rl_in_supply=(
                both / b.sum() if b.sum() else np.nan
            ),
            frac_supply_in_rl=(
                both / a.sum() if a.sum() else np.nan
            )
        ))

    overlap_df = pd.DataFrame(overlap_rows)

    print('\nSupply–RL episode overlap:')
    print(overlap_df.round(3).to_string(index=False))

    # Positive RL: daily occurrence versus episode coverage
    # ----------------------------------------------------

    positive_episode_bool = event_day_bool(
        events_rl_abs,
        winter_dates,
        countries
    )

    positive_daily_bool = (
        rl_df.loc[winter_dates, countries] > 0
    )

    positive_rl_diagnostics = pd.DataFrame({
        'positive_RL_day_fraction': positive_daily_bool.mean(),
        'positive_RL_episode_day_fraction': (
            positive_episode_bool.mean()
        )
    })

    positive_rl_diagnostics.index.name = 'country'

    print('\nPositive residual-load occurrence:')
    print(positive_rl_diagnostics.round(3).to_string())

    # Winter variability and correlations
    # -----------------------------------

    variance_rows = []

    for country in countries:

        D = demand_df.loc[winter_dates, country].to_numpy()
        S = supply_df.loc[winter_dates, country].to_numpy()
        R = rl_df.loc[winter_dates, country].to_numpy()

        std_D = float(np.std(D))
        std_S = float(np.std(S))

        variance_rows.append(dict(
            country=country,
            std_D=std_D,
            std_S=std_S,

            # Relative variability, not a variance contribution.
            std_ratio=std_S / std_D if std_D > 0 else np.nan,

            corr_D_S=safe_correlation(D, S),
            corr_RL_S=safe_correlation(R, S),
            corr_RL_D=safe_correlation(R, D)
        ))

    var_df = pd.DataFrame(variance_rows)

    print('\nWinter variability and correlations:')
    print(var_df.round(3).to_string(index=False))

    variability_overlap = var_df.merge(
        overlap_df[['country', 'frac_rl_in_supply']],
        on='country'
    )

    overlap_correlation = safe_correlation(
        variability_overlap['std_ratio'],
        variability_overlap['frac_rl_in_supply']
    )

    print(
        '\nAcross-country correlation between σS/σD and '
        'the overlapping fraction of RL episode days: '
        f'{overlap_correlation:.3f}'
    )

    # High-demand episodes and overlap classification
    # -----------------------------------------------

    events_dem = run_detection(
        demand_df,
        DEMAND_TRACK,
        'percentile',
        DEMAND_PCT
    )

    dem_bool = event_day_bool(
        events_dem,
        winter_dates,
        countries
    )

    demand_overlap_rows = []
    classification_rows = []

    for country in countries:

        r = rl_bool[country].to_numpy()
        s = sup_bool[country].to_numpy()
        d = dem_bool[country].to_numpy()

        n = int(r.sum())

        demand_overlap_rows.append(dict(
            country=country,
            rl_in_supply=(
                (r & s).sum() / n if n else np.nan
            ),
            rl_in_demand=(
                (r & d).sum() / n if n else np.nan
            )
        ))

        classification_rows.append(dict(
            country=country,
            both=(r & s & d).sum() / n if n else np.nan,
            supply_only=(
                (r & s & ~d).sum() / n if n else np.nan
            ),
            demand_only=(
                (r & ~s & d).sum() / n if n else np.nan
            ),
            neither=(
                (r & ~s & ~d).sum() / n if n else np.nan
            )
        ))

    demand_overlap_df = pd.DataFrame(demand_overlap_rows)
    episode_classification = pd.DataFrame(classification_rows)

    print('\nRL overlap with supply and demand episodes:')
    print(demand_overlap_df.round(3).to_string(index=False))

    print('\nRL episode-day classification:')
    print(episode_classification.round(3).to_string(index=False))
    print(
        'Classification uses pooled episode spans, '
        'not simultaneous daily threshold exceedance.'
    )

    # Pairwise spatial concurrence
    # ----------------------------

    J_rl = pairwise_jaccard(rl_bool)
    J_sup = pairwise_jaccard(sup_bool)

    def offdiagonal_mean(matrix):

        values = matrix.to_numpy().copy()
        np.fill_diagonal(values, np.nan)

        finite = values[np.isfinite(values)]

        return finite.mean() if finite.size else np.nan

    print(
        f'\nMean pairwise RL Jaccard: '
        f'{offdiagonal_mean(J_rl):.3f}'
    )
    print(
        f'Mean pairwise supply Jaccard: '
        f'{offdiagonal_mean(J_sup):.3f}'
    )

    # Concurrence distribution and conditional concurrence
    # ---------------------------------------------------

    n_countries = len(countries)

    rl_conc = rl_bool.sum(axis=1)
    sup_conc = sup_bool.sum(axis=1)

    def active_distribution(concurrence):

        counts = (
            concurrence[concurrence > 0]
            .value_counts()
            .reindex(
                range(1, n_countries + 1),
                fill_value=0
            )
            .sort_index()
        )

        if counts.sum() == 0:
            return pd.Series(
                np.nan,
                index=counts.index,
                dtype=float
            )

        return counts / counts.sum()

    distribution_rl = active_distribution(rl_conc)
    distribution_supply = active_distribution(sup_conc)

    # Descriptive independence reference using winter-wide rates.
    marginal = rl_bool.mean()
    concurrence_rows = []

    for country in countries:

        active = rl_bool[country].to_numpy()

        expected = float(marginal.drop(country).sum())

        if active.any():

            observed = float(
                rl_bool.loc[active]
                .drop(columns=country)
                .sum(axis=1)
                .mean()
            )

        else:
            observed = np.nan

        concurrence_rows.append(dict(
            country=country,
            observed=observed,
            independent=expected,
            amplification=(
                observed / expected if expected > 0 else np.nan
            )
        ))

    cond_df = (
        pd.DataFrame(concurrence_rows)
        .set_index('country')
    )

    print('\nConditional concurrence:')
    print(cond_df.round(3).to_string())

    # Widespread RL episode days by winter
    # -----------------------------------

    winter_of_date = pd.Series(
        winter_id[winter_ok],
        index=winter_dates
    )

    widespread_day = rl_conc >= CONCURRENCE_K

    widespread_by_winter = (
        widespread_day.groupby(winter_of_date)
        .sum()
        .reindex(valid_winters, fill_value=0)
    )

    widespread_by_winter.index.name = 'winter'
    widespread_by_winter.name = 'widespread_episode_days'

    if len(valid_winters) > 1:

        tau, p_value = kendalltau(
            widespread_by_winter.index.to_numpy(),
            widespread_by_winter.to_numpy()
        )

    else:
        tau, p_value = np.nan, np.nan

    print(
        f'\nDays with ≥{CONCURRENCE_K} countries in RL episodes: '
        f'{int(widespread_by_winter.sum())} total | '
        f'{widespread_by_winter.mean():.1f} per winter'
    )
    print(f'Kendall tau = {tau:.3f} (p = {p_value:.3f})')

    # Country-level summary
    # ---------------------

    positive_rl_gwh = {}
    positive_rl_fraction = {}

    for country in countries:

        R = rl_df.loc[winter_dates, country].to_numpy()

        # All positive RL days, not only retained episodes.
        # Daily mean MW × 24 h / 1000 = GWh.
        positive_rl_gwh[country] = (
            np.clip(R, 0, None).sum()
            * 24 / 1e3
            / n_winters
        )

        positive_rl_fraction[country] = (R > 0).mean()

    summary = (
        var_df.set_index('country')[
            ['std_ratio', 'corr_D_S']
        ]
        .join(
            overlap_df.set_index('country')[
                ['frac_rl_in_supply', 'jaccard']
            ]
        )
        .join(
            cond_df[['amplification']].rename(columns={
                'amplification':
                    'concurrence_ratio_to_independence'
            })
        )
        .assign(
            positive_rl_gwh_per_winter=pd.Series(
                positive_rl_gwh
            ),
            positive_rl_day_fraction=pd.Series(
                positive_rl_fraction
            ),
            scenario=output_id,
            generation_track=fleet_name,
            demand_reference=DEMAND_REF,
            supply_threshold_type=SUPPLY_THRESHOLD_TYPE,
            supply_threshold_value=SUPPLY_THRESHOLD_VALUE
        )
    )

    summary.index.name = 'country'

    print('\nCountry-level summary:')
    print(summary.round(3).to_string())

    # Save outputs before plotting
    # ----------------------------

    rl_df.to_parquet(
        OUT_DIR / f'rl_series_{output_id}.parquet'
    )
    supply_df.to_parquet(
        OUT_DIR / f'supply_{output_id}.parquet'
    )
    master_rl_df.to_parquet(
        OUT_DIR / f'rl_events_{output_id}.parquet',
        index=False
    )
    annual_df.to_parquet(
        OUT_DIR / f'rl_annual_{output_id}.parquet',
        index=False
    )
    overlap_df.to_parquet(
        OUT_DIR / f'rl_supply_overlap_{output_id}.parquet',
        index=False
    )
    summary.to_parquet(
        OUT_DIR / f'scenario_summary_{output_id}.parquet'
    )

    print(f'\nSaved {fleet_name} results → {OUT_DIR}')

    # Plot 1: example RL time series
    # -----------------------------

    sample_start = pd.Timestamp(SAMPLE_START)
    sample_end = pd.Timestamp(SAMPLE_END)

    sample_mask = (
        (times >= sample_start)
        & (times <= sample_end)
    )

    if sample_mask.any():

        sample_threshold = np.percentile(
            rl_df.loc[winter_dates, SAMPLE_COUNTRY],
            RL_PCT
        )

        fig, ax = plt.subplots(figsize=(14, 4.5))

        ax.plot(
            times[sample_mask],
            rl_df.loc[times[sample_mask], SAMPLE_COUNTRY],
            color='k',
            lw=1,
            label='Daily mean residual load'
        )

        ax.axhline(
            sample_threshold,
            color='#d62728',
            ls='--',
            lw=1.5,
            label=f'Winter P{RL_PCT:g}'
        )

        ax.axhline(0, color='0.5', lw=0.8)

        for year in range(
                sample_start.year - 1,
                sample_end.year + 1):

            ax.axvspan(
                pd.Timestamp(year=year, month=11, day=1),
                pd.Timestamp(year=year + 1, month=4, day=1),
                color='gray',
                alpha=0.08
            )

        sample_events = events_rl_rel.loc[
            events_rl_rel['country'].eq(SAMPLE_COUNTRY)
            & (events_rl_rel['end_date'] >= sample_start)
            & (events_rl_rel['start_date'] <= sample_end)
        ]

        for _, event in sample_events.iterrows():

            ax.axvspan(
                event['start_date'],
                event['end_date'] + pd.Timedelta(days=1),
                color='#d62728',
                alpha=0.25
            )

        ax.set_xlim(sample_start, sample_end)
        ax.set_ylabel('Residual load (MW)')
        ax.set_title(
            f'{SAMPLE_COUNTRY} — upper-tail RL episodes\n'
            f'{fleet_label}',
            fontweight='bold',
            loc='left'
        )
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%b %Y')
        )
        ax.legend()

        plt.tight_layout()
        plt.show()

    # Plot 2: absolute episode overlap counts
    # --------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(overlap_df))

    ax.bar(
        x, overlap_df['both'],
        label='Overlapping',
        color='#4c72b0'
    )
    ax.bar(
        x, overlap_df['supply_only'],
        bottom=overlap_df['both'],
        label='Supply-only',
        color='#dd8452'
    )
    ax.bar(
        x, overlap_df['rl_only'],
        bottom=overlap_df['both'] + overlap_df['supply_only'],
        label='Residual-load-only',
        color='#c44e52'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(overlap_df['country'])
    ax.set_ylabel('Episode days, including pooled gaps')
    ax.set_title(
        f'Supply and residual-load episode overlap\n{fleet_label}',
        fontweight='bold'
    )
    ax.grid(axis='y', ls='--', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Plot 3: maximum daily mean RL versus accumulated exceedance
    # ---------------------------------------------------------

    ev = events_rl_rel.copy()
    ev['peak_GW'] = ev['peak_val'].astype(float) / 1e3
    ev['excess_GWd'] = ev['excess_sum'].astype(float) / 1e3

    fig, ax = plt.subplots(figsize=(8, 6))

    for country in countries:

        selected = ev.loc[ev['country'].eq(country)]

        ax.scatter(
            selected['peak_GW'],
            selected['excess_GWd'],
            s=20 + 6 * selected['duration'].astype(float),
            alpha=0.6,
            label=country
        )

    ax.set_xlabel('Maximum daily mean residual load (GW)')
    ax.set_ylabel(
        f'Accumulated exceedance above winter P{RL_PCT:g} '
        '(GW·days)'
    )
    ax.set_title(
        f'{fleet_label}\nPoint size indicates episode duration',
        fontweight='bold'
    )
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()

    # Plot 4: pairwise spatial concurrence
    # -----------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        constrained_layout=True
    )

    for ax, matrix, title in [
        (axes[0], J_sup, 'Supply episodes'),
        (axes[1], J_rl, 'Residual-load episodes')
    ]:

        plot_values = np.ma.masked_invalid(
            matrix.to_numpy(dtype=float)
        )

        colour_plot = ax.imshow(
            plot_values,
            cmap='Blues',
            vmin=0,
            vmax=1
        )

        for i in range(len(matrix)):
            for j in range(len(matrix)):

                value = matrix.iloc[i, j]
                text = f'{value:.2f}' if np.isfinite(value) else '—'

                ax.text(
                    j, i, text,
                    ha='center',
                    va='center',
                    color=(
                        'white'
                        if np.isfinite(value) and value >= 0.6
                        else 'black'
                    ),
                    fontsize=9
                )

        ax.set_xticks(range(len(matrix)))
        ax.set_yticks(range(len(matrix)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticklabels(matrix.index)
        ax.set_title(title, fontweight='bold')

    fig.colorbar(
        colour_plot,
        ax=axes,
        shrink=0.8,
        label='Episode-day Jaccard index'
    )
    fig.suptitle(fleet_label)

    plt.show()

    # Plot 6: widespread episode days by winter
    # ----------------------------------------

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.bar(
        widespread_by_winter.index,
        widespread_by_winter.to_numpy(),
        color='#4c72b0',
        edgecolor='k',
        width=0.7
    )

    if len(valid_winters) > 1:

        winter_numbers = np.asarray(valid_winters, dtype=float)
        relative_years = winter_numbers - winter_numbers[0]

        coefficients = np.polyfit(
            relative_years,
            widespread_by_winter.to_numpy(),
            1
        )

        ax.plot(
            winter_numbers,
            np.polyval(coefficients, relative_years),
            color='#c44e52',
            lw=2,
            label=(
                f'Linear fit; Kendall τ={tau:.2f}, '
                f'p={p_value:.3f}'
            )
        )

        ax.legend()

    ax.set_xlabel('Winter')
    ax.set_ylabel(
        f'Days with ≥{CONCURRENCE_K} countries in RL episodes'
    )
    ax.set_title(
        f'Widespread residual-load episodes\n{fleet_label}',
        fontweight='bold'
    )
    ax.grid(axis='y', ls='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 7: fractional episode overlap
    # ---------------------------------

    panel = (
        overlap_df.merge(
            var_df[['country', 'std_ratio']],
            on='country'
        )
        .sort_values('std_ratio')
        .reset_index(drop=True)
    )

    union = (
        panel['both']
        + panel['supply_only']
        + panel['rl_only']
    ).replace(0, np.nan)

    panel['f_both'] = panel['both'] / union
    panel['f_supply'] = panel['supply_only'] / union
    panel['f_rl'] = panel['rl_only'] / union

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(panel))

    ax.bar(
        x, panel['f_both'],
        color='#4c72b0',
        label='Overlapping'
    )
    ax.bar(
        x, panel['f_supply'],
        bottom=panel['f_both'],
        color='#dd8452',
        label='Supply-only'
    )
    ax.bar(
        x, panel['f_rl'],
        bottom=panel['f_both'] + panel['f_supply'],
        color='#c44e52',
        label='Residual-load-only'
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f'{country}\nσS/σD={ratio:.1f}'
        for country, ratio in zip(
            panel['country'],
            panel['std_ratio']
        )
    ])

    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of the union of episode days')
    ax.set_title(
        f'Supply and residual-load episode overlap\n{fleet_label}',
        fontweight='bold',
        loc='left'
    )
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    # Return results for cross-fleet comparisons
    # -----------------------------------------

    return {
        'events': master_rl_df,
        'annual': annual_df,
        'overlap': overlap_df,
        'summary': summary,
        'variability': var_df,
        'concurrence': cond_df,
        'positive_rl': positive_rl_diagnostics,
        'demand_overlap': demand_overlap_df,
        'episode_classification': episode_classification,
        'pairwise_rl': J_rl,
        'pairwise_supply': J_sup,
        'widespread_by_winter': widespread_by_winter,
    }


#%%
# Run the analysis for both deployment fleets
# ===========================================

results = {}

for fleet_name, fleet_config in FLEETS.items():

    results[fleet_name] = analyse_fleet(
        fleet_name=fleet_name,
        fleet_config=fleet_config,
        supply_df=generation[fleet_name],
        demand_df=demand_df,
        rl_df=residual_load[fleet_name],
        supply_events=supply_events
    )

print('\nCompleted residual-load analysis for both fleets.')


#%%
# Combine results across deployment fleets
# =========================================

events_rl_all = pd.concat(
    [
        fleet_results['events'].assign(fleet=fleet_name)
        for fleet_name, fleet_results in results.items()
    ],
    ignore_index=True
)

annual_rl_all = pd.concat(
    [
        fleet_results['annual'].assign(fleet=fleet_name)
        for fleet_name, fleet_results in results.items()
    ],
    ignore_index=True
)

overlap_rl_all = pd.concat(
    [
        fleet_results['overlap'].assign(fleet=fleet_name)
        for fleet_name, fleet_results in results.items()
    ],
    ignore_index=True
)

summary_rl_all = pd.concat(
    [
        fleet_results['summary']
        .reset_index()
        .assign(fleet=fleet_name)
        for fleet_name, fleet_results in results.items()
    ],
    ignore_index=True
)

events_rl_all.to_parquet(
    OUT_DIR / 'rl_events_all.parquet',
    index=False
)
annual_rl_all.to_parquet(
    OUT_DIR / 'rl_annual_all.parquet',
    index=False
)
overlap_rl_all.to_parquet(
    OUT_DIR / 'rl_supply_overlap_all.parquet',
    index=False
)
summary_rl_all.to_parquet(
    OUT_DIR / 'rl_summary_all.parquet',
    index=False
)

# Use the annual table so zero-event definitions remain visible.
event_counts = (
    annual_rl_all.groupby(['fleet', 'track'])['n_events']
    .sum()
    .rename('n_events')
)

mean_durations = (
    events_rl_all.groupby(['fleet', 'track'])['duration']
    .mean()
    .rename('mean_duration')
)

comparison = event_counts.to_frame().join(mean_durations)

print('\nEvent counts and mean duration:')
print(comparison.round(2).to_string())

print(f'\nSaved individual and combined results → {OUT_DIR}')
# %%
#%%
# Compare spatial concurrence of RL episodes between fleets
# =========================================================

# Use the same complete winters for both fleets.
comparison_winters = sorted(
    results['IRENA Hu']['annual']['winter'].unique()
)

other_winters = sorted(
    results['Max Hu']['annual']['winter'].unique()
)

if comparison_winters != other_winters:
    raise ValueError(
        'Both fleets must use the same winters for comparison.'
    )

dates = demand_df.index

winter_id = np.where(
    dates.month >= 11,
    dates.year + 1,
    dates.year
)

winter_mask = (
    np.isin(dates.month, WINTER_MONTHS)
    & np.isin(winter_id, comparison_winters)
)

winter_dates = dates[winter_mask]
countries = list(COUNTRIES.values())
n_countries = len(countries)

concurrence_distributions = {}
active_day_counts = {}

for fleet_name in FLEETS:

    # Select only upper-tail residual-load episodes.
    events = results[fleet_name]['events']

    events = events.loc[
        events['track'].eq(RL_REL_TRACK)
    ].copy()

    episode_days = event_day_bool(
        events,
        winter_dates,
        countries
    )

    # Number of countries within an RL episode on each winter day.
    concurrence = episode_days.sum(axis=1)

    # Preserve the original denominator:
    # days with at least one country in an RL episode.
    active_days = concurrence[concurrence > 0]
    active_day_counts[fleet_name] = len(active_days)

    counts = (
        active_days.value_counts()
        .reindex(
            range(1, n_countries + 1),
            fill_value=0
        )
        .sort_index()
    )

    if active_days.empty:
        distribution = pd.Series(
            np.nan,
            index=counts.index
        )
        print(f'{fleet_name}: no active RL episode days.')
    else:
        distribution = counts / len(active_days)

    concurrence_distributions[fleet_name] = distribution

concurrence_comparison = pd.DataFrame(
    concurrence_distributions
)

concurrence_comparison.index.name = 'countries_in_RL_episode'

# Plot both deployment scenarios together.
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(1, n_countries + 1)
width = 0.36

ax.bar(
    x - width / 2,
    concurrence_comparison['IRENA Hu'],
    width=width,
    color='#4c72b0',
    label='Present-day capacities (IRENA Hu)'
)

ax.bar(
    x + width / 2,
    concurrence_comparison['Max Hu'],
    width=width,
    color='#dd8452',
    label='Fully-built potential (Max Hu)'
)

ax.set_xticks(x)
ax.set_xlabel('Countries simultaneously within an RL episode')
ax.set_ylabel('Fraction of days with ≥1 active RL episode')

ax.set_title(
    f'Spatial concurrence of residual-load episodes\n'
    f'Fleet-specific winter P{RL_PCT:g} thresholds',
    fontweight='bold'
)

ax.grid(axis='y', ls='--', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(frameon=False)

plt.tight_layout()
plt.show()

print('\nActive RL episode days used as denominators:')
print(pd.Series(active_day_counts).to_string())

concurrence_comparison.to_csv(
    OUT_DIR / 'rl_concurrence_fleet_comparison.csv'
)
#%%
# RL episodes relative to a shared IRENA Hu P90 threshold
# ======================================================

country_styles = {
    'FR': ('#0072B2', 'o'),
    'BE': ('#E69F00', 's'),
    'NL': ('#009E73', '^'),
    'DE': ('#D55E00', 'D'),
    'DK': ('#CC79A7', 'v'),
    'GB': ('#56B4E9', 'P'),
    'IE': ('#333333', 'X'),
}

# Use the same complete winters as the IRENA Hu analysis.
reference_rl = residual_load['IRENA Hu']
dates = reference_rl.index

analysis_winters = sorted(
    results['IRENA Hu']['annual']['winter'].unique()
)

winter_id = np.where(
    dates.month >= 11,
    dates.year + 1,
    dates.year
)

winter_mask = (
    np.isin(dates.month, WINTER_MONTHS)
    & np.isin(winter_id, analysis_winters)
)

# Country-specific reference thresholds, identical across fleets.
reference_p90 = reference_rl.loc[winter_mask].quantile(0.90)

# Use the same reference variability if retaining severity_S.
reference_sigma = reference_rl.loc[winter_mask].std(ddof=0)

print('IRENA Hu winter P90 thresholds applied to both fleets (MW):')
print(reference_p90.round(1).to_string())

# Re-detect episodes using the shared thresholds.
plot_tables = {}

for fleet_name in FLEETS:

    fleet_rl = residual_load[fleet_name]

    pd.testing.assert_index_equal(
        reference_rl.index,
        fleet_rl.index
    )

    event_rows = []

    for country in country_styles:

        x = fleet_rl[country].to_numpy(dtype=float)
        threshold = float(reference_p90[country])

        # Supply a fixed MW threshold to the detection function.
        # Its value was derived from the IRENA Hu winter P90.
        event_mask, _, _ = detect_events_1d(
            x,
            winter_mask,
            abs_thr=threshold,
            max_gap=MAX_GAP,
            min_duration=MIN_DURATION
        )

        country_events = characterize_events_1d(
            event_mask,
            x,
            threshold,
            float(reference_sigma[country])
        )

        for event in country_events:

            start_idx = event.pop('start_idx')
            end_idx = event.pop('end_idx')

            event_rows.append(dict(
                country=country,
                fleet=fleet_name,
                start_date=dates[start_idx],
                end_date=dates[end_idx],
                duration=event['duration'],
                threshold_MW=threshold,

                # MW-days × 24 h / 1000 = GWh.
                severity_GWh=event['excess_sum'] * 24 / 1000
            ))

    plot_tables[fleet_name] = pd.DataFrame(
        event_rows,
        columns=[
            'country', 'fleet', 'start_date', 'end_date',
            'duration', 'threshold_MW', 'severity_GWh'
        ]
    )

# Plot both fleets with the same axes and country styling.
fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5.5),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

for ax, fleet_name in zip(axes, FLEETS):

    selected = plot_tables[fleet_name]

    for country, (colour, marker) in country_styles.items():

        country_events = selected.loc[
            selected['country'].eq(country)
        ]

        ax.scatter(
            country_events['duration'].astype(float),
            country_events['severity_GWh'].astype(float),
            color=colour,
            marker=marker,
            s=45,
            alpha=0.65,
            edgecolors='white',
            linewidths=0.4,
            label=country
        )

    if selected.empty:
        ax.text(
            0.5, 0.5,
            'No qualifying episodes above the reference threshold',
            transform=ax.transAxes,
            ha='center',
            va='center',
            wrap=True
        )

    ax.set_title(
        FLEETS[fleet_name]['label'],
        fontweight='bold',
        fontsize=11
    )
    ax.set_xlabel('Episode duration, including pooled gaps (days)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(ls='--', alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel(
    'Cumulative RL exceedance above IRENA Hu P90 (GWh)'
)

axes[1].legend(
    title='Country',
    frameon=False,
    fontsize=9
)

fig.suptitle(
    'Residual-load episodes above a shared reference threshold\n'
    'Country-specific IRENA Hu winter P90 applied to both fleets',
    fontsize=13
)

figure_dir = OUT_DIR / 'figures'
figure_dir.mkdir(parents=True, exist_ok=True)

for extension in ['pdf', 'png']:
    fig.savefig(
        figure_dir / f'RL_reference_P90_duration_severity.{extension}',
        dpi=300,
        bbox_inches='tight'
    )

plt.show()

# Save this catalogue separately from fleet-specific P90 results.
reference_p90_events = pd.concat(
    plot_tables.values(),
    ignore_index=True
)

reference_p90_events.to_csv(
    OUT_DIR / 'rl_events_shared_irena_p90.csv',
    index=False
)
# %%
