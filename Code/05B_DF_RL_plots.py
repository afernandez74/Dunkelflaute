#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05B_RL_severity_heatmaps.py
=========================

Compare winter residual-load episode severity for:
    1. IRENA Hu
    2. Max Hu

Outputs:
    - Two fleet heatmaps with a shared colour scale.
    - Relative-change heatmap.
    - Absolute-change heatmap.
    - Annual metrics, thresholds and plotted values as CSV files.

Thresholds use all complete analysis winters, matching the current
step 05. Positive RL and threshold exceedance are not unserved energy.

Run with dunkelflaute/Code as the working directory.
"""

#%%
# Imports and configuration
# =========================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent

INPUT = ROOT / 'Results/DF_RL'
DEMAND = ROOT / 'Results/demand/demand_weather.parquet'
OUT = ROOT / 'Figures/RL_heatmaps'

COUNTRIES = ['FR', 'BE', 'NL', 'DE', 'DK', 'GB', 'IE']

FLEETS = {
    'IRENA Hu': 'present_day',
    'Max Hu': 'fully_built',
}

PERCENTILES = [85, 90, 95, 97.5]

MAX_GAP = 1
MIN_DURATION = 3

# baseline_percentile:
#   IRENA Hu percentile thresholds are applied to BOTH fleets.
#
# fleet_percentile:
#   Each fleet uses its OWN percentile thresholds.
THRESHOLD_MODE = 'baseline_percentile'

# demand_days: equivalent mean-winter-demand days
# gwh: absolute accumulated exceedance in GWh
METRIC = 'demand_days'

SHOW = True


#%%
# Load and validate daily inputs
# ==============================

def read_daily(path):

    df = pd.read_parquet(path).copy()
    df.index = pd.DatetimeIndex(df.index, name='date')

    if df.empty or df.index.hasnans or not df.index.is_unique:
        raise ValueError(f'Empty input or invalid dates: {path}')

    if not df.index.is_monotonic_increasing:
        raise ValueError(f'Unsorted dates: {path}')

    time_steps = df.index.to_series().diff().dropna()

    if not time_steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError(f'Non-continuous daily input: {path}')

    if (
        not df.columns.is_unique
        or not set(COUNTRIES).issubset(df.columns)
    ):
        raise ValueError(
            f'Missing or duplicate country columns: {path}'
        )

    df = df.loc[:, COUNTRIES].astype(float)

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            f'Missing or non-finite observations: {path}'
        )

    return df


#%%
# Episode metrics for one complete winter
# =======================================

def episode_metrics(x, threshold):
    """
    Pool gaps and retain episodes with the required minimum span.

    Duration includes gap days.
    Accumulated exceedance includes only values above the threshold.
    """

    x = np.asarray(x, dtype=float)
    flag = x > threshold

    edges = np.diff(
        np.r_[False, flag, False].astype(int)
    )

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1

    merged = []

    for start, end in zip(starts, ends):

        if merged and start - merged[-1][1] - 1 <= MAX_GAP:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    spans = [
        (start, end)
        for start, end in merged
        if end - start + 1 >= MIN_DURATION
    ]

    excess = sum(
        np.maximum(x[start:end + 1] - threshold, 0).sum()
        for start, end in spans
    )

    event_days = sum(
        end - start + 1
        for start, end in spans
    )

    return len(spans), event_days, float(excess)


#%%
# Identify complete winters
# =========================

def complete_winter_masks(index):

    winter_id = np.where(
        index.month >= 11,
        index.year + 1,
        index.year
    )

    seasonal = np.isin(
        index.month,
        [11, 12, 1, 2, 3]
    )

    masks = {}

    for winter in np.unique(winter_id[seasonal]):

        first = pd.Timestamp(int(winter) - 1, 11, 1)
        last = pd.Timestamp(int(winter), 3, 31)

        mask = seasonal & (winter_id == winter)

        dates = index[mask].normalize()
        expected = pd.date_range(first, last, freq='D')

        if dates.equals(expected):
            masks[int(winter)] = mask

    if not masks:
        raise ValueError('No complete winters.')

    return masks


#%%
# Heatmap helpers
# ===============

def draw_heatmap(
        ax, table, cmap, vmin, vmax, signed=False):

    values = table.to_numpy(dtype=float)

    palette = plt.get_cmap(cmap).copy()
    palette.set_bad('#e6e6e6')

    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=palette,
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )

    ax.set_xticks(
        range(len(table.columns)),
        labels=table.columns
    )
    ax.set_yticks(
        range(len(table.index)),
        labels=table.index
    )

    ax.set_xlabel('RL detection threshold')

    ax.set_xticks(
        np.arange(-0.5, len(table.columns), 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, len(table.index), 1),
        minor=True
    )

    ax.grid(
        which='minor',
        color='white',
        linewidth=1
    )
    ax.tick_params(
        which='minor',
        bottom=False,
        left=False
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):

            value = values[i, j]

            if not np.isfinite(value):
                text = 'n/a'
                colour = '#333333'

            else:
                text = (
                    f'{value:+.1f}'
                    if signed
                    else f'{value:.2g}'
                )

                rgb = palette(image.norm(value))[:3]

                luminance = np.dot(
                    rgb,
                    [0.2126, 0.7152, 0.0722]
                )

                colour = (
                    'white' if luminance < 0.45 else 'black'
                )

            ax.text(
                j, i, text,
                ha='center',
                va='center',
                color=colour,
                fontsize=9
            )

    return image


def save_figure(fig, stem):

    for extension in ['pdf', 'png']:

        fig.savefig(
            OUT / f'{stem}.{extension}',
            dpi=300,
            bbox_inches='tight'
        )

    if SHOW:
        plt.show()

    plt.close(fig)


#%%
# Calculate annual severity and plot comparisons
# ==============================================

def main():

    if THRESHOLD_MODE not in [
        'baseline_percentile',
        'fleet_percentile'
    ]:
        raise ValueError('Unknown THRESHOLD_MODE.')

    if METRIC not in ['demand_days', 'gwh']:
        raise ValueError('Unknown METRIC.')

    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # Load daily data and verify the RL identity.
    demand = read_daily(DEMAND)
    residual = {}

    for fleet, identifier in FLEETS.items():

        residual[fleet] = read_daily(
            INPUT / f'rl_series_{identifier}.parquet'
        )

        generation = read_daily(
            INPUT / f'supply_{identifier}.parquet'
        )

        pd.testing.assert_index_equal(
            demand.index,
            residual[fleet].index
        )
        pd.testing.assert_index_equal(
            demand.index,
            generation.index
        )

        if not np.allclose(
            residual[fleet],
            demand - generation,
            rtol=1e-8,
            atol=1e-5
        ):
            raise ValueError(
                f'{fleet}: RL differs from demand minus generation. '
                'Check input vintages.'
            )

    # Use exactly the same complete winters for both fleets.
    masks = complete_winter_masks(demand.index)

    valid = np.logical_or.reduce(
        list(masks.values())
    )

    mean_demand = demand.loc[valid].mean()

    if (mean_demand <= 0).any():
        raise ValueError(
            'Normalisation requires positive mean winter demand.'
        )

    columns = ['0 MW'] + [
        f'P{percentile:g}'
        for percentile in PERCENTILES
    ]

    annual_rows = []
    threshold_rows = []

    for fleet, rl in residual.items():

        if THRESHOLD_MODE == 'baseline_percentile':
            reference = residual['IRENA Hu']
        else:
            reference = rl

        for country in COUNTRIES:

            thresholds = [0.0] + list(
                np.percentile(
                    reference.loc[valid, country],
                    PERCENTILES
                )
            )

            for name, threshold in zip(columns, thresholds):

                threshold_rows.append(dict(
                    fleet=fleet,
                    country=country,
                    definition=name,
                    threshold_MW=threshold
                ))

                for winter, mask in masks.items():

                    x = rl.loc[mask, country].to_numpy()

                    count, days, excess = episode_metrics(
                        x, threshold
                    )

                    annual_rows.append(dict(
                        fleet=fleet,
                        country=country,
                        definition=name,
                        winter=winter,
                        n_events=count,
                        event_days=days,

                        # Daily mean MW × 24 h / 1000 = GWh.
                        excess_gwh=24 * excess / 1000,

                        # MW-days / mean demand MW = demand-days.
                        excess_demand_days=(
                            excess / mean_demand[country]
                        ),

                        # Unfiltered daily exceedance, for comparison
                        # with retained-episode exceedance.
                        raw_exceedance_gwh=(
                            24
                            * np.maximum(x - threshold, 0).sum()
                            / 1000
                        )
                    ))

    annual = pd.DataFrame(annual_rows)
    thresholds = pd.DataFrame(threshold_rows)

    if METRIC == 'demand_days':
        metric = 'excess_demand_days'
        unit = 'Equivalent mean-winter-demand days per winter'
    else:
        metric = 'excess_gwh'
        unit = 'GWh per winter'

    # Every accepted winter is present, including zero-event winters.
    means = (
        annual.groupby(
            ['fleet', 'country', 'definition']
        )[metric]
        .mean()
    )

    tables = {
        fleet: (
            means.loc[fleet]
            .unstack('definition')
            .reindex(
                index=COUNTRIES,
                columns=columns
            )
        )
        for fleet in FLEETS
    }

    present = tables['IRENA Hu']
    maximum = tables['Max Hu']

    absolute_change = maximum - present

    # Undefined when baseline severity is zero.
    relative_change = (
        100 * absolute_change
        / present.where(present > 0)
    )

    # Save source tables.
    stem = (
        f'RL_severity_{THRESHOLD_MODE}_{METRIC}'
    )

    annual.to_csv(
        OUT / f'{stem}_annual.csv',
        index=False
    )
    thresholds.to_csv(
        OUT / f'{stem}_thresholds_MW.csv',
        index=False
    )

    for fleet, table in tables.items():

        table.to_csv(
            OUT / f'{stem}_{FLEETS[fleet]}_mean.csv'
        )

    absolute_change.to_csv(
        OUT / f'{stem}_absolute_change.csv'
    )
    relative_change.to_csv(
        OUT / f'{stem}_relative_change_pct.csv'
    )

    if THRESHOLD_MODE == 'baseline_percentile':

        note = (
            'Pxx denotes IRENA Hu winter percentiles, '
            'applied to both fleets.'
        )

    else:

        note = (
            "Pxx denotes each fleet's own winter percentiles; "
            'physical thresholds differ.'
        )

    note += (
        '\nAll complete winters; pooled gaps ≤1 day; '
        'episode span ≥3 days.'
    )

    # Figure 1: fleet comparison with a shared colour scale.
    colour_max = max(
        float(table.to_numpy().max())
        for table in tables.values()
    )

    if colour_max == 0:
        colour_max = 1.0

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5.4),
        layout='constrained'
    )

    for ax, (fleet, table) in zip(
        axes, tables.items()
    ):

        image = draw_heatmap(
            ax,
            table,
            cmap='YlOrRd',
            vmin=0,
            vmax=colour_max
        )

        ax.set_title(fleet)

    axes[0].set_ylabel('Country')

    fig.colorbar(
        image,
        ax=axes,
        label=unit,
        shrink=0.85
    )

    fig.suptitle(
        'Mean annual cumulative RL exceedance within episodes\n'
        + note,
        fontsize=11
    )

    save_figure(fig, stem + '_fleets')

    # Figures 2 and 3: relative and absolute changes.
    change_figures = [
        (
            relative_change,
            'relative_change',
            'Change relative to IRENA Hu (%)'
        ),
        (
            absolute_change,
            'absolute_change',
            f'Max Hu minus IRENA Hu ({unit})'
        ),
    ]

    for table, suffix, colour_label in change_figures:

        values = table.to_numpy()
        finite = values[np.isfinite(values)]

        limit = (
            float(np.max(np.abs(finite)))
            if finite.size
            else 1.0
        )

        if limit == 0:
            limit = 1.0

        fig, ax = plt.subplots(
            figsize=(8, 5.4),
            layout='constrained'
        )

        image = draw_heatmap(
            ax,
            table,
            cmap='RdBu_r',
            vmin=-limit,
            vmax=limit,
            signed=True
        )

        ax.set_ylabel('Country')
        ax.set_title(
            'Change in mean annual RL episode exceedance\n'
            + note,
            fontsize=10
        )

        fig.colorbar(
            image,
            ax=ax,
            label=colour_label
        )

        if suffix == 'relative_change':

            fig.supxlabel(
                'Blue: decrease. Red: increase. '
                'n/a: zero baseline severity.',
                fontsize=9
            )

        save_figure(fig, stem + '_' + suffix)

    print(
        f'Complete winters: {min(masks)}–{max(masks)} '
        f'({len(masks)})'
    )

    print('\nDetection thresholds (MW):')
    print(thresholds.to_string(index=False))

    print('\nRelative change (%):')
    print(relative_change.round(1).to_string())

    print(f'\nSaved figures and source tables → {OUT}')


#%%
# Execute
# =======

if __name__ == '__main__':
    main()
# %%
