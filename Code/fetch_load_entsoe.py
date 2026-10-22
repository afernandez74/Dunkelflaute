#%%
# Fetch national load and save daily mean MW.
# Run from dunkelflaute/Code.

from pathlib import Path
import time
import pandas as pd
from entsoe import EntsoePandasClient

client = EntsoePandasClient(
    api_key=Path.home().joinpath(".entsoe_token").read_text().strip(),
    timeout=60,
)

COUNTRIES = ["FR", "BE", "NL", "DE", "UK", "IE", "DK"]
START_YEAR = 2015
END_YEAR = 2019  # Inclusive; use 2019 for the calibration period only.

OUT = Path("../Data/entsoe_load")
OUT.mkdir(parents=True, exist_ok=True)

#%%
def fetch_one(code):
    chunks = []

    for year in range(START_YEAR, END_YEAR + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

        # Let errors stop execution rather than save a silently incomplete file.
        raw = client.query_load(code, start=start, end=end)

        if isinstance(raw, pd.DataFrame):
            raw = raw["Actual Load"]

        raw = raw.tz_convert("UTC").sort_index()
        raw = raw.loc[(raw.index >= start) & (raw.index < end)]

        if raw.empty:
            raise ValueError(f"{code}: no data returned for {year}")

        chunks.append(raw)
        print(f"  {code} {year}: {raw.notna().sum():,} valid observations")
        time.sleep(0.5)

    load = pd.concat(chunks).sort_index()

    if load.index.has_duplicates:
        raise ValueError(f"{code}: duplicate timestamps need inspection")

    daily = load.resample("1D").mean().rename("load")

    # Preserve missing days, including gaps at the record boundaries.
    dates = pd.date_range(
        f"{START_YEAR}-01-01",
        f"{END_YEAR + 1}-01-01",
        freq="D",
        inclusive="left",
        tz="UTC",
    )
    daily = daily.reindex(dates)

    # Remove timezone only AFTER aggregation into UTC days.
    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"

    path = OUT / f"load_{code}.parquet"
    daily.to_frame().to_parquet(path)

    print(
        f"Saved {path.name}: {len(daily):,} days, "
        f"{daily.isna().sum()} missing days, "
        f"mean={daily.mean():,.0f} MW\n"
    )


#%%

for code in COUNTRIES:
    print(f"Fetching {code}...")
    fetch_one(code)

print("Done.")
