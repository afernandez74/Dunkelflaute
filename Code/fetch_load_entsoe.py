#%%
# fetch_entsoe_load.py — run once; writes ./../Data/entsoe_load/load_{CODE}.parquet
import os, pandas as pd, time
from pathlib import Path
from entsoe import EntsoePandasClient

# access token for ENTSOE API
os.environ["ENTSOE_API"] = Path.home().joinpath(".entsoe_token").read_text().strip()


#%%
client = EntsoePandasClient(api_key=os.environ["ENTSOE_API"])   # your security token
ZONES = {
    "FR":  ("FR",      "Europe/Paris",      2015),
    "BE":  ("BE",      "Europe/Brussels",   2015),
    "NL":  ("NL",      "Europe/Amsterdam",  2015),
    "DE":  ("DE_LU",   "Europe/Berlin",     2015),   # falls back to DE_AT_LU pre-2018
    "GB":  ("GB",      "Europe/London",     2015),
    "IE":  ("IE_SEM",  "Europe/Dublin",     2017),   # SEM data thin before 2017
    "DK":  ("DK",      "Europe/Copenhagen", 2015),
}
out = Path("./../Data/entsoe_load"); out.mkdir(parents=True, exist_ok=True)

#%%
def fetch_one(zone_code, tz, start_yr):
    chunks = []
    for yr in range(start_yr, 2024):
        s = pd.Timestamp(f"{yr}0101", tz=tz)
        e = pd.Timestamp(f"{yr+1}0101", tz=tz)
        try:
            raw = client.query_load(zone_code, start=s, end=e)
        except Exception as ex:
            # Germany: try old zone code for pre-2018 years
            if zone_code == "DE_LU" and yr < 2019:
                try:
                    raw = client.query_load("DE_AT_LU", start=s, end=e)
                except Exception as ex2:
                    print(f"  {zone_code} {yr}: {ex2}"); continue
            else:
                print(f"  {zone_code} {yr}: {ex}"); continue
        # result is a Series or a DataFrame (DK sometimes returns two columns)
        if isinstance(raw, pd.DataFrame):
            raw = raw.sum(axis=1)           # DK: sum DK1 + DK2
        raw.name = "load"
        chunks.append(raw)
        time.sleep(0.5)                   
    if not chunks:
        print(f"  {zone_code}: no data retrieved"); return
    s = pd.concat(chunks).sort_index()
    s.index = s.index.tz_localize(None)     # strip tz for uniform storage
    daily = s.resample("1D").mean().rename("load")
    daily.to_frame().to_parquet(out / f"load_{short}.parquet")
    print(f"  {short}: {len(daily)} days, "
          f"{daily.index[0].date()} → {daily.index[-1].date()}, "
          f"mean={daily.mean():.0f} MW")

for short, (zone_code, tz, start_yr) in ZONES.items():
    print(f"\nFetching {short} ({zone_code})...")
    fetch_one(zone_code, tz, start_yr)

print("\nDone.")
# %%
