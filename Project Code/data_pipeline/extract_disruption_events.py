#!/usr/bin/env python3
"""
extract_disruption_events.py
============================
Event-study extraction of real disruption signatures from OPDI parquets.

Three events are extracted:
  1. Storm Eunice     — Feb 18–19, 2022  (primary calibration event)
     Multi-city Northern European storm; hit London, Amsterdam, Hamburg, Brussels
     simultaneously. Closest real-world analogue to Scenario A (Northern Storm).

  2. July 2022 UK heatwave — Jul 18–19, 2022  (compound calibration event)
     Both Heathrow/Luton aviation AND UK rail disrupted simultaneously.
     Validates the compound rail+air disruption mechanism.

  3. Dec 2022 Heathrow snow — Dec 19–21, 2022  (out-of-sample validation)
     UK aviation snow chaos. Used ONLY for held-out validation after
     parameters are calibrated on Eunice.

Method: local event-study baseline.
  - Baseline = same calendar weekday(s) in the same month,
               excluding event dates and a 1-day buffer on each side.
  - Disruption score per city = 1 - (event_flights / baseline_flights)
    Score = 1.0: city lost all service.
    Score = 0.0: city unaffected.
    Score < 0:   city had MORE flights than baseline (re-routing overflow).

Outputs in data/:
  eunice_disruption_signature.csv
  heatwave_disruption_signature.csv
  dec2022_disruption_signature.csv
  eunice_daily_counts.csv
  heatwave_daily_counts.csv
  dec2022_daily_counts.csv

Usage:
  python data_pipeline/extract_disruption_events.py
"""

import os
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import date, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data_pipeline", "raw", "opdi")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ── ICAO → city mapping (from build_air_od.py) ────────────────────────────────
ICAO_TO_CITY = {
    "EGLL": "London",      "EGKK": "London",      "EGSS": "London",
    "EGGW": "London",      "EGLC": "London",
    "LFPG": "Paris",       "LFPO": "Paris",        "LFOB": "Paris",
    "EHAM": "Amsterdam",
    "EBBR": "Brussels",    "EBCI": "Brussels",
    "EDDF": "Frankfurt",   "EDFH": "Frankfurt",
    "EDDB": "Berlin",
    "EDDH": "Hamburg",
    "EDDM": "Munich",
    "EDDK": "Cologne",     "EDLW": "Cologne",
    "EDDS": "Stuttgart",
    "LSZH": "Zurich",
    "LSGG": "Geneva",
    "LOWW": "Vienna",
    "LIMC": "Milan",       "LIML": "Milan",        "LIME": "Milan",
    "LIPZ": "Venice",      "LIPH": "Venice",
    "LIMF": "Turin",
    "LIRF": "Rome",        "LIRA": "Rome",
    "LIRN": "Naples",
    "LEMD": "Madrid",
    "LEBL": "Barcelona",
    "LEVC": "Valencia",
    "LEZL": "Seville",
    "LPPT": "Lisbon",
    "LPPR": "Porto",
    "LFLL": "Lyon",
    "LFML": "Marseille",
    "LFBO": "Toulouse",
    "LFST": "Strasbourg",  "LFSB": "Strasbourg",
    "EKCH": "Copenhagen",
    "ESSA": "Stockholm",   "ESSB": "Stockholm",
    "ENGM": "Oslo",
    "EFHK": "Helsinki",
    "EPWA": "Warsaw",
    "LKPR": "Prague",
    "LHBP": "Budapest",
    "LROP": "Bucharest",
    "LDZA": "Zagreb",
    "LJLJ": "Ljubljana",
    "LYBE": "Belgrade",
    "LGAV": "Athens",
    "LCLK": "Nicosia",     "LCPH": "Nicosia",
    "EETN": "Tallinn",
    "EVRA": "Riga",
    "EYVI": "Vilnius",
    "LEPA": "Palma",       "LESB": "Palma",
    "GCXO": "Tenerife",    "GCLP": "Tenerife",
    "LIBD": "Bari",
    "LICJ": "Palermo",
    "LICC": "Catania",
    "EHRD": "Rotterdam",
    "EGPH": "Edinburgh",
    "EGCC": "Manchester",
    "EGNX": "Nottingham",  "EGBE": "Nottingham",
    "EGBB": "Birmingham",
}
ALL_AIRPORTS = set(ICAO_TO_CITY.keys())

# ── Event definitions ─────────────────────────────────────────────────────────
# Each event: event_dates = core disruption days
#             buffer = days around event excluded from baseline
EVENTS = {
    "eunice": {
        "name":        "Storm Eunice",
        "yyyymm":      "202202",
        "event_dates": [date(2022, 2, 18), date(2022, 2, 19)],
        "buffer_days": 1,
        "note": (
            "Calibration event. Simultaneous severe disruption at London, "
            "Amsterdam, Hamburg, Brussels. Analogue to Scenario A."
        ),
    },
    "heatwave": {
        "name":        "July 2022 UK Heatwave",
        "yyyymm":      "202207",
        "event_dates": [date(2022, 7, 18), date(2022, 7, 19)],
        "buffer_days": 1,
        "note": (
            "Compound calibration event. Heathrow and Luton aviation closure "
            "plus UK rail speed restrictions simultaneously."
        ),
    },
    "dec2022": {
        "name":        "December 2022 Heathrow Snow",
        "yyyymm":      "202212",
        "event_dates": [date(2022, 12, 19), date(2022, 12, 20), date(2022, 12, 21)],
        "buffer_days": 1,
        "note": (
            "Out-of-sample VALIDATION event. Do not use to calibrate parameters. "
            "Use only to verify that calibrated parameters generalise."
        ),
    },
    "bert": {
        "name":        "Storm Bert",
        "yyyymm":      "202411",
        "event_dates": [date(2024, 11, 22), date(2024, 11, 23)],
        "buffer_days": 1,
        "note": (
            "UK/Ireland storm. Paris +18.6% overflow — strongest overflow signal "
            "of all scanned events. Only London disrupted (cleaner spatial footprint "
            "than Eunice). Used as independent calibration event."
        ),
    },
    "amy": {
        "name":        "Storm Amy",
        "yyyymm":      "202510",
        "event_dates": [date(2025, 10, 1), date(2025, 10, 2)],
        "buffer_days": 1,
        "note": (
            "Atlantic storm. Bilbao (score=1.0) and Valencia (score=0.385) disrupted. "
            "Overflow at Rotterdam, Lisbon, Nice, Cologne, Naples, Genoa (6 cities). "
            "Independent geography from Eunice — calibrate on Amy, validate on Eunice."
        ),
    },
    "schiphol_snow": {
        "name":        "Schiphol Snow Jan 2026",
        "yyyymm":      "202601",
        "event_dates": [date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)],
        "buffer_days": 2,
        "note": (
            "Major Schiphol snow/de-icing event. 3200+ flights cancelled Jan 2-12. "
            "Peak disruption Jan 6-8. Requires fresh 202601.parquet download — "
            "original download was incomplete (missing Jan 2-10)."
        ),
    },
}

# Threshold for classifying a city as 'severely disrupted' (used for ABM input)
DISRUPTION_THRESHOLD = 0.25   # >25% of normal daily flights lost


# ── Core functions ─────────────────────────────────────────────────────────────

def load_month_daily(yyyymm: str) -> pd.DataFrame:
    """
    Load one monthly parquet and return a DataFrame of daily city-level flight
    counts (departures + arrivals combined, counting each flight once per city).

    Columns: date (datetime.date), city, n_flights
    """
    path = os.path.join(RAW_DIR, f"flight_list_{yyyymm}.parquet")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Download it first with fetch_opdi_frequency.py")
        sys.exit(1)

    schema  = pq.read_schema(path)
    names   = schema.names
    needed  = [c for c in ["adep", "ades", "dof"] if c in names]
    df      = pq.read_table(path, columns=needed).to_pandas()

    # Parse dates
    df["dof"] = pd.to_datetime(df["dof"], errors="coerce").dt.date
    df = df.dropna(subset=["dof"])

    # Keep only flights touching our airports
    mask = df["adep"].isin(ALL_AIRPORTS) | df["ades"].isin(ALL_AIRPORTS)
    df   = df[mask].copy()

    # Map to city: count each flight once for its departure city AND once for
    # its arrival city (a city is 'involved' in a flight whether it departs or arrives)
    dep_rows = df[df["adep"].isin(ALL_AIRPORTS)][["dof", "adep"]].copy()
    dep_rows["city"] = dep_rows["adep"].map(ICAO_TO_CITY)

    arr_rows = df[df["ades"].isin(ALL_AIRPORTS)][["dof", "ades"]].copy()
    arr_rows["city"] = arr_rows["ades"].map(ICAO_TO_CITY)

    combined = pd.concat([
        dep_rows[["dof", "city"]],
        arr_rows[["dof", "city"]],
    ], ignore_index=True)
    combined = combined.dropna(subset=["city"])

    # Remove intra-city movements (same city dep and arr already filtered upstream)
    daily = (combined.groupby(["dof", "city"])
                     .size()
                     .reset_index(name="n_flights"))
    daily.rename(columns={"dof": "date"}, inplace=True)
    return daily


def compute_disruption_signature(daily: pd.DataFrame, event_cfg: dict) -> pd.DataFrame:
    """
    Given a daily city counts DataFrame and an event config, return a city-level
    disruption signature.

    Logic:
      - event_flights:    mean daily flights per city on event_dates
      - baseline_flights: mean daily flights per city on same weekdays in same
                          month, EXCLUDING event dates and buffer window
      - disruption_score: 1 - (event / baseline). Positive = service reduction.
      - cancelled_pct:    disruption_score * 100

    Returns DataFrame sorted by disruption_score descending.
    """
    event_dates  = event_cfg["event_dates"]
    buffer_days  = event_cfg["buffer_days"]

    # Build exclusion window: event dates ± buffer
    excluded = set()
    for d in event_dates:
        for delta in range(-buffer_days, buffer_days + 1):
            excluded.add(d + timedelta(days=delta))

    # Weekdays (Mon=0 … Sun=6) of event dates
    event_weekdays = {d.weekday() for d in event_dates}

    # Baseline: same weekday(s) in the same month, not in exclusion window
    all_dates = daily["date"].unique()
    baseline_dates = [
        d for d in all_dates
        if d.weekday() in event_weekdays and d not in excluded
    ]

    if not baseline_dates:
        print("  WARNING: no baseline dates found — check event/month config")
        return pd.DataFrame()

    # Event flights per city (mean across event_dates present in data)
    ev_df = daily[daily["date"].isin(event_dates)]
    ev    = (ev_df.groupby("city")["n_flights"]
                  .mean()
                  .reset_index()
                  .rename(columns={"n_flights": "event_flights"}))

    # Baseline flights per city (mean across baseline_dates present in data)
    bl_df = daily[daily["date"].isin(baseline_dates)]
    bl    = (bl_df.groupby("city")["n_flights"]
                  .mean()
                  .reset_index()
                  .rename(columns={"n_flights": "baseline_flights"}))

    sig = pd.merge(bl, ev, on="city", how="left")
    sig["event_flights"]    = sig["event_flights"].fillna(0)
    sig["disruption_score"] = (
        1.0 - sig["event_flights"] / sig["baseline_flights"].replace(0, np.nan)
    ).round(4)
    sig["cancelled_pct"] = (sig["disruption_score"] * 100).round(1)
    sig["severely_disrupted"] = sig["disruption_score"] >= DISRUPTION_THRESHOLD

    sig["n_baseline_days"] = len(baseline_dates)
    sig["baseline_dates"]  = str([str(d) for d in sorted(baseline_dates)])
    sig["event_dates"]     = str([str(d) for d in sorted(event_dates)])

    sig = sig.sort_values("disruption_score", ascending=False).reset_index(drop=True)
    return sig


def print_summary(sig: pd.DataFrame, event_cfg: dict):
    """Print a concise summary of event impact."""
    name = event_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Event dates : {[str(d) for d in event_cfg['event_dates']]}")
    print(f"  Note        : {event_cfg['note']}")
    print(f"{'='*60}")

    severely = sig[sig["severely_disrupted"]]
    overflow = sig[sig["disruption_score"] < -0.10]   # >10% more flights than baseline

    print(f"\nSeverely disrupted cities (score ≥ {DISRUPTION_THRESHOLD:.0%}):")
    if severely.empty:
        print("  None above threshold.")
    else:
        for _, row in severely.iterrows():
            marker = "***" if row["disruption_score"] >= 0.50 else "  "
            print(f"  {marker} {row['city']:<15}  "
                  f"score={row['disruption_score']:.3f}  "
                  f"cancelled={row['cancelled_pct']:.1f}%  "
                  f"(baseline {row['baseline_flights']:.1f} → event {row['event_flights']:.1f} flights/day)")

    if not overflow.empty:
        print(f"\nCities with OVERFLOW (>10% above baseline) — likely absorbed rerouted pax:")
        for _, row in overflow.iterrows():
            print(f"    {row['city']:<15}  score={row['disruption_score']:.3f}  "
                  f"+{-row['cancelled_pct']:.1f}% above baseline")


def make_abm_disruption_input(sig: pd.DataFrame, event_name: str) -> dict:
    """
    Translate disruption signature into ABM-ready disrupted_nodes list.

    Severe disruption (score ≥ threshold) → disrupted_node (all incident edges blocked).
    This mirrors how scenarios.py defines Scenario A.

    Returns a dict with keys:
      disrupted_nodes:  list of city names
      overflow_cities:  cities showing >10% above-baseline flights (likely rerouting hubs)
    """
    disrupted = sig[sig["disruption_score"] >= DISRUPTION_THRESHOLD]["city"].tolist()
    overflow  = sig[sig["disruption_score"] < -0.10]["city"].tolist()
    return {
        "event":           event_name,
        "disrupted_nodes": disrupted,
        "overflow_cities": overflow,
        "threshold_used":  DISRUPTION_THRESHOLD,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for event_key, event_cfg in EVENTS.items():
        yyyymm = event_cfg["yyyymm"]
        print(f"\n[Loading {yyyymm} for {event_cfg['name']}]")

        daily = load_month_daily(yyyymm)
        print(f"  Loaded {len(daily)} city-day rows across "
              f"{daily['date'].nunique()} days, {daily['city'].nunique()} cities")

        # Save raw daily counts
        daily_out = os.path.join(DATA_DIR, f"{event_key}_daily_counts.csv")
        daily.to_csv(daily_out, index=False)
        print(f"  Daily counts → {daily_out}")

        # Compute and save disruption signature
        sig = compute_disruption_signature(daily, event_cfg)
        if sig.empty:
            continue

        sig_out = os.path.join(DATA_DIR, f"{event_key}_disruption_signature.csv")
        sig.to_csv(sig_out, index=False)
        print(f"  Signature    → {sig_out}")

        print_summary(sig, event_cfg)

        abm_input = make_abm_disruption_input(sig, event_key)
        print(f"\n  ABM disrupted_nodes (score ≥ {DISRUPTION_THRESHOLD:.0%}):")
        print(f"    {abm_input['disrupted_nodes']}")
        print(f"  Likely rerouting overflow hubs:")
        print(f"    {abm_input['overflow_cities']}")

    print("\n\nDone. Next steps:")
    print("  1. Review eunice_disruption_signature.csv")
    print("  2. Compare ABM disrupted_nodes list to known Eunice impact geography")
    print("  3. Run ABM with Eunice disrupted_nodes to get predicted stranding")
    print("  4. Face-validity check: London + Amsterdam should top the stranding list")
    print("  5. Grid-search reroute_every and p_reschedule to best match recovery timeline")
    print("  6. Use dec2022 signature to validate calibrated parameters hold-out")


if __name__ == "__main__":
    main()
