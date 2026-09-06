#!/usr/bin/env python3
"""
build_air_od.py
==================
Convert OPDI flight records (already on disk) into estimated annual air
passenger OD for all 53 cities, covering 2022 + 2023.

Method:
    estimated_pax = n_flights × avg_seats_by_typecode × LOAD_FACTOR

Aircraft seat counts by ICAO typecode follow manufacturer standard
single-class configurations; load factor 0.80 matches IATA's reported
European average for 2022-2023.

Outputs:
    data/air_od.csv   — undirected city-pair, columns:
                           city_a, city_b, n_flights, avg_seats,
                           air_pax_estimated, distance_km
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from math import radians, sin, cos, sqrt, atan2

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(PROJECT_ROOT, "data_pipeline", "raw", "opdi")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOAD_FACTOR   = 0.80   # IATA European average 2022-2023
TARGET_YEARS  = {2022, 2023}

# ── Seat count by ICAO typecode ───────────────────────────────────────────────
# Standard single-class or typical high-density config for European routes.
# Source: manufacturer specs / ICAO aircraft type data.
SEATS = {
    # Narrowbody workhorses
    "B738": 189, "B737": 149, "B736": 119, "B734": 149, "B735": 122,
    "B38M": 178, "B39M": 196,                      # MAX family
    "A319": 140, "A320": 180, "A321": 220,
    "A19N": 140, "A20N": 180, "A21N": 220,         # neo family
    # Embraer
    "E170": 76,  "E175": 76,  "E190": 98,  "E195": 118,
    "E75L": 76,  "E75S": 76,  "E7W5": 76,
    "E290": 98,  "E295": 136,
    # Bombardier / Airbus A220
    "BCS1": 110, "BCS3": 130,                      # A220-100 / A220-300
    "CRJ2": 50,  "CRJ7": 70,  "CRJ9": 90,  "CRJX": 104,
    # ATR turboprops
    "AT43": 48,  "AT45": 48,  "AT72": 70,
    "AT75": 70,  "AT76": 70,
    # Dash-8
    "DH8A": 37,  "DH8B": 39,  "DH8C": 50,  "DH8D": 78,
    # Widebody (long-haul, rare on intra-Europe but present)
    "A332": 293, "A333": 335, "A338": 257, "A339": 287,
    "A359": 315, "A35K": 369,
    "B763": 218, "B764": 261,
    "B772": 396, "B77L": 396, "B77W": 396,
    "B788": 242, "B789": 296, "B78X": 330,
    "A380": 555,
    # Sukhoi Superjet
    "SU95": 98,
    # Misc
    "B752": 200, "B753": 243, "B762": 174,
    "F100": 107, "F70":  80,
    "SF34": 34,  "JS41": 29,
}
DEFAULT_SEATS = 150   # fallback for unknown typecodes

# ── ICAO airport → city mapping (same as fetch_opdi_frequency.py + extras) ───
ICAO_TO_CITY = {
    "EGLL": "London",    "EGKK": "London",    "EGSS": "London",
    "EGGW": "London",    "EGLC": "London",
    "LFPG": "Paris",     "LFPO": "Paris",     "LFOB": "Paris",
    "EHAM": "Amsterdam",
    "EBBR": "Brussels",  "EBCI": "Brussels",
    "EDDF": "Frankfurt", "EDFH": "Frankfurt",
    "EDDB": "Berlin",
    "EDDH": "Hamburg",
    "EDDM": "Munich",
    "EDDK": "Cologne",   "EDLW": "Cologne",
    "EDDS": "Stuttgart",
    "LSZH": "Zurich",
    "LSGG": "Geneva",
    "LOWW": "Vienna",
    "LIMC": "Milan",     "LIML": "Milan",     "LIME": "Milan",
    "LIPZ": "Venice",    "LIPH": "Venice",
    "LIMF": "Turin",
    "LIRF": "Rome",      "LIRA": "Rome",
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
    "LFST": "Strasbourg", "LFSB": "Strasbourg",
    "EKCH": "Copenhagen",
    "ESSA": "Stockholm",  "ESSB": "Stockholm",
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
    "LCLK": "Nicosia",   "LCPH": "Nicosia",
    "EETN": "Tallinn",
    "EVRA": "Riga",
    "EYVI": "Vilnius",
    "UMMS": "Minsk",
    "LEPA": "Palma",     "LESB": "Palma",
    "GCXO": "Tenerife",  "GCLP": "Tenerife",
    "LIBD": "Bari",
    "LICJ": "Palermo",
    "LICC": "Catania",
    "EHRD": "Rotterdam",
    "EGPH": "Edinburgh",
    "EGCC": "Manchester",
    "EGNX": "Nottingham", "EGBE": "Nottingham",
    "EGBB": "Birmingham",
}
ALL_AIRPORTS = set(ICAO_TO_CITY.keys())


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def process_month(path):
    """Return DataFrame with columns: from_city, to_city, n_flights, seat_sum."""
    schema = pq.read_schema(path)
    names  = schema.names
    cols   = [c for c in ["adep", "ades", "dof", "typecode"] if c in names]
    df     = pq.read_table(path, columns=cols).to_pandas()

    # Filter to target years
    if "dof" in df.columns:
        df["dof"] = pd.to_datetime(df["dof"], errors="coerce")
        df = df[df["dof"].dt.year.isin(TARGET_YEARS)]

    mask = df["adep"].isin(ALL_AIRPORTS) & df["ades"].isin(ALL_AIRPORTS)
    df   = df[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["from_city","to_city","n_flights","seat_sum"])

    df["from_city"] = df["adep"].map(ICAO_TO_CITY)
    df["to_city"]   = df["ades"].map(ICAO_TO_CITY)
    df = df[df["from_city"] != df["to_city"]]

    df["seats"] = df["typecode"].map(SEATS).fillna(DEFAULT_SEATS).astype(int)
    result = (df.groupby(["from_city","to_city"])
                .agg(n_flights=("seats","count"), seat_sum=("seats","sum"))
                .reset_index())
    return result


def main():
    print("=== build_air_od.py ===")
    print(f"Load factor: {LOAD_FACTOR}  |  Target years: {sorted(TARGET_YEARS)}")

    files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".parquet")
    ])
    print(f"Parquet files found: {len(files)}\n")

    all_chunks = []
    for path in files:
        fname = os.path.basename(path)
        chunk = process_month(path)
        if not chunk.empty:
            print(f"  {fname}: {chunk['n_flights'].sum():,} flights on {len(chunk)} routes")
            all_chunks.append(chunk)

    if not all_chunks:
        print("ERROR: no matching routes found.")
        return

    combined = (pd.concat(all_chunks, ignore_index=True)
                  .groupby(["from_city","to_city"])
                  .agg(n_flights=("n_flights","sum"), seat_sum=("seat_sum","sum"))
                  .reset_index())
    combined["avg_seats"] = (combined["seat_sum"] / combined["n_flights"]).round(1)

    # Make undirected: fold both directions, sum flights, average seats
    combined["city_a"] = combined[["from_city","to_city"]].min(axis=1)
    combined["city_b"] = combined[["from_city","to_city"]].max(axis=1)
    undirected = (combined.groupby(["city_a","city_b"])
                          .agg(n_flights=("n_flights","sum"),
                               seat_sum=("seat_sum","sum"))
                          .reset_index())
    undirected["avg_seats"] = (undirected["seat_sum"] / undirected["n_flights"]).round(1)

    # Divide by 2 years to get annual average
    undirected["n_flights_annual"]  = (undirected["n_flights"] / 2).round(0)
    undirected["air_pax_estimated"] = (
        undirected["n_flights_annual"] * undirected["avg_seats"] * LOAD_FACTOR
    ).round(0).astype(int)

    # Add city coordinates for distance
    cities_df = pd.read_csv(os.path.join(DATA_DIR, "cities.csv"))
    coord = cities_df.set_index("city")[["lat","lon"]].to_dict("index")

    def dist(row):
        a, b = coord.get(row["city_a"]), coord.get(row["city_b"])
        if a and b:
            return round(haversine(a["lat"], a["lon"], b["lat"], b["lon"]), 1)
        return None

    undirected["distance_km"] = undirected.apply(dist, axis=1)

    out = undirected[["city_a","city_b","n_flights_annual","avg_seats",
                       "air_pax_estimated","distance_km"]].copy()
    out_path = os.path.join(DATA_DIR, "air_od.csv")
    out.to_csv(out_path, index=False)

    print(f"\nSaved {len(out)} city-pairs → {out_path}")
    print(f"Total estimated air passengers (annual avg): {out['air_pax_estimated'].sum():,.0f}")
    print("\nTop 15 routes by estimated passengers:")
    print(out.nlargest(15,"air_pax_estimated")
             [["city_a","city_b","n_flights_annual","avg_seats","air_pax_estimated"]]
             .to_string(index=False))


if __name__ == "__main__":
    main()
