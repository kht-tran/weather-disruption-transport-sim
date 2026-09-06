#!/usr/bin/env python3
"""
build_combined_od.py
=======================
Merge air OD (from build_air_od.py) with rail OD (from rail_edges.csv)
into a single combined OD matrix with per-route modal split.

Rail passengers:
    rail_pax = frequency_per_day × 365 × capacity_seats × load_factor_implied

Air passengers:
    air_pax = from air_od.csv (already estimated)

Both are in comparable annual passenger units before merging.

Outputs:
    data/od_matrix_combined.csv  — one row per city-pair, with:
        city_a, city_b, air_pax, rail_pax, total_pax,
        air_share, rail_share, distance_km, has_rail, has_air
    data/modal_split.csv         — same but cleaner, for ABM use
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MIN_FLIGHTS_ANNUAL = 50   # drop city-pairs with fewer flights (noise floor)


def load_air(path):
    df = pd.read_csv(path)
    df = df[df["n_flights_annual"] >= MIN_FLIGHTS_ANNUAL].copy()
    df = df[["city_a","city_b","air_pax_estimated","distance_km"]].copy()
    df.rename(columns={"air_pax_estimated": "air_pax"}, inplace=True)
    return df


def load_rail(path):
    df = pd.read_csv(path)

    # Compute annual passenger estimate
    # rail_pax = frequency_per_day × 365 × capacity_seats × load_factor_implied
    df["rail_pax"] = (
        df["frequency_per_day"] * 365 *
        df["capacity_seats"] * df["load_factor_implied"]
    ).round(0)

    # Make undirected (take the larger of the two directions if both exist)
    df["city_a"] = df[["from_city","to_city"]].min(axis=1)
    df["city_b"] = df[["from_city","to_city"]].max(axis=1)

    rail_und = (df.groupby(["city_a","city_b"])
                  .agg(rail_pax=("rail_pax","max"),  # max across directions
                       rail_time_min=("travel_time_min","min"),
                       frequency_per_day=("frequency_per_day","max"))
                  .reset_index())
    return rail_und


def main():
    print("=== build_combined_od.py ===")

    air_path  = os.path.join(DATA_DIR, "air_od.csv")
    rail_path = os.path.join(DATA_DIR, "rail_edges.csv")

    if not os.path.exists(air_path):
        print("ERROR: air_od.csv not found. Run build_air_od.py first.")
        return

    air  = load_air(air_path)
    rail = load_rail(rail_path)

    print(f"Air city-pairs:  {len(air)}")
    print(f"Rail city-pairs: {len(rail)}")

    # Outer join — keep pairs that have at least one mode
    merged = pd.merge(air, rail, on=["city_a","city_b"], how="outer")

    # Fill missing distance from whichever side has it
    # (rail_edges_v4 also has distance_km)
    rail_dist = pd.read_csv(rail_path)[["from_city","to_city","distance_km"]].copy()
    rail_dist["city_a"] = rail_dist[["from_city","to_city"]].min(axis=1)
    rail_dist["city_b"] = rail_dist[["from_city","to_city"]].max(axis=1)
    rail_dist = (rail_dist.groupby(["city_a","city_b"])["distance_km"]
                          .min().reset_index()
                          .rename(columns={"distance_km":"distance_rail"}))
    merged = pd.merge(merged, rail_dist, on=["city_a","city_b"], how="left")
    merged["distance_km"] = merged["distance_km"].fillna(merged["distance_rail"])
    merged.drop(columns=["distance_rail"], inplace=True)

    merged["air_pax"]  = merged["air_pax"].fillna(0).astype(int)
    merged["rail_pax"] = merged["rail_pax"].fillna(0).astype(int)
    merged["has_air"]  = merged["air_pax"] > 0
    merged["has_rail"] = merged["rail_pax"] > 0

    # Combined OD and modal split
    merged["total_pax"] = merged["air_pax"] + merged["rail_pax"]
    merged = merged[merged["total_pax"] > 0].copy()

    merged["air_share"]  = (merged["air_pax"]  / merged["total_pax"]).round(4)
    merged["rail_share"] = (merged["rail_pax"] / merged["total_pax"]).round(4)

    # Summary statistics
    both  = merged[merged["has_air"] & merged["has_rail"]]
    print(f"\nCity-pairs with both modes: {len(both)}")
    print(f"City-pairs air only:        {len(merged[merged['has_air'] & ~merged['has_rail']])}")
    print(f"City-pairs rail only:       {len(merged[~merged['has_air'] & merged['has_rail']])}")
    print(f"Total city-pairs:           {len(merged)}")

    print(f"\nTotal air pax (annual):  {merged['air_pax'].sum():>15,.0f}")
    print(f"Total rail pax (annual): {merged['rail_pax'].sum():>15,.0f}")
    overall_air_share = merged["air_pax"].sum() / merged["total_pax"].sum()
    print(f"Overall air share:       {overall_air_share:.1%}")
    print(f"Overall rail share:      {1-overall_air_share:.1%}")

    print("\nSample — high rail-share routes:")
    sample = (both.nsmallest(10, "air_share")
                  [["city_a","city_b","air_pax","rail_pax","air_share","distance_km"]]
                  .to_string(index=False))
    print(sample)

    print("\nSample — high air-share routes:")
    sample2 = (both.nlargest(10, "air_share")
                   [["city_a","city_b","air_pax","rail_pax","air_share","distance_km"]]
                   .to_string(index=False))
    print(sample2)

    # Save full combined matrix
    cols_full = ["city_a","city_b","air_pax","rail_pax","total_pax",
                 "air_share","rail_share","distance_km","has_air","has_rail",
                 "rail_time_min","frequency_per_day"]
    cols_full = [c for c in cols_full if c in merged.columns]
    merged[cols_full].sort_values(["city_a","city_b"]).to_csv(
        os.path.join(DATA_DIR, "od_matrix_combined.csv"), index=False
    )

    # Save clean modal split for ABM use
    modal = merged[["city_a","city_b","air_share","rail_share",
                    "has_air","has_rail"]].copy()
    modal.to_csv(os.path.join(DATA_DIR, "modal_split.csv"), index=False)

    print(f"\nSaved od_matrix_combined.csv  ({len(merged)} rows)")
    print(f"Saved modal_split.csv         ({len(modal)} rows)")


if __name__ == "__main__":
    main()
