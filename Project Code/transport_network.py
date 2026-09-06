"""
transport_network.py
====================
Builds a NetworkX MultiDiGraph of the European transport network.

Nodes  — cities with lat, lon, country, population_k attributes
Edges  — rail or air connections with travel_time_min, operator, mode,
          and night_train attributes

Night trains
------------
Rail edges with night_train=True in rail_edges.csv are classified as night
trains. This flag is set by process_gtfs.py based on actual GTFS departure
times (departure >= 19:00 AND service crosses midnight or duration >= 6h).
The flag is trip-level and can propagate to short sub-segments (e.g. a Nightjet
Hamburg→Munich passing through Frankfurt tags Frankfurt→Cologne as night-only).
build_network() clears the flag for any edge shorter than MIN_NIGHT_SEGMENT=300 min
to remove these false positives. Known Nightjet routes retained: Vienna–Amsterdam,
Vienna–Hamburg, Vienna–Zurich, Zurich–Hamburg, Munich–Amsterdam, Hamburg–Zurich.

No capacity constraints
-----------------------
Edge capacity is deliberately omitted. Stranding in this model is driven purely
by network topology under disruption — an agent strands only when no path exists
from their current node to their destination through non-disrupted edges. Capacity
constraints are excluded for three reasons:
  1. Scale: with a stylised agent count (N≈500), no principled per-edge integer
     limit can be grounded in real throughput data.
  2. Data: frequency data is available but seat counts and real load factors are
     not — any capacity value would be an arbitrary proxy.
  3. Realism: during weather disruptions operators dynamically adjust capacity
     (extra trains, charter flights); static capacity would artificially amplify
     stranding beyond what topology alone produces.

Usage
-----
    from transport_network import build_network, print_stats
    G = build_network()
"""

import os
import networkx as nx
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_network(data_dir: str = DATA_DIR) -> nx.MultiDiGraph:
    """Load CSVs and return a MultiDiGraph with rail + air edges."""
    cities = pd.read_csv(os.path.join(data_dir, "cities.csv")).set_index("city")
    rail   = pd.read_csv(os.path.join(data_dir, "rail_edges.csv"))
    air    = pd.read_csv(os.path.join(data_dir, "air_edges.csv"))

    # GTFS night-train flag propagates from the full trip to every sub-segment,
    # so short daytime legs (e.g. Cologne→Frankfurt 37 min) can be falsely marked
    # night_train=True. Clear the flag for any edge shorter than 5 hours.
    MIN_NIGHT_SEGMENT = 300
    rail.loc[rail["travel_time_min"] < MIN_NIGHT_SEGMENT, "night_train"] = False

    G = nx.MultiDiGraph()

    for city, row in cities.iterrows():
        pop = row.get("population_k")
        G.add_node(city,
                   country=row["country"],
                   lat=float(row["lat"]),
                   lon=float(row["lon"]),
                   population_k=float(pop) if pd.notna(pop) else 100.0)

    for _, row in rail.iterrows():
        night = bool(row.get("night_train", False))
        G.add_edge(
            row["from_city"], row["to_city"],
            mode="rail",
            operator=row["operator"],
            travel_time_min=int(row["travel_time_min"]),
            night_train=night,
        )

    for _, row in air.iterrows():
        G.add_edge(
            row["from_city"], row["to_city"],
            mode="air",
            operator="air",
            travel_time_min=int(row["travel_time_min"]),
            night_train=False,
        )

    return G


def fastest_edge(G: nx.MultiDiGraph, u: str, v: str, mode: str = None):
    """Return the fastest edge dict between u and v, optionally filtered by mode."""
    edges = G.get_edge_data(u, v)
    if not edges:
        return None
    candidates = [e for e in edges.values() if mode is None or e["mode"] == mode]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e["travel_time_min"])


def print_stats(G: nx.MultiDiGraph):
    """Print a summary of the network."""
    rail_all   = [(u, v, d) for u, v, d in G.edges(data=True) if d["mode"] == "rail"]
    night_rail = [(u, v, d) for u, v, d in rail_all if d["night_train"]]
    air_edges  = [(u, v, d) for u, v, d in G.edges(data=True) if d["mode"] == "air"]

    print(f"Nodes (cities)    : {G.number_of_nodes()}")
    print(f"Rail edges        : {len(rail_all)}"
          f"  ({len(night_rail)} night trains, {len(rail_all) - len(night_rail)} day)")
    print(f"Air edges         : {len(air_edges)}")
    print(f"Total edges       : {G.number_of_edges()}")

    G_simple = nx.Graph(G)
    components = list(nx.connected_components(G_simple))
    print(f"Connected components: {len(components)}")
    if len(components) == 1:
        print("  Network is fully connected.")
    else:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {sorted(comp)}")

    degrees = sorted(G.degree(), key=lambda x: -x[1])
    print(f"\nTop 5 most connected cities:")
    for city, deg in degrees[:5]:
        print(f"  {city:<14} degree={deg}")

    print(f"\nNight train routes:")
    shown = set()
    for u, v, d in night_rail:
        pair = tuple(sorted([u, v]))
        if pair not in shown:
            shown.add(pair)
            print(f"  {u} <-> {v}: {d['travel_time_min']} min")

    print("\nSample fastest rail connections:")
    sample = [("Paris", "Brussels"), ("Frankfurt", "Cologne"),
              ("Munich", "Vienna"), ("Barcelona", "Madrid"), ("Milan", "Rome")]
    for a, b in sample:
        e = fastest_edge(G, a, b, mode="rail")
        if e:
            tag = " [night]" if e["night_train"] else ""
            print(f"  {a} -> {b}: {e['travel_time_min']} min{tag}")
        else:
            print(f"  {a} → {b}: no rail edge")


if __name__ == "__main__":
    G = build_network()
    print_stats(G)
