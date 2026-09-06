"""
scenarios.py
============
Three weather disruption scenarios for the European transport ABM.

Each scenario is defined geographically in SCENARIO_DEFS using storm zones
(center lat/lon + radii). Call build_scenarios(G) once after build_network()
to get the SCENARIOS dict with disrupted_nodes and disrupted_edges computed
automatically from the cities in the graph — so adding new cities never
requires touching this file.

Zone parameters
---------------
  center          : (lat, lon) of the weather event eye / focal point
  node_radius_km  : cities within this distance are fully closed
                    (all their edges blocked in the simulation)
  edge_radius_km  : cities within this distance have specific mode(s) disrupted
  edge_modes      : list of modes to disrupt in the edge zone, e.g. ["rail"]
                    or ["air", "rail"]

Scenario A — Northern Storm
    North Sea storm fully closes the main coastal hubs (London, Amsterdam,
    Hamburg, and any other city within 500 km of the North Sea centre).
    Tests total hub closure: stranded agents cannot leave or arrive.

Scenario B — Alpine Freeze
    Heavy snow locks Alpine passes and shuts Zurich Airport (ZRH).
    Zurich (within 110 km of St. Gotthard) is fully closed; all rail through
    the Alpine corridor (within 320 km) is suspended. Air is unaffected —
    Alpine airports divert to lowland hubs.

Scenario C — Atlantic Cascade
    Atlantic storm sweeps eastward in two phases:
      Phase 1 — Portuguese coast: Lisbon and Porto fully closed (within 250 km);
                surrounding Iberian and French Atlantic cities lose air + rail
                (within 700 km — catches Madrid, Seville, Bilbao, Bordeaux, …).
      Phase 2 — Paris air bridges: storm front reaches Île-de-France,
                severing all air links through Paris (within 180 km).
"""

import math


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


SCENARIO_DEFS = {
    "A": {
        "name": "Northern Storm",
        "description": (
            "Atlantic storm centred on North Sea: all hubs within 500 km "
            "fully closed (air + rail)"
        ),
        "zones": [
            {
                "center":         (55.0, 3.0),  # North Sea
                "node_radius_km": 500,           # London, Amsterdam, Hamburg, Rotterdam, …
                "edge_radius_km": 0,
                "edge_modes":     [],
            },
        ],
    },

    "B": {
        "name": "Alpine Freeze",
        "description": (
            "Heavy snow closes Alpine rail corridors and shuts Zurich Airport; "
            "all rail within 320 km of St. Gotthard suspended"
        ),
        "zones": [
            {
                "center":         (46.8, 9.5),   # Central Alps (St. Gotthard)
                "node_radius_km": 110,            # Zurich fully closed
                "edge_radius_km": 320,            # Geneva, Milan, Turin, Munich, Venice, …
                "edge_modes":     ["rail"],
            },
        ],
    },

    "C": {
        "name": "Atlantic Cascade",
        "description": (
            "Atlantic storm sweeps east: Portuguese coast fully closed, "
            "western Iberia and French Atlantic coast lose air + rail, "
            "Paris air bridges severed"
        ),
        "zones": [
            {
                "center":         (39.5, -8.5),  # Portuguese Atlantic coast
                "node_radius_km": 250,            # Lisbon, Porto
                "edge_radius_km": 700,            # Madrid, Seville, Bilbao, Bordeaux, …
                "edge_modes":     ["air", "rail"],
            },
            {
                "center":         (48.5, 2.5),   # Paris / Île-de-France
                "node_radius_km": 0,
                "edge_radius_km": 180,            # Paris air bridges
                "edge_modes":     ["air"],
            },
        ],
    },
}


def build_scenarios(G):
    """
    Compute disrupted_nodes and disrupted_edges for each scenario from G.

    City coordinates are read from G node attributes (lat, lon), so any city
    added to the network is automatically included if it falls within a zone.
    Only edges that exist in G are listed in disrupted_edges.

    Returns a dict keyed "A"/"B"/"C" with the same structure as the old
    SCENARIOS constant, suitable as a drop-in replacement.
    """
    city_coords = {n: (G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in G.nodes}
    existing    = {(u, v, d["mode"]) for u, v, d in G.edges(data=True)}

    scenarios = {}
    for key, defn in SCENARIO_DEFS.items():
        closed_nodes = set()
        edge_cities  = {}   # city -> set of modes to disrupt

        for zone in defn["zones"]:
            clat, clon = zone["center"]
            node_r = zone.get("node_radius_km", 0)
            edge_r = zone.get("edge_radius_km", 0)
            modes  = zone.get("edge_modes", [])

            for city, (lat, lon) in city_coords.items():
                d = _haversine_km(clat, clon, lat, lon)
                if node_r > 0 and d <= node_r:
                    closed_nodes.add(city)
                elif edge_r > 0 and d <= edge_r and modes:
                    edge_cities.setdefault(city, set()).update(modes)

        # Collect all existing edges involving an edge-zone city
        disrupted_edges = set()
        for city, modes in edge_cities.items():
            if city in closed_nodes:
                continue  # fully closed already via node
            for u, v, m in existing:
                if m in modes and (u == city or v == city):
                    disrupted_edges.add((u, v, m))

        scenarios[key] = {
            "name":            defn["name"],
            "description":     defn["description"],
            "disrupted_nodes": sorted(closed_nodes),
            "disrupted_edges": sorted(disrupted_edges),
        }

    return scenarios


def describe_scenarios(G):
    """Print a summary of which cities each scenario affects — useful for tuning radii."""
    scenarios = build_scenarios(G)
    for key, sc in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario {key}: {sc['name']}")
        print(f"  Fully closed ({len(sc['disrupted_nodes'])} cities): "
              f"{', '.join(sc['disrupted_nodes'])}")
        modes = {}
        for _, _, m in sc["disrupted_edges"]:
            modes[m] = modes.get(m, 0) + 1
        edge_summary = ", ".join(f"{v} {k}" for k, v in modes.items())
        print(f"  Disrupted edges: {len(sc['disrupted_edges'])} ({edge_summary})")
