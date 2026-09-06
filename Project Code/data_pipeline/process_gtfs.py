#!/usr/bin/env python3
"""
HPC Data Pipeline: European Multimodal Transport Network (enriched)
====================================================================
Processes GTFS rail feeds + NeTEx feeds + OpenFlights air data.
Outputs enriched CSVs with frequency_per_day and night_train columns.

Outputs (saved to ../data/):
  rail_edges.csv  — from_city, to_city, operator, travel_time_min,
                    frequency_per_day, night_train
  air_edges.csv   — from_city, to_city, operator, travel_time_min,
                    frequency_per_day

Night train detection
---------------------
A rail service is flagged night_train=True if the first city-stop departure
is >= 19:00 AND either:
  (a) the last city-stop arrival is >= 24:00 in GTFS time convention
      (i.e. arrival_time > "24:00:00" indicating next-day), or
  (b) total journey duration across city stops >= 360 min (6 h).
Criterion (b) is a fallback for feeds that don't use the >24h convention.

Frequency
---------
Rail: for each GTFS feed, a trip's days-per-week is read from calendar.txt
(summing the 7 weekday flags for its service_id). If calendar.txt is absent,
calendar_dates.txt is used to estimate days/week from distinct active dates.
frequency_per_day = sum(days_per_week for all trips on route) / 7.

Air: OpenFlights routes.dat has no frequency data. Proxy used:
frequency_per_day = number of distinct airlines serving the route * 3.5,
capped at 20, minimum 2. Documented as a known limitation.

Eurostar (London connections)
-----------------------------
The UK ATOC CIF feed does not include Eurostar international services.
Eurostar is processed from its own GTFS static commercial feed (v2), listed
in GTFS_FEEDS as "Eurostar" → gtfs_static_commercial_v2.zip. It goes through
the same process_gtfs() pipeline as all other feeds.
CITATION: Eurostar (2026). GTFS static commercial feed v2.
          Feed covers 2026-04-29 to 2026-07-27.

Porto–Madrid manual edge
------------------------
No GTFS/NeTEx feed covers this cross-border through-route because the transfer
city (Vigo) is not a target city. Added as a hardcoded constant in
MANUAL_RAIL_EDGES: Porto→Vigo (CP Celta, ~150 min) + Vigo→Madrid (Renfe Alvia,
~155 min) + connection buffer = 360 min total, frequency 1 round trip/day.
CITATION: CP (2024). Celta Porto–Vigo. https://www.cp.pt
          Renfe (2024). Alvia Galicia–Madrid. https://www.renfe.com
[Accessed: 30/04/2026]

Setup on HPC:
  pip install pandas numpy

Run (from data_pipeline/ folder):
  python process_gtfs.py
"""

import gzip
import os
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR    = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# ── Target cities (population >= 400k in BE/DE/FR/IT/NL/AT/ES/PT/CH/GB) ──────
CITIES = {
    # Original 28
    "London":      {"country": "GB", "lat": 51.50853, "lon":  -0.12574},
    "Paris":       {"country": "FR", "lat": 48.85341, "lon":   2.34880},
    "Amsterdam":   {"country": "NL", "lat": 52.37403, "lon":   4.88969},
    "Brussels":    {"country": "BE", "lat": 50.85045, "lon":   4.34878},
    "Frankfurt":   {"country": "DE", "lat": 50.11552, "lon":   8.68417},
    "Berlin":      {"country": "DE", "lat": 52.52437, "lon":  13.41053},
    "Hamburg":     {"country": "DE", "lat": 53.55073, "lon":   9.99302},
    "Munich":      {"country": "DE", "lat": 48.13743, "lon":  11.57549},
    "Cologne":     {"country": "DE", "lat": 50.93333, "lon":   6.95000},
    "Stuttgart":   {"country": "DE", "lat": 48.78232, "lon":   9.17702},
    "Zurich":      {"country": "CH", "lat": 47.36667, "lon":   8.55000},
    "Geneva":      {"country": "CH", "lat": 46.20222, "lon":   6.14569},
    "Vienna":      {"country": "AT", "lat": 48.20849, "lon":  16.37208},
    "Milan":       {"country": "IT", "lat": 45.46427, "lon":   9.18951},
    "Venice":      {"country": "IT", "lat": 45.43713, "lon":  12.33265},
    "Turin":       {"country": "IT", "lat": 45.07049, "lon":   7.68682},
    "Rome":        {"country": "IT", "lat": 41.89193, "lon":  12.51133},
    "Naples":      {"country": "IT", "lat": 40.85216, "lon":  14.26811},
    "Madrid":      {"country": "ES", "lat": 40.41650, "lon":  -3.70256},
    "Barcelona":   {"country": "ES", "lat": 41.38879, "lon":   2.15899},
    "Valencia":    {"country": "ES", "lat": 39.47391, "lon":  -0.37966},
    "Seville":     {"country": "ES", "lat": 37.38283, "lon":  -5.97317},
    "Lisbon":      {"country": "PT", "lat": 38.72509, "lon":  -9.14980},
    "Porto":       {"country": "PT", "lat": 41.14850, "lon":  -8.61097},
    "Lyon":        {"country": "FR", "lat": 45.74906, "lon":   4.84789},
    "Marseille":   {"country": "FR", "lat": 43.29695, "lon":   5.38107},
    "Toulouse":    {"country": "FR", "lat": 43.60426, "lon":   1.44367},
    "Strasbourg":  {"country": "FR", "lat": 48.58392, "lon":   7.74553},
    # New cities
    "Rotterdam":   {"country": "NL", "lat": 51.92250, "lon":   4.47917},
    "Lille":       {"country": "FR", "lat": 50.63297, "lon":   3.05858},
    "The Hague":   {"country": "NL", "lat": 52.07667, "lon":   4.29861},
    "Bilbao":      {"country": "ES", "lat": 43.26271, "lon":  -2.92528},
    "Bordeaux":    {"country": "FR", "lat": 44.84044, "lon":  -0.58050},
    "Dusseldorf":  {"country": "DE", "lat": 51.22172, "lon":   6.77616},
    "Palermo":     {"country": "IT", "lat": 38.13205, "lon":  13.33561},
    "Leipzig":     {"country": "DE", "lat": 51.33962, "lon":  12.37129},
    "Dortmund":    {"country": "DE", "lat": 51.51494, "lon":   7.46603},
    "Malaga":      {"country": "ES", "lat": 36.72016, "lon":  -4.42034},
    "Essen":       {"country": "DE", "lat": 51.45657, "lon":   7.01228},
    "Bremen":      {"country": "DE", "lat": 53.07516, "lon":   8.80777},
    "Dresden":     {"country": "DE", "lat": 51.05089, "lon":  13.73832},
    "Genoa":       {"country": "IT", "lat": 44.40478, "lon":   8.94439},
    "Hannover":    {"country": "DE", "lat": 52.37052, "lon":   9.73322},
    "Antwerp":     {"country": "BE", "lat": 51.21940, "lon":   4.40250},
    "Nuremberg":   {"country": "DE", "lat": 49.45421, "lon":  11.07752},
    "Duisburg":    {"country": "DE", "lat": 51.43247, "lon":   6.76516},
    "Nantes":      {"country": "FR", "lat": 47.21725, "lon":  -1.55336},
    "Nice":        {"country": "FR", "lat": 43.70313, "lon":   7.26608},
    "Murcia":      {"country": "ES", "lat": 37.98704, "lon":  -1.13004},
    "Utrecht":     {"country": "NL", "lat": 52.09083, "lon":   5.12222},
    "Palma":       {"country": "ES", "lat": 39.56939, "lon":   2.65024},
    "Granada":     {"country": "ES", "lat": 37.18817, "lon":  -3.60667},
    "Liege":       {"country": "BE", "lat": 50.63232, "lon":   5.56749},
}

CITY_MATCH_KM = 30

RAIL_ROUTE_TYPES = {2, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109}

# Minimum plausible travel time for an intercity rail segment (minutes).
# High-speed trains passing through a station without stopping (GTFS pass-through
# stops) produce phantom segment times of 1–9 minutes. All legitimate city-pair
# connections in the 53-city network are at least ~20 min apart geographically.
# 15 minutes is a safe floor: rejects all pass-through phantoms while keeping
# every real intercity segment.
MIN_SEGMENT_MIN = 15

IATA_TO_CITY = {
    # Original 28
    "LHR": "London",    "LGW": "London",    "STN": "London",
    "LTN": "London",    "LCY": "London",
    "CDG": "Paris",     "ORY": "Paris",     "BVA": "Paris",
    "AMS": "Amsterdam",
    "BRU": "Brussels",  "CRL": "Brussels",
    "FRA": "Frankfurt", "HHN": "Frankfurt",
    "BER": "Berlin",
    "HAM": "Hamburg",
    "MUC": "Munich",
    "CGN": "Cologne",
    "STR": "Stuttgart",
    "ZRH": "Zurich",
    "GVA": "Geneva",
    "VIE": "Vienna",
    "MXP": "Milan",     "LIN": "Milan",     "BGY": "Milan",
    "VCE": "Venice",    "TSF": "Venice",
    "TRN": "Turin",
    "FCO": "Rome",      "CIA": "Rome",
    "NAP": "Naples",
    "MAD": "Madrid",
    "BCN": "Barcelona",
    "VLC": "Valencia",
    "SVQ": "Seville",
    "LIS": "Lisbon",
    "OPO": "Porto",
    "LYS": "Lyon",
    "MRS": "Marseille",
    "TLS": "Toulouse",
    "SXB": "Strasbourg", "BSL": "Strasbourg", "EAP": "Strasbourg", "MLH": "Strasbourg",
    # New cities
    "RTM": "Rotterdam",
    "LIL": "Lille",
    # The Hague: no dedicated airport (uses AMS/RTM)
    "BIO": "Bilbao",
    "BOD": "Bordeaux",
    "DUS": "Dusseldorf",
    "PMO": "Palermo",
    "LEJ": "Leipzig",
    "DTM": "Dortmund",  # moved from Cologne
    "AGP": "Malaga",
    # Essen: no dedicated airport (uses DUS)
    "BRE": "Bremen",
    "DRS": "Dresden",
    "GOA": "Genoa",
    "HAJ": "Hannover",
    "ANR": "Antwerp",
    "NUE": "Nuremberg",
    # Duisburg: no dedicated airport (uses DUS)
    "NTE": "Nantes",
    "NCE": "Nice",
    "MJV": "Murcia",    "RMU": "Murcia",
    # Utrecht: no dedicated airport (uses AMS)
    "PMI": "Palma",
    "GRX": "Granada",
    "LGG": "Liege",
}

GTFS_FEEDS = {
    "Germany":     os.path.join(RAW_DIR, "germany_db.zip"),
    "France":      os.path.join(RAW_DIR, "france_sncf.zip"),
    "Netherlands": os.path.join(RAW_DIR, "netherlands.zip"),
    "Switzerland": os.path.join(RAW_DIR, "switzerland_sbb.zip"),
    "Austria":     os.path.join(RAW_DIR, "austria_oebb.zip"),
    "Spain":       os.path.join(RAW_DIR, "spain_renfe.zip"),
    "Portugal":    os.path.join(RAW_DIR, "portugal_cp.zip"),
    "Belgium":     os.path.join(RAW_DIR, "belgium_sncb_netex.zip"),
    # Eurostar international services (London, Brussels, Paris, Amsterdam, Cologne).
    # CITATION: Eurostar (2026). GTFS static commercial feed v2.
    #           Feed covers 2026-04-29 to 2026-07-27.
    "Eurostar":    os.path.join(RAW_DIR, "gtfs_static_commercial_v2.zip"),
}

NETEX_FEEDS = {
    "Italy": os.path.join(RAW_DIR, "IT-IT-TRENITALIA_L1.xml.gz"),
}

# ── Manual rail edges (cross-border routes absent from all GTFS/NeTEx feeds) ──
# Porto→Madrid via CP Celta (Porto→Vigo-Guixar, ~150 min) + Renfe Alvia
# (Vigo-Guixar→Madrid-Chamartin, ~155 min). Vigo is a transfer point, not a
# target city, so no GTFS feed produces this city-pair automatically.
# Total journey incl. connection: ~6 h (360 min). Frequency: ~1 round trip/day.
# CITATION: CP (2024). Celta Porto–Vigo timetable. https://www.cp.pt
#           Renfe (2024). Alvia Galicia–Madrid timetable. https://www.renfe.com
MANUAL_RAIL_EDGES = [
    {"from_city": "Porto",  "to_city": "Madrid", "operator": "Renfe/CP",
     "travel_time_min": 360, "frequency_per_day": 1.0, "night_train": False},
    {"from_city": "Madrid", "to_city": "Porto",  "operator": "Renfe/CP",
     "travel_time_min": 360, "frequency_per_day": 1.0, "night_train": False},
]



# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_city(lat, lon, city_df, radius_km=CITY_MATCH_KM):
    dists = city_df.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
    idx = dists.idxmin()
    return idx if dists[idx] <= radius_km else None


def parse_time_to_min(t):
    """Parse GTFS/NeTEx HH:MM:SS time string to minutes. Handles times > 24:00."""
    try:
        h, m, s = (t or "").strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def open_zip(path):
    if not os.path.exists(path):
        print(f"  Skipping — file not found: {path}")
        return None
    try:
        return zipfile.ZipFile(path)
    except Exception as e:
        print(f"  Not a valid zip: {e}")
        return None


def find_in_zip(zf, filename):
    for name in zf.namelist():
        if name.endswith("/" + filename) or name == filename:
            return name
    return None


def read_csv_from_zip(zf, filename):
    member = find_in_zip(zf, filename)
    if member is None:
        return pd.DataFrame()
    return pd.read_csv(zf.open(member), low_memory=False, dtype=str)


# ── GTFS frequency helpers ────────────────────────────────────────────────────

def load_service_days(zf):
    """
    Return {service_id: avg_days_per_week} from calendar.txt.
    Falls back to calendar_dates.txt if calendar.txt is absent.
    Returns {} if neither file is available.
    """
    cal = read_csv_from_zip(zf, "calendar.txt")
    if not cal.empty and "service_id" in cal.columns:
        day_cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        present  = [c for c in day_cols if c in cal.columns]
        cal["days_per_week"] = cal[present].apply(
            lambda r: sum(1 for v in r if str(v).strip() == "1"), axis=1
        )
        return cal.set_index("service_id")["days_per_week"].to_dict()

    cd = read_csv_from_zip(zf, "calendar_dates.txt")
    if not cd.empty and "service_id" in cd.columns:
        cd = cd[cd.get("exception_type", pd.Series(["1"] * len(cd))).astype(str) == "1"]
        counts = cd.groupby("service_id")["date"].nunique()
        # Use the feed's full date span — not counts.max() — so that feeds where
        # each service_id covers exactly one date (expanded-per-day pattern) give
        # the correct days/week fraction rather than 7.0 for every trip.
        try:
            dates    = pd.to_datetime(cd["date"], format="%Y%m%d")
            n_days   = max((dates.max() - dates.min()).days + 1, 1)
        except Exception:
            n_days   = max(int(counts.max()), 1) if not counts.empty else 90
        return (counts / n_days * 7).clip(0, 7).round(4).to_dict()

    return {}


def load_trip_service_map(zf):
    """Return {trip_id: service_id} from trips.txt."""
    trips = read_csv_from_zip(zf, "trips.txt")
    if trips.empty or "service_id" not in trips.columns:
        return {}
    return trips.set_index("trip_id")["service_id"].to_dict()


def is_night_service(dep_min, last_arr_min, duration_min):
    """
    True if the service is a genuine overnight sleeper train.
    Requires departure >= 19:00 AND (
      crosses midnight with journey >= 5 h, OR total duration >= 6 h
    ).
    The 5 h / 300 min gate on the midnight-crossing path excludes late Eurostar
    services (London-Amsterdam ~4 h) while keeping Nightjet routes (8–13 h).
    dep_min and last_arr_min are in GTFS minutes (may exceed 1440 for next-day).
    """
    dep_hour = int(dep_min % 1440 // 60)
    crosses_midnight = last_arr_min >= 1440 and duration_min >= 300
    is_long_haul     = duration_min >= 360
    return dep_hour >= 19 and (crosses_midnight or is_long_haul)


# ── GTFS processing ───────────────────────────────────────────────────────────

def process_gtfs(name, path, city_df):
    """Read a GTFS zip and return rail edges with frequency_per_day and night_train."""
    print(f"\n[{name}] Reading {os.path.basename(path)}...", flush=True)
    zf = open_zip(path)
    if zf is None:
        return pd.DataFrame()

    routes     = read_csv_from_zip(zf, "routes.txt")
    trips_df   = read_csv_from_zip(zf, "trips.txt")
    stops      = read_csv_from_zip(zf, "stops.txt")
    stop_times = read_csv_from_zip(zf, "stop_times.txt")

    if any(df.empty for df in [routes, trips_df, stops, stop_times]):
        missing = [n for n, df in zip(
            ["routes", "trips", "stops", "stop_times"],
            [routes, trips_df, stops, stop_times]) if df.empty]
        print(f"  Skipping — missing: {missing}")
        return pd.DataFrame()

    service_days = load_service_days(zf)
    trip_service = load_trip_service_map(zf)
    no_calendar  = not service_days

    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce")
    rail_routes  = routes[routes["route_type"].isin(RAIL_ROUTE_TYPES)]
    if rail_routes.empty:
        print(f"  No rail routes found.")
        return pd.DataFrame()
    rail_trip_ids = set(trips_df[trips_df["route_id"].isin(rail_routes["route_id"])]["trip_id"])

    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    print(f"  Mapping {len(stops):,} stops to cities...", flush=True)
    stops["city"] = stops.apply(
        lambda r: nearest_city(r["stop_lat"], r["stop_lon"], city_df), axis=1
    )
    city_stop_map = stops[stops["city"].notna()].set_index("stop_id")["city"].to_dict()
    if not city_stop_map:
        print(f"  No stops matched to target cities.")
        return pd.DataFrame()
    print(f"  {len(city_stop_map)} stops -> {len(set(city_stop_map.values()))} cities.", flush=True)

    st = stop_times[
        stop_times["trip_id"].isin(rail_trip_ids) &
        stop_times["stop_id"].isin(city_stop_map)
    ].copy()
    st["city"]    = st["stop_id"].map(city_stop_map)
    st["arr_min"] = st["arrival_time"].apply(parse_time_to_min)
    st["seq"]     = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["arr_min", "seq"]).sort_values(["trip_id", "seq"])

    # Accumulate per (from_city, to_city)
    edge_acc = defaultdict(lambda: {"min_time": float("inf"), "days": 0.0, "night": False})

    for trip_id, grp in st.groupby("trip_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue

        service_id = trip_service.get(trip_id)
        if no_calendar:
            days_pw = 7.0  # assume daily when no calendar info
        else:
            days_pw = float(service_days.get(service_id, 1.0)) if service_id else 1.0

        first_dep = grp.loc[0, "arr_min"]
        last_arr  = grp.loc[len(grp) - 1, "arr_min"]
        night     = is_night_service(first_dep, last_arr, last_arr - first_dep)

        for i in range(len(grp) - 1):
            fc, tc = grp.loc[i, "city"], grp.loc[i + 1, "city"]
            t0, t1 = grp.loc[i, "arr_min"], grp.loc[i + 1, "arr_min"]
            if fc == tc or t1 <= t0:
                continue
            seg_time = t1 - t0
            if seg_time < MIN_SEGMENT_MIN:
                # Pass-through stop: high-speed train crosses station without
                # stopping. GTFS records a phantom segment of 1–9 min that
                # would override legitimate stopping-service times via the min().
                continue
            key = (fc, tc)
            rec = edge_acc[key]
            rec["min_time"] = min(rec["min_time"], seg_time)
            rec["days"] += days_pw
            if night:
                rec["night"] = True

    if not edge_acc:
        print(f"  No city-pair edges extracted.")
        return pd.DataFrame()

    rows = [
        {"from_city": fc, "to_city": tc, "operator": name,
         "travel_time_min": int(rec["min_time"]),
         "frequency_per_day": round(rec["days"] / 7, 1),
         "night_train": bool(rec["night"])}
        for (fc, tc), rec in edge_acc.items()
    ]
    print(f"  {len(rows)} city-pair connections.")
    return pd.DataFrame(rows)


# ── NeTEx processing (Italy) ──────────────────────────────────────────────────

def strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag

def iter_tag(root, tag):
    return [el for el in root.iter() if strip_ns(el.tag) == tag]


def process_netex_italy(path, city_df):
    """
    Parse Italian Trenitalia NeTEx (IT-IT-TRENITALIA_L1.xml.gz).
    Two-pass streaming parser. Returns edges with frequency and night_train.
    Frequency = distinct ServiceJourneys per city pair (each assumed to run daily).
    DayOffset handled for next-day arrivals.
    """
    import gzip as _gzip

    print(f"\n[Italy] Reading {os.path.basename(path)} (streaming)...", flush=True)
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return pd.DataFrame()

    def open_gz():
        return _gzip.open(path, "rb")

    # ── Pass 1: stop and pattern lookups ──────────────────────────────────────
    print("  Pass 1: mapping stops and journey patterns...", flush=True)
    ssp_city  = {}   # ScheduledStopPoint.id → city
    spijp_ssp = {}   # StopPointInJourneyPattern.id → ScheduledStopPoint.id

    with open_gz() as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            tag = strip_ns(elem.tag)
            if tag == "ScheduledStopPoint":
                sid = elem.get("id", "")
                loc = next(iter(iter_tag(elem, "Location")), None)
                if loc is not None:
                    lat_el = next(iter(iter_tag(loc, "Latitude")), None)
                    lon_el = next(iter(iter_tag(loc, "Longitude")), None)
                    if lat_el is not None and lon_el is not None:
                        try:
                            city = nearest_city(float(lat_el.text), float(lon_el.text),
                                                city_df, radius_km=50)
                            if city:
                                ssp_city[sid] = city
                        except (ValueError, TypeError):
                            pass
                elem.clear()
            elif tag == "StopPointInJourneyPattern":
                spijp_id = elem.get("id", "")
                ssp_ref  = next(iter(iter_tag(elem, "ScheduledStopPointRef")), None)
                if ssp_ref is not None:
                    ref = ssp_ref.get("ref", "")
                    if ref:
                        spijp_ssp[spijp_id] = ref
                elem.clear()

    print(f"  {len(ssp_city)} stops -> cities, {len(spijp_ssp)} pattern points.", flush=True)
    if not ssp_city or not spijp_ssp:
        print("  No usable stop/pattern data.")
        return pd.DataFrame()

    # ── Pass 2: extract ServiceJourney edges ──────────────────────────────────
    print("  Pass 2: extracting service journeys...", flush=True)
    edge_acc  = defaultdict(lambda: {"min_time": float("inf"), "count": 0, "night": False})
    n_journeys = 0

    with open_gz() as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if strip_ns(elem.tag) != "ServiceJourney":
                continue
            n_journeys += 1

            mode_el = next(iter(iter_tag(elem, "TransportMode")), None)
            if mode_el is not None and "rail" not in (mode_el.text or "").lower():
                elem.clear()
                continue

            pts = []  # (city, minutes_from_midnight_possibly_>1440)
            for ptime in iter_tag(elem, "TimetabledPassingTime"):
                spijp_ref_el = next(iter(iter_tag(ptime, "StopPointInJourneyPatternRef")), None)
                arr_el  = next(iter(iter_tag(ptime, "ArrivalTime")), None)
                dep_el  = next(iter(iter_tag(ptime, "DepartureTime")), None)
                time_el = arr_el if arr_el is not None else dep_el
                # DayOffset: NeTEx next-day convention
                dayoff_el = next(iter(iter_tag(ptime, "DayOffset")), None)
                day_offset = 0
                if dayoff_el is not None and dayoff_el.text:
                    try:
                        day_offset = int(dayoff_el.text.strip())
                    except ValueError:
                        pass

                if spijp_ref_el is None or time_el is None:
                    continue
                spijp_id = spijp_ref_el.get("ref", "")
                ssp_id   = spijp_ssp.get(spijp_id)
                city     = ssp_city.get(ssp_id) if ssp_id else None
                mins     = parse_time_to_min(time_el.text or "")
                if city and mins is not None:
                    pts.append((city, mins + day_offset * 1440))

            if len(pts) < 2:
                elem.clear()
                continue

            first_dep = pts[0][1]
            last_arr  = pts[-1][1]
            night     = is_night_service(first_dep, last_arr, last_arr - first_dep)

            for i in range(len(pts) - 1):
                fc, t0 = pts[i]
                tc, t1 = pts[i + 1]
                if fc != tc and t1 > t0:
                    seg_time = t1 - t0
                    if seg_time < MIN_SEGMENT_MIN:
                        continue
                    key = (fc, tc)
                    rec = edge_acc[key]
                    rec["min_time"] = min(rec["min_time"], seg_time)
                    rec["count"]   += 1
                    if night:
                        rec["night"] = True
            elem.clear()

    print(f"  Processed {n_journeys:,} journeys.", flush=True)
    if not edge_acc:
        print("  No city-pair edges extracted.")
        return pd.DataFrame()

    rows = [
        {"from_city": fc, "to_city": tc, "operator": "Italy",
         "travel_time_min": int(rec["min_time"]),
         "frequency_per_day": float(rec["count"]),   # each journey assumed daily
         "night_train": bool(rec["night"])}
        for (fc, tc), rec in edge_acc.items()
    ]
    print(f"  {len(rows)} city-pair connections.")
    return pd.DataFrame(rows)


# ── NeTEx processing (generic, legacy — not called for current feeds) ─────────

def load_xml_contents(path):
    if not os.path.exists(path):
        return []
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        return [(f, zf.open(f).read()) for f in zf.namelist() if f.endswith(".xml")]
    elif path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            return [(m.name, tf.extractfile(m).read())
                    for m in tf.getmembers() if m.name.endswith(".xml") and tf.extractfile(m)]
    elif path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return [(os.path.basename(path), f.read())]
    else:
        with open(path, "rb") as f:
            return [(os.path.basename(path), f.read())]


def process_netex(name, path, city_df):
    """Generic NeTEx parser (legacy). Does not extract frequency or night_train."""
    print(f"\n[{name}] Reading NeTEx {os.path.basename(path)}...", flush=True)
    try:
        xml_contents = load_xml_contents(path)
    except Exception as e:
        print(f"  Could not open: {e}")
        return pd.DataFrame()
    if not xml_contents:
        print(f"  No XML content found.")
        return pd.DataFrame()

    stops    = {}
    journeys = {}

    for xml_name, xml_bytes in xml_contents:
        try:
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            print(f"  Could not parse {xml_name}: {e}")
            continue
        for tag in ("ScheduledStopPoint", "StopPlace"):
            for sp in iter_tag(root, tag):
                sid = sp.get("id", "")
                loc = next(iter(iter_tag(sp, "Location")), None)
                lat, lon = None, None
                if loc is not None:
                    lat_el = next(iter(iter_tag(loc, "Latitude")), None)
                    lon_el = next(iter(iter_tag(loc, "Longitude")), None)
                    if lat_el is not None and lon_el is not None:
                        try:
                            lat, lon = float(lat_el.text), float(lon_el.text)
                        except (ValueError, TypeError):
                            pass
                if lat is None:
                    pos_el = next(iter(iter_tag(sp, "pos")), None)
                    if pos_el is not None:
                        try:
                            parts = pos_el.text.strip().split()
                            lat, lon = float(parts[0]), float(parts[1])
                        except Exception:
                            pass
                if lat is not None and lon is not None:
                    city = nearest_city(lat, lon, city_df, radius_km=50)
                    if city:
                        stops[sid] = city

        rail_line_ids = set()
        for line in iter_tag(root, "Line"):
            mode_el = next(iter(iter_tag(line, "TransportMode")), None)
            if mode_el is not None and "rail" in (mode_el.text or "").lower():
                lid = line.get("id", "")
                if lid:
                    rail_line_ids.add(lid)

        for sj in iter_tag(root, "ServiceJourney"):
            line_ref = next(iter(iter_tag(sj, "LineRef")), None)
            if line_ref is not None and line_ref.get("ref", "") not in rail_line_ids:
                continue
            jid = sj.get("id", "")
            pts = []
            for ptime in iter_tag(sj, "TimetabledPassingTime"):
                stop_ref = next(iter(iter_tag(ptime, "StopPointRef")), None)
                arr_el   = next(iter(iter_tag(ptime, "ArrivalTime")), None)
                dep_el   = next(iter(iter_tag(ptime, "DepartureTime")), None)
                time_el  = arr_el if arr_el is not None else dep_el
                if stop_ref is None or time_el is None:
                    continue
                sid  = stop_ref.get("ref", "") or (stop_ref.text or "")
                city = stops.get(sid)
                mins = parse_time_to_min(time_el.text or "")
                if city and mins is not None:
                    pts.append((city, mins))
            if len(pts) >= 2:
                journeys[jid] = pts

    if not journeys:
        return pd.DataFrame()

    edges = []
    for jid, pts in journeys.items():
        for i in range(len(pts) - 1):
            fc, t0 = pts[i]
            tc, t1 = pts[i + 1]
            if fc != tc and t1 > t0:
                edges.append({"from_city": fc, "to_city": tc,
                               "travel_time_min": t1 - t0, "operator": name,
                               "frequency_per_day": 1.0, "night_train": False})
    if not edges:
        return pd.DataFrame()

    df = (pd.DataFrame(edges)
            .groupby(["from_city", "to_city", "operator"])
            .agg(travel_time_min=("travel_time_min", "min"),
                 frequency_per_day=("frequency_per_day", "sum"),
                 night_train=("night_train", "any"))
            .reset_index())
    print(f"  {len(df)} city-pair connections.")
    return df


# ── Air processing ────────────────────────────────────────────────────────────

def process_air(city_df):
    """
    Extract direct flights between target cities from OpenFlights + optional
    OpenSky frequency data.

    Travel times: computed from haversine distance (OpenFlights routes.dat).

    Frequency proxy: n_distinct_airlines * 3.5, capped at [2, 20].
    Per-route frequency data is not freely available for European aviation
    (OpenSky historical access requires institutional affiliation; OAG/Cirium
    are commercial). The proxy captures relative ordering adequately for the
    ABM's stranding analysis. Documented as a known limitation.
    """
    print("\n[OpenFlights] Reading airports + routes...", flush=True)

    ap_cols = ["airport_id","name","city","country","iata","icao",
               "lat","lon","alt","tz","dst","tz_db","type","source"]
    rt_cols = ["airline","airline_id","src_iata","src_id",
               "dst_iata","dst_id","codeshare","stops","equipment"]

    ap_local = os.path.join(RAW_DIR, "airports.dat.txt")
    rt_local = os.path.join(RAW_DIR, "routes.dat.txt")
    try:
        airports = pd.read_csv(ap_local, header=None, names=ap_cols, na_values=["\\N"])
        routes   = pd.read_csv(rt_local, header=None, names=rt_cols, na_values=["\\N"])
    except Exception as e:
        print(f"  Failed to read OpenFlights data: {e}")
        return pd.DataFrame()

    routes = routes[routes["stops"].astype(str) == "0"].copy()
    routes["from_city"] = routes["src_iata"].map(IATA_TO_CITY)
    routes["to_city"]   = routes["dst_iata"].map(IATA_TO_CITY)
    routes = routes.dropna(subset=["from_city", "to_city"])
    routes = routes[routes["from_city"] != routes["to_city"]]

    if routes.empty:
        print("  No air routes found.")
        return pd.DataFrame()

    city_coords = {c: (row["lat"], row["lon"]) for c, row in city_df.iterrows()}

    def flight_time(row):
        la1, lo1 = city_coords[row["from_city"]]
        la2, lo2 = city_coords[row["to_city"]]
        dist = haversine_km(la1, lo1, la2, lo2)
        return round((dist / 800) * 60 + 45)

    routes["travel_time_min"] = routes.apply(flight_time, axis=1)
    result = (routes.groupby(["from_city", "to_city"])["travel_time_min"]
              .min().reset_index())

    # ── Frequency: OPDI real data if available, else n_airlines proxy ────────
    opdi_path = os.path.join(OUTPUT_DIR, "air_frequency_opdi.csv")
    if os.path.exists(opdi_path):
        print(f"  Using OPDI frequency data from {opdi_path}", flush=True)
        opdi = pd.read_csv(opdi_path)
        result = result.merge(opdi, on=["from_city", "to_city"], how="left")
        # proxy fill for any routes not in OPDI
        missing = result["frequency_per_day"].isna()
        if missing.any():
            airline_counts = (routes.groupby(["from_city", "to_city"])["airline"]
                              .nunique().reset_index()
                              .rename(columns={"airline": "n_airlines"}))
            result = result.merge(airline_counts, on=["from_city", "to_city"], how="left")
            result.loc[missing, "frequency_per_day"] = (
                (result.loc[missing, "n_airlines"].fillna(1) * 3.5).clip(2, 20)
            )
            print(f"  {missing.sum()} routes filled with proxy (not in OPDI).")
    else:
        print("  OPDI frequency file not found — using n_airlines proxy.", flush=True)
        airline_counts = (routes.groupby(["from_city", "to_city"])["airline"]
                          .nunique().reset_index()
                          .rename(columns={"airline": "n_airlines"}))
        result = result.merge(airline_counts, on=["from_city", "to_city"], how="left")
        result["frequency_per_day"] = (
            (result["n_airlines"].fillna(1) * 3.5).clip(2, 20).round(1)
        )

    result["frequency_per_day"] = result["frequency_per_day"].round(1)
    result["operator"] = "air"
    print(f"  {len(result)} city-pair air connections.")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cities_path = os.path.join(OUTPUT_DIR, "cities.csv")
    if not os.path.exists(cities_path):
        print("ERROR: data/cities.csv not found. Run process_cities.py first.")
        return
    city_df = pd.read_csv(cities_path, dtype={"country": str})
    city_df = city_df.set_index("city")
    city_df[["lat","lon"]] = city_df[["lat","lon"]].apply(pd.to_numeric, errors="coerce")
    city_df = city_df.dropna(subset=["lat","lon"]).reset_index().set_index("city")
    print(f"Loaded {len(city_df)} cities from cities.csv")

    all_rail = []

    for name, path in GTFS_FEEDS.items():
        df = process_gtfs(name, path, city_df)
        if not df.empty:
            all_rail.append(df)

    for name, path in NETEX_FEEDS.items():
        df = (process_netex_italy(path, city_df) if name == "Italy"
              else process_netex(name, path, city_df))
        if not df.empty:
            all_rail.append(df)


    if all_rail:
        rail_raw = pd.concat(all_rail, ignore_index=True)
        # For each (from, to) pair: keep fastest travel time, sum frequencies,
        # flag night_train if any feed says so.
        rail = (rail_raw
                .groupby(["from_city", "to_city"])
                .agg(
                    operator         =("operator",          "first"),
                    travel_time_min  =("travel_time_min",   "min"),
                    frequency_per_day=("frequency_per_day", "sum"),
                    night_train      =("night_train",       "any"),
                )
                .reset_index())
        rail["night_train"] = rail["night_train"].astype(bool)

        # Append manual edges (idempotent: skips pairs already captured by GTFS)
        manual_df = pd.DataFrame(MANUAL_RAIL_EDGES)
        existing  = set(zip(rail["from_city"], rail["to_city"]))
        to_add    = manual_df[~manual_df.apply(
            lambda r: (r["from_city"], r["to_city"]) in existing, axis=1
        )]
        if not to_add.empty:
            rail = pd.concat([rail, to_add], ignore_index=True)
            added = ", ".join(f"{r.from_city}→{r.to_city}" for _, r in to_add.iterrows())
            print(f"  Manual edges added: {added}")

        rail.to_csv(f"{OUTPUT_DIR}/rail_edges.csv", index=False)
        print(f"\nSaved rail_edges.csv  ({len(rail)} connections)")
        night = rail[rail["night_train"]]
        print(f"  Night trains: {len(night)}")
        for _, r in night.iterrows():
            print(f"    {r.from_city} -> {r.to_city}: {r.travel_time_min} min, "
                  f"{r.frequency_per_day}/day")
    else:
        print("\nWarning: no rail edges extracted.")

    air = process_air(city_df)
    if not air.empty:
        air.to_csv(f"{OUTPUT_DIR}/air_edges.csv", index=False)
        print(f"Saved air_edges.csv   ({len(air)} connections)")

    print("\nAll done. Files in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
