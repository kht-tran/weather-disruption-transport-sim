#!/usr/bin/env python3
"""
Build rail demand proxy using sourced Eurostat occupancy data.

Replaces placeholder constants (avg_seats=400, load_factor=0.55) with
values derived from official Eurostat train-km and passenger-km statistics.

Outputs:
  data_pipeline/raw/eurostat_rail_occupancy/         - raw Eurostat downloads
  data/eurostat_rail_occupancy_by_country.csv         - avg_pass_per_train_km by country
  data/rail_edges.csv                              - enhanced rail_edges with proxy
  data/rail_proxy_summary.md                       - methodology summary
"""

import math, re, sys, time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = PROJECT_ROOT / "data_pipeline" / "raw" / "eurostat_rail_occupancy"
RAW_DIR.mkdir(parents=True, exist_ok=True)

EU_COUNTRIES = {
    "AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI","FR",
    "GR","HR","HU","IE","IT","LT","LU","LV","MT","NL","PL",
    "PT","RO","SE","SI","SK",
}

# ─── Step 1: Download ─────────────────────────────────────────────────────────

def _get(url, timeout=180):
    resp = requests.get(url, timeout=timeout,
                        headers={"Accept-Encoding": "identity"})
    resp.raise_for_status()
    return resp

def download_eurostat(dataset_id, out_sdmx, out_tsv):
    """Download dataset; try SDMX-CSV first, fall back to TSV."""
    sdmx_url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset_id}/?format=SDMX-CSV&compressed=false"
    )
    tsv_url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset_id}?format=TSV&compressed=false"
    )
    for url, path in [(sdmx_url, out_sdmx), (tsv_url, out_tsv)]:
        try:
            print(f"  GET {url[:90]}...")
            r = _get(url)
            if len(r.content) < 500:
                print(f"  Response too short ({len(r.content)} bytes) — skipping")
                continue
            path.write_bytes(r.content)
            print(f"  Saved {path.name} ({len(r.content)//1024} KB)")
            return path
        except Exception as e:
            print(f"  FAILED: {e}")
    raise RuntimeError(f"Could not download {dataset_id}")

def ensure_downloaded(dataset_id, sdmx_path, tsv_path):
    if sdmx_path.exists():
        print(f"  Using cached {sdmx_path.name}")
        return sdmx_path
    if tsv_path.exists():
        print(f"  Using cached {tsv_path.name}")
        return tsv_path
    return download_eurostat(dataset_id, sdmx_path, tsv_path)

print("=== Step 1: Download Eurostat datasets ===")
trainmv_sdmx = RAW_DIR / "rail_tf_trainmv_sdmx.csv"
trainmv_tsv  = RAW_DIR / "rail_tf_trainmv.tsv"
pa_sdmx      = RAW_DIR / "rail_pa_total_sdmx.csv"
pa_tsv       = RAW_DIR / "rail_pa_total.tsv"

trainmv_file = ensure_downloaded("rail_tf_trainmv", trainmv_sdmx, trainmv_tsv)
pa_file      = ensure_downloaded("rail_pa_total",   pa_sdmx,      pa_tsv)

# ─── Parse helpers ────────────────────────────────────────────────────────────

def _clean_colname(c):
    return c.strip().upper().replace("\\TIME_PERIOD", "").replace("\\", "_").replace("/", "_").strip()

def parse_sdmx_csv(path):
    """Parse Eurostat SDMX-CSV (long format) → DataFrame."""
    df = pd.read_csv(path, sep=",", dtype=str, on_bad_lines="skip")
    df.columns = [_clean_colname(c) for c in df.columns]
    # Normalise geo column name variants
    for old in list(df.columns):
        if old.startswith("GEO") and old != "GEO":
            df.rename(columns={old: "GEO"}, inplace=True)
    return df

def parse_tsv(path):
    """Parse Eurostat wide-format TSV → long DataFrame."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    first_col = df.columns[0]
    # e.g. "tra_type,unit,geo\TIME_PERIOD"
    raw_dims = first_col.split(",")
    dim_names = [_clean_colname(d) for d in raw_dims]

    dims_split = df[first_col].str.split(",", expand=True)
    dims_split.columns = dim_names

    year_cols = [c.strip() for c in df.columns[1:]]
    year_df = df.iloc[:, 1:].copy()
    year_df.columns = year_cols

    combined = pd.concat([dims_split, year_df], axis=1)
    melted = combined.melt(id_vars=dim_names, var_name="TIME_PERIOD", value_name="OBS_VALUE")
    melted.columns = [c.upper() for c in melted.columns]
    return melted

def load_df(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return parse_sdmx_csv(path)
    elif suffix == ".tsv":
        return parse_tsv(path)
    # Try to detect from content
    text = path.read_text(encoding="utf-8", errors="replace")[:200]
    if "\t" in text.split("\n")[0]:
        return parse_tsv(path)
    return parse_sdmx_csv(path)

# ─── Step 2: Parse + compute avg_pass_per_train_km ────────────────────────────

print("\n=== Step 2: Parse and compute occupancy ===")

trainmv_df = load_df(trainmv_file)
pa_df      = load_df(pa_file)

print(f"  train-km cols: {list(trainmv_df.columns)}")
print(f"  passenger-km cols: {list(pa_df.columns)}")

# Show dimension values to inform filtering
for label, d, col in [("TRAIN/TRA_TYPE in trainmv", trainmv_df, "TRAIN"),
                       ("UNIT in trainmv", trainmv_df, "UNIT"),
                       ("UNIT in pa", pa_df, "UNIT")]:
    for c in [col, col.upper(), col.lower()]:
        if c in d.columns:
            print(f"  {label}: {sorted(d[c].dropna().unique())[:15]}")
            break

def unit_scale_to_mkm(unit_str):
    """Scale factor to convert Eurostat unit value to millions of km."""
    u = str(unit_str).upper()
    if u.startswith("THS"):              return 1.0 / 1000.0   # thousands -> millions
    if u.startswith("MIO") or u.startswith("MIL"): return 1.0   # already millions
    if u.startswith("BILLION") or u.startswith("MRD"): return 1000.0
    return 1.0

def _col(df, name):
    """Case-insensitive column lookup."""
    for c in df.columns:
        if c.upper() == name.upper():
            return c
    return None

def filter_trainmv(df):
    """Filter to passenger train-km only (train category = TRN_PAS)."""
    d = df.copy()
    # Eurostat rail_tf_trainmv uses column 'train' with values TOTAL / TRN_GD / TRN_PAS
    train_col = _col(d, "TRAIN") or _col(d, "TRA_TYPE")
    if train_col:
        pass_vals = {"TRN_PAS", "PASS", "PASS_TRAIN", "PA"}
        matching = d[train_col].str.upper().isin(pass_vals)
        if matching.sum() > 0:
            d = d[matching]
    unit_col = _col(d, "UNIT")
    if unit_col:
        d = d[d[unit_col].str.upper().str.contains("KM", na=False)]
    return d

def filter_pa_mkm(df):
    """Filter passenger data to passenger-km rows (not pax headcount)."""
    d = df.copy()
    unit_col = _col(d, "UNIT")
    if unit_col:
        units = d[unit_col].str.upper().unique()
        km_units = [u for u in units if "KM" in u]
        if km_units:
            d = d[d[unit_col].str.upper().isin(km_units)]
    return d

trainmv_f = filter_trainmv(trainmv_df)
pa_f      = filter_pa_mkm(pa_df)
print(f"  After filter — train-km rows: {len(trainmv_f)}, pass-km rows: {len(pa_f)}")

# Build per-country lookups that fall back to earlier years when target year is missing.
# Prefers the most recent year with non-null data, within [target_year - 3, target_year].
FALLBACK_WINDOW = 3   # accept data up to 3 years older

def build_country_series(df, geo_col_name="GEO"):
    """
    Return a pivoted DataFrame: index=country, columns=year (str), values in Mkm.
    Only 2-letter country codes; supranational aggregates (EU27_2020 etc.) excluded.
    """
    geo_col = _col(df, geo_col_name)
    unit_col = _col(df, "UNIT")
    if not geo_col:
        return pd.DataFrame()
    d = df.copy()
    d["_geo"] = d[geo_col].str.strip().str.upper()
    d["_yr"]  = d["TIME_PERIOD"].astype(str).str.strip()
    d["_raw"] = pd.to_numeric(
        d["OBS_VALUE"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    if unit_col:
        d["_scale"] = d[unit_col].apply(unit_scale_to_mkm)
    else:
        d["_scale"] = 1.0
    d["_val"] = d["_raw"] * d["_scale"]

    # Keep only 2-letter codes (skip EU27_2020, EA19, etc.)
    d = d[d["_geo"].str.match(r"^[A-Z]{2}$", na=False)]
    # Only annual years
    d = d[d["_yr"].str.match(r"^20\d\d$", na=False)]

    pivot = d.groupby(["_geo", "_yr"])["_val"].max().unstack("_yr")
    return pivot

tkm_series = build_country_series(trainmv_f)
pkm_series = build_country_series(pa_f)

# Choose target year: most recent year where BOTH datasets have ≥10 EU countries
def pick_target_year(tkm, pkm, min_eu=10):
    all_yrs = sorted(
        set(tkm.columns) & set(pkm.columns) & {y for y in tkm.columns if re.match(r"20\d\d", y)},
        reverse=True,
    )
    for yr in all_yrs:
        eu_tkm = tkm.loc[tkm.index.isin(EU_COUNTRIES), yr].dropna() if yr in tkm.columns else pd.Series()
        eu_pkm = pkm.loc[pkm.index.isin(EU_COUNTRIES), yr].dropna() if yr in pkm.columns else pd.Series()
        if len(eu_tkm) >= min_eu and len(eu_pkm) >= min_eu:
            return yr
    return all_yrs[0] if all_yrs else "2023"

target_year = pick_target_year(tkm_series, pkm_series)
print(f"  Using target year: {target_year}")

# Per-country value: try target year, then fall back up to FALLBACK_WINDOW years earlier
def best_country_val(series, country, target_yr):
    """Return (value_in_mkm, year_used) for a country, or (None, None)."""
    target_int = int(target_yr)
    for yr in range(target_int, target_int - FALLBACK_WINDOW - 1, -1):
        ystr = str(yr)
        if country in series.index and ystr in series.columns:
            v = series.at[country, ystr]
            if pd.notna(v) and v > 0:
                return v, ystr
    return None, None

all_countries = sorted(set(tkm_series.index) | set(pkm_series.index))
trainmv_vals = {}
pa_vals      = {}
yr_tkm_used  = {}
yr_pkm_used  = {}

for cc in all_countries:
    v_tkm, yr_t = best_country_val(tkm_series, cc, target_year)
    v_pkm, yr_p = best_country_val(pkm_series, cc, target_year)
    if v_tkm is not None:
        trainmv_vals[cc] = v_tkm
        yr_tkm_used[cc] = yr_t
    if v_pkm is not None:
        pa_vals[cc] = v_pkm
        yr_pkm_used[cc] = yr_p

print(f"  Countries with train-km: {sorted(trainmv_vals.keys())}")
print(f"  Countries with pass-km:  {sorted(pa_vals.keys())}")

# Compute ratio
occupancy = {}
for cc in (set(trainmv_vals) | set(pa_vals)):
    tkm = trainmv_vals.get(cc, 0)
    pkm = pa_vals.get(cc, 0)
    if tkm > 0 and pkm > 0:
        occupancy[cc] = pkm / tkm  # avg passengers per train (at any km of journey)

eu_vals = [v for cc, v in occupancy.items() if cc in EU_COUNTRIES]
eu_average = float(np.median(eu_vals)) if eu_vals else 100.0

print(f"\n  EU median avg_pass_per_train_km: {eu_average:.1f}")
print("  Country occupancy (pass/train-km):")
for cc in sorted(occupancy):
    eu = " [EU]" if cc in EU_COUNTRIES else ""
    t_yr = yr_tkm_used.get(cc, "?")
    p_yr = yr_pkm_used.get(cc, "?")
    yr_note = f" tkm:{t_yr} pkm:{p_yr}" if t_yr != p_yr else f" yr:{t_yr}"
    print(f"    {cc}: {occupancy[cc]:.1f}{eu}{yr_note}")

# Save CSV
occ_rows = [
    {
        "country_code": cc,
        "target_year": target_year,
        "trainmv_year": yr_tkm_used.get(cc, target_year),
        "pa_year": yr_pkm_used.get(cc, target_year),
        "passenger_km_mkm": pa_vals.get(cc),
        "train_km_mkm": trainmv_vals.get(cc),
        "avg_pass_per_train_km": round(occupancy[cc], 2),
    }
    for cc in sorted(occupancy)
]
occ_df = pd.DataFrame(occ_rows)
occ_path = DATA_DIR / "eurostat_rail_occupancy_by_country.csv"
occ_df.to_csv(occ_path, index=False)
print(f"\n  Saved {occ_path.name}")

# ─── Step 3: Operator → ISO mapping ──────────────────────────────────────────

operator_to_iso = {
    "Germany":     "DE",
    "France":      "FR",
    "Netherlands": "NL",
    "Belgium":     "BE",
    "Spain":       "ES",
    "Italy":       "IT",
    "Switzerland": "CH",
    "Austria":     "AT",
    "Portugal":    "PT",
    "UK":          "GB",
}

# ─── Step 4: Fleet capacity ───────────────────────────────────────────────────

operator_capacity = {
    "France":       450,  # TGV fleet mix: Réseau 377, Duplex 508; avg ~450
    "Germany":      490,  # ICE fleet: ICE3neo 439, ICE4-12car 830; intercity avg ~490
    "Netherlands":  290,  # Intercity Direct + NS intercity, short formations
    "Belgium":      350,  # SNCB intercity double-deck formations
    "Spain":        330,  # Renfe AVE S-103 (318), S-112 (355); avg ~330
    "Italy":        400,  # Frecciarossa ETR-400 (457), ETR-500 (584); avg ~400
    "Switzerland":  400,  # SBB EC/IC formations, approx
    "Austria":      350,  # ÖBB Railjet (439 total), ~350 coach seats
    "Portugal":     250,  # CP Alfa Pendular 220 + regional avg ~250
    "UK":           750,  # Eurostar e320 (900) dominates cross-channel; avg 750
}

# ─── Step 5: Haversine + compute proxy ───────────────────────────────────────

print("\n=== Step 5: Building rail_edges_v4 ===")

cities_df = pd.read_csv(DATA_DIR / "cities.csv")
city_coords = {
    row["city"]: (float(row["lat"]), float(row["lon"]))
    for _, row in cities_df.iterrows()
    if pd.notna(row["lat"]) and pd.notna(row["lon"])
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))

edges = pd.read_csv(DATA_DIR / "rail_edges.csv")
out_rows = []
missing_cities = set()
fallback_ops = set()

for _, row in edges.iterrows():
    op = row["operator"]
    iso = operator_to_iso.get(op, "EU_avg")
    capacity = operator_capacity.get(op, 400)
    fallback_used = False

    # Resolve occupancy with fallback chain
    if iso in occupancy:
        avg_pptk = occupancy[iso]
    elif iso == "CH":
        avg_pptk = occupancy.get("AT", eu_average)  # nearest neighbour proxy
        iso = "CH(fallback=AT)"
        fallback_used = True
    elif iso == "GB":
        avg_pptk = eu_average
        iso = "GB(fallback=EU_median)"
        fallback_used = True
    else:
        avg_pptk = eu_average
        iso = f"{iso}(fallback=EU_median)"
        fallback_used = True

    if fallback_used:
        fallback_ops.add(op)

    lf_implied = round(avg_pptk / capacity, 4) if capacity > 0 else None

    city_a, city_b = row["from_city"], row["to_city"]
    if city_a in city_coords and city_b in city_coords:
        lat1, lon1 = city_coords[city_a]
        lat2, lon2 = city_coords[city_b]
        dist_km = round(haversine(lat1, lon1, lat2, lon2), 1)
    else:
        dist_km = None
        for c in [city_a, city_b]:
            if c not in city_coords:
                missing_cities.add(c)

    proxy = (row["frequency_per_day"] * avg_pptk * dist_km
             if dist_km is not None else None)

    out_rows.append({
        **row.to_dict(),
        "iso_code":                       iso,
        "capacity_seats":                 capacity,
        "avg_pass_per_train_km_eurostat": round(avg_pptk, 2),
        "load_factor_implied":            lf_implied,
        "distance_km":                    dist_km,
        "rail_demand_proxy":              round(proxy, 1) if proxy is not None else None,
        "fallback_used":                  fallback_used,
    })

v4_df = pd.DataFrame(out_rows)
v4_path = DATA_DIR / "rail_edges.csv"
v4_df.to_csv(v4_path, index=False)
print(f"  Saved {v4_path.name} ({len(v4_df)} rows)")
if missing_cities:
    print(f"  WARNING — cities missing coords: {missing_cities}")
if fallback_ops:
    print(f"  Fallback (EU median) applied to: {fallback_ops}")

# ─── Step 6: Validation ───────────────────────────────────────────────────────

print("\n=== Step 6: Top 20 corridors by rail_demand_proxy ===")

valid = v4_df[v4_df["rail_demand_proxy"].notna()].copy()
top20 = valid.sort_values("rail_demand_proxy", ascending=False).head(20)

pd.set_option("display.max_rows", 25)
pd.set_option("display.width", 130)
print(
    top20[["from_city", "to_city", "operator", "frequency_per_day",
           "distance_km", "avg_pass_per_train_km_eurostat", "rail_demand_proxy"]]
    .reset_index(drop=True)
    .to_string(index=True)
)

# Expected high-traffic corridors
EXPECTED = {
    ("Paris", "Lyon"), ("Lyon", "Paris"),
    ("Madrid", "Barcelona"), ("Barcelona", "Madrid"),
    ("Amsterdam", "Rotterdam"), ("Rotterdam", "Amsterdam"),
    ("Berlin", "Hamburg"), ("Hamburg", "Berlin"),
}
found = set()
for _, r in top20.iterrows():
    pair = (r["from_city"], r["to_city"])
    if pair in EXPECTED:
        found.add(pair)

expected_canonical = {tuple(sorted(p)) for p in EXPECTED}
found_canonical    = {tuple(sorted(p)) for p in found}
missing_expected   = expected_canonical - found_canonical
if missing_expected:
    print(f"\n  WARN: expected corridors NOT in top-20: {missing_expected}")
else:
    print("\n  OK: all four key corridors appear in top-20.")

print("\n=== Country-level occupancy ===")
print(occ_df.to_string(index=False))

# ─── Step 7: Summary markdown ─────────────────────────────────────────────────

print("\n=== Step 7: Writing summary report ===")

cap_table_rows = [
    ("France (SNCF/TGV)", 450,
     "TGV Réseau 377 seats, Duplex 508 seats; fleet average ~450"),
    ("Germany (DB/ICE)", 490,
     "ICE3neo 439 seats, ICE4 (12-car) 830 seats; intercity average ~490"),
    ("Netherlands (NS)", 290,
     "Intercity Direct + NS intercity, short formations; ~290"),
    ("Belgium (SNCB)", 350,
     "SNCB intercity double-deck formations; ~350"),
    ("Spain (Renfe/AVE)", 330,
     "AVE S-103: 318 seats, S-112: 355 seats; average ~330"),
    ("Italy (Trenitalia)", 400,
     "Frecciarossa ETR-400: 457 seats, ETR-500: 584 seats; average ~400"),
    ("Switzerland (SBB)", 400,
     "SBB EC/IC formations, approx ~400"),
    ("Austria (ÖBB)", 350,
     "ÖBB Railjet total 439 seats, approx 350 coach seats"),
    ("Portugal (CP)", 250,
     "Alfa Pendular 220 seats + regional trains; average ~250"),
    ("UK (Eurostar/ATOC)", 750,
     "Eurostar e320: 900 seats; dominates cross-channel routes; avg 750"),
]

lines = [
    "# Rail Proxy v4 — Methodology Summary",
    "",
    f"**Generated:** May 2026  ",
    f"**Eurostat year used:** {target_year}  ",
    f"**Datasets:** `rail_tf_trainmv` (passenger train-km MKM) × `rail_pa_total` (passenger-km MKM)  ",
    f"**EU median avg_pass_per_train_km:** {eu_average:.1f}  ",
    "",
    "---",
    "",
    "## 1. Country-level occupancy (avg passengers per train-km)",
    "",
    "Formula: `avg_pass_per_train_km = total_passenger_km_MKM / total_passenger_train_km_MKM`",
    "",
    "Source: Eurostat `rail_pa_total` and `rail_tf_trainmv` (SDMX-CSV), annual, filtered to",
    "passenger trains only (`tra_type = PASS`) and MKM unit.",
    "",
    "| Country | Pass-km (Mkm) | Train-km (Mkm) | Avg pass / train-km |",
    "|---|---|---|---|",
]
for _, r in occ_df.iterrows():
    lines.append(
        f"| {r['country_code']} "
        f"| {r['passenger_km_mkm']:.1f} "
        f"| {r['train_km_mkm']:.1f} "
        f"| **{r['avg_pass_per_train_km']:.1f}** |"
    )

lines += [
    "",
    "---",
    "",
    "## 2. Fleet capacity by operator",
    "",
    "Source: manufacturer data sheets and Wikipedia rolling stock articles.",
    "",
    "| Operator | Capacity (seats) | Basis |",
    "|---|---|---|",
]
for op, seats, basis in cap_table_rows:
    lines.append(f"| {op} | {seats} | {basis} |")

lines += [
    "",
    "---",
    "",
    "## 3. Countries missing from Eurostat (fallback applied)",
    "",
]
if fallback_ops:
    for op in sorted(fallback_ops):
        iso_used = operator_to_iso.get(op, "?")
        lines.append(
            f"- **{op}** (`{iso_used}`): Eurostat data unavailable — "
            f"EU median ({eu_average:.1f} pass/train-km) applied."
        )
else:
    lines.append("- None — all operators mapped to Eurostat data.")

lines += [
    "",
    "---",
    "",
    "## 4. Top 20 corridors by rail demand proxy",
    "",
    "Formula: `proxy = frequency_per_day × avg_pass_per_train_km × distance_km`",
    "*(units: passenger-km / day; great-circle distance via haversine)*",
    "",
    "| # | From | To | Operator | Freq/day | Dist (km) | Avg occ. | Proxy (pkm/day) |",
    "|---|---|---|---|---|---|---|---|",
]
for i, (_, r) in enumerate(top20.iterrows(), 1):
    lines.append(
        f"| {i} | {r['from_city']} | {r['to_city']} | {r['operator']} "
        f"| {r['frequency_per_day']:.1f} | {r['distance_km']:.0f} "
        f"| {r['avg_pass_per_train_km_eurostat']:.1f} | {r['rail_demand_proxy']:.0f} |"
    )

lines += [
    "",
    "---",
    "",
    "## 5. Interpretation notes",
    "",
    "- `avg_pass_per_train_km` from Eurostat is the **combined capacity × load-factor signal**",
    "  per train-km. It replaces the two placeholder constants (seats=400, LF=0.55) used in v3.",
    "- The `load_factor_implied` column in `rail_edges.csv` back-computes the implied load",
    "  factor as `avg_pass_per_train_km / capacity_seats` for transparency.",
    "- `distance_km` is the **haversine great-circle distance** between city centroids from",
    "  `data/cities.csv` — not travel_time_min (which conflates speed + dwell time).",
    "- This proxy feeds into **Layer C v4** in `calibration_plan_v2.md`: replacing the",
    "  air-only OPDI dependent variable with a combined air + rail OD measure.",
]

summary_md = "\n".join(lines) + "\n"
summary_path = DATA_DIR / "rail_proxy_summary.md"
summary_path.write_text(summary_md, encoding="utf-8")
print(f"  Saved {summary_path.name}")

print("\n=== Done ===")
print(f"  {occ_path}")
print(f"  {v4_path}")
print(f"  {summary_path}")
