#!/usr/bin/env python3
"""
calibrate_layer_e.py
====================
Calibrate Layer E (disruption-response) parameters against Storm Eunice
(Feb 18–19, 2022), then validate on Dec 2022 Heathrow snow event.

Parameters calibrated
---------------------
  reroute_every   — how many steps between proactive rerouting checks (1–12 h)
  p_reschedule    — probability informed agent cancels trip (0.05–0.25)

Calibration objective
---------------------
Clean two-part design — avoids circularity between calibration input and target:

  Part A — disruption nodes (INPUT to ABM)
      Cities with OPDI disruption_score ≥ 0.25 are treated as disrupted nodes.
      (Positive score = service LOST during event.)

  Part B — overflow correlation (CALIBRATION TARGET)
      Cities with OPDI disruption_score < −0.05 received MORE flights than
      baseline during the event — rerouting overflow hubs.
      The ABM should predict elevated agent-in-transit counts at those same cities.
      Objective: maximise Spearman ρ between OPDI overflow magnitude and
                 ABM in-transit accumulation at non-disrupted cities.

These two parts use different cities (Part A = heavily disrupted, Part B = overflow
absorbers), so the objective is genuinely independent of the disruption definition.

Validation
----------
After selecting best parameters from Eunice calibration, run the Dec 2022 event
with those parameters.  Check:
  1. London is top-stranded city (known from media reports)
  2. Overflow Spearman ρ on Dec 2022 is positive (> 0.0)

Outputs
-------
  data/layer_e_calibration_grid.csv   — full grid search results
  data/layer_e_best_params.json       — best parameters to wire into notebook
  data/layer_e_validation_dec2022.csv — held-out validation results

Usage
-----
  python data_pipeline/calibrate_layer_e.py

Runtime: ~10–20 min on laptop (5×5 grid × 2 events × 3 MC runs each).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path so we can import model modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from transport_network import build_network
from transport_info_abm import TransportInfoModel
from scenarios import build_scenarios

# ── Grid search parameters ────────────────────────────────────────────────────
# reroute_every fixed at literature-anchored value (Prague airport closure:
# modal switch within 2–6hr; Haneda mass-cancellation: rebooking within 4–8hr).
# Full-node-closure calibration confirms it has no leverage — topology dominates.
REROUTE_EVERY       = 4
P_RESCHEDULE_VALUES = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25]

# Fixed parameters (not being calibrated here)
N_AGENTS   = 500
N_STEPS    = 72
N_RUNS_CAL = 3     # fewer MC runs for speed during grid search
N_RUNS_VAL = 5     # more runs for final validation
START_HOUR = 8

# INFO_PARAMS matching the notebook (all except p_reschedule which we vary)
BASE_INFO_PARAMS = {
    "initial_info_prob":   0.25,
    "internet_info_prob":  0.00084,
    "internet_growth":     0.03,
    "internet_max_prob":   0.30,
    "contact_spread_prob": 0.0211,
    "proactive_rerouting": True,
}

# ── Event definitions (must match extract_disruption_events.py) ───────────────
CALIB_EVENT = "eunice"
VALID_EVENT  = "dec2022"

DISRUPTION_THRESHOLD = 0.25   # score ≥ this → disrupted node (used only for logging)
OVERFLOW_THRESHOLD   = -0.05  # score < this → overflow hub (more flights)

# Hardcoded disrupted nodes per event — derived from contemporaneous news/industry
# reports, NOT from OPDI threshold. OPDI flight_list aggregates actual operations
# across the full day including recovery flights, so the cancellation signal is
# attenuated at large hub airports. We use OPDI only for the overflow target.
EVENT_DISRUPTED_NODES = {
    "eunice":   ["London", "Amsterdam", "Hamburg", "Brussels"],
    "dec2022":  ["London"],
    "heatwave": ["London"],
}



# ── Helpers ───────────────────────────────────────────────────────────────────

def load_signature(event_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{event_key}_disruption_signature.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print(f"  Run extract_disruption_events.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def get_disrupted_nodes(sig: pd.DataFrame) -> list:
    """Cities with disruption_score >= threshold → ABM disrupted nodes."""
    return sig[sig["disruption_score"] >= DISRUPTION_THRESHOLD]["city"].tolist()


def get_overflow_targets(sig: pd.DataFrame, network_cities: list) -> dict:
    """
    Cities with disruption_score < overflow_threshold, NOT in disrupted set,
    AND present in the ABM network.  Cities outside the 53-node network
    (e.g. Nicosia, Riga, Belgrade, Tenerife) appear in OPDI but are not ABM
    nodes — including them makes the Spearman intersection empty.
    """
    network_set = set(network_cities)
    overflow = sig[
        (sig["disruption_score"] < OVERFLOW_THRESHOLD) &
        (sig["disruption_score"] >= -1.0)
    ]
    result = {}
    for _, row in overflow.iterrows():
        if row["city"] in network_set and row["city"] not in get_disrupted_nodes(sig):
            result[row["city"]] = abs(row["disruption_score"])
    return result


def run_single(G, disrupted_nodes, reroute_every, p_reschedule, seed):
    """
    Run one simulation with given disruption and parameters.
    Returns:
      in_transit — {city: count} snapshot at TRANSIT_SNAPSHOT_STEP (step 24)
                   when rerouting demand peaks; much richer than a step-72 tail
      stranded   — {city: count} at end of simulation (step 72)
    """
    info_params = {
        **BASE_INFO_PARAMS,
        "p_reschedule": p_reschedule,
    }
    model = TransportInfoModel(
        G,
        n_agents=N_AGENTS,
        seed=seed,
        start_hour=START_HOUR,
        info_params=info_params,
        reroute_every=reroute_every,
    )
    model.apply_disruption(disrupted_nodes=disrupted_nodes, disrupted_edges=[])

    # Cumulative in-transit: sum agent-steps spent at each city while traveling.
    # One hop = one step, so a snapshot captures near-zero agents per city.
    # Cumulative across all 72 steps is the correct measure of rerouting burden.
    in_transit = {}
    for _ in range(N_STEPS):
        model.run_step()
        for agent in model.agents:
            city = getattr(agent, "current_node", None)
            if city and agent.status == "traveling":
                in_transit[city] = in_transit.get(city, 0) + 1

    stranded = {}
    for agent in model.agents:
        city = getattr(agent, "current_node", None)
        if city and agent.status == "stranded":
            stranded[city] = stranded.get(city, 0) + 1

    return in_transit, stranded


def run_mc_summary(G, disrupted_nodes, reroute_every, p_reschedule, n_runs):
    """Run n_runs Monte Carlo simulations and return averaged city-level results."""
    all_transit  = {}
    all_stranded = {}

    for seed in range(n_runs):
        transit, strand = run_single(G, disrupted_nodes, reroute_every, p_reschedule, seed)
        for city, n in transit.items():
            all_transit[city]  = all_transit.get(city, []) + [n]
        for city, n in strand.items():
            all_stranded[city] = all_stranded.get(city, []) + [n]

    avg_transit  = {c: np.mean(v) for c, v in all_transit.items()}
    avg_stranded = {c: np.mean(v) for c, v in all_stranded.items()}
    return avg_transit, avg_stranded


def overflow_spearman(avg_transit: dict, overflow_targets: dict) -> float:
    """
    Compute Spearman ρ between:
      x = OPDI overflow magnitude (|score|) for overflow cities
      y = ABM in-transit agent count at those same cities
    Only cities present in both dicts are used.
    """
    cities = sorted(set(overflow_targets.keys()) & set(avg_transit.keys()))
    if len(cities) < 3:
        return np.nan   # not enough data points

    x = [overflow_targets[c] for c in cities]
    y = [avg_transit[c]      for c in cities]
    rho, _ = spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else np.nan


def compute_stranding_index(avg_stranded: dict, G) -> dict:
    """
    Stranding index = stranded_at_city / agents_involving_city.
    Uses city degree in G as a proxy for how many agents pass through it.
    (A proper version would use model.agents, but this is sufficient for ranking.)
    """
    total_stranded = sum(avg_stranded.values()) or 1
    degree = dict(G.degree())
    max_deg = max(degree.values()) or 1
    index = {}
    for city in avg_stranded:
        weight = degree.get(city, 1) / max_deg
        index[city] = avg_stranded[city] / (total_stranded * weight + 1e-6)
    return index


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search(G, sig, event_key, n_runs):
    disrupted_nodes  = EVENT_DISRUPTED_NODES.get(event_key, get_disrupted_nodes(sig))
    overflow_targets = get_overflow_targets(sig, list(G.nodes()))

    print(f"\n  Disrupted nodes ({len(disrupted_nodes)}): {disrupted_nodes}")
    print(f"  Overflow targets ({len(overflow_targets)}): "
          f"{list(overflow_targets.keys())}")
    print(f"\n  reroute_every = {REROUTE_EVERY} (fixed, literature-anchored)")
    print(f"  p_reschedule grid: {P_RESCHEDULE_VALUES}")
    print(f"  MC runs per cell: {n_runs}   (total cells: {len(P_RESCHEDULE_VALUES)})\n")

    rows = []
    for p_reschedule in P_RESCHEDULE_VALUES:
        tag = f"pr={p_reschedule:.2f}"
        print(f"  [{tag}] running ...", end="", flush=True)

        avg_transit, avg_stranded = run_mc_summary(
            G, disrupted_nodes, REROUTE_EVERY, p_reschedule, n_runs
        )

        rho = overflow_spearman(avg_transit, overflow_targets)
        top_stranded = sorted(avg_stranded.items(), key=lambda x: -x[1])[:5]
        top_stranded_str = ", ".join(f"{c}({n:.1f})" for c, n in top_stranded)
        total_strand = sum(avg_stranded.values())

        print(f" rho={rho:.3f}  total_stranded={total_strand:.1f}  "
              f"top5=[{top_stranded_str}]")

        rows.append({
            "reroute_every":         REROUTE_EVERY,
            "p_reschedule":          p_reschedule,
            "overflow_spearman_rho": round(rho, 4),
            "total_stranded":        round(total_strand, 2),
            "top5_stranded":         top_stranded_str,
            "disrupted_nodes":       str(disrupted_nodes),
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Layer E Calibration (Storm Eunice / Dec 2022) ===")
    print(f"Project root: {PROJECT_ROOT}")

    # ── Build network once ─────────────────────────────────────────────────────
    print("\n[Building transport network ...]")
    G = build_network()
    all_cities = list(G.nodes())
    print(f"  Network: {len(all_cities)} cities, {G.number_of_edges()} edges")

    # ── Load disruption signatures ─────────────────────────────────────────────
    print(f"\n[Loading disruption signatures ...]")
    calib_sig = load_signature(CALIB_EVENT)
    valid_sig  = load_signature(VALID_EVENT)
    print(f"  Calibration event ({CALIB_EVENT}): {len(calib_sig)} cities")
    print(f"  Validation event  ({VALID_EVENT}):  {len(valid_sig)} cities")

    print(f"\n  Top 10 cities by Eunice disruption score:")
    print(calib_sig[["city","disruption_score","cancelled_pct","severely_disrupted"]]
          .head(10).to_string(index=False))

    # ── Grid search on calibration event ──────────────────────────────────────
    print(f"\n[Grid search on {CALIB_EVENT} ...]")
    grid_df = run_grid_search(G, calib_sig, CALIB_EVENT, N_RUNS_CAL)

    grid_out = os.path.join(DATA_DIR, "layer_e_calibration_grid.csv")
    grid_df.to_csv(grid_out, index=False)
    print(f"\n  Full grid saved → {grid_out}")

    # ── Select best parameters ─────────────────────────────────────────────────
    valid_rows = grid_df.dropna(subset=["overflow_spearman_rho"])
    if valid_rows.empty:
        print("\nWARNING: all grid cells returned NaN rho — "
              "check overflow targets. Falling back to literature value.")
        best_p_reschedule = 0.13
    else:
        best_row = valid_rows.loc[valid_rows["overflow_spearman_rho"].idxmax()]
        best_p_reschedule = float(best_row["p_reschedule"])

    print(f"\n  *** Best parameters ***")
    print(f"      reroute_every = {REROUTE_EVERY}  (literature-anchored, not calibrated)")
    print(f"      p_reschedule  = {best_p_reschedule:.2f}")
    print(f"      overflow ρ    = "
          f"{valid_rows['overflow_spearman_rho'].max():.4f}")

    best_params = {
        "reroute_every":               REROUTE_EVERY,
        "p_reschedule":                best_p_reschedule,
        "overflow_spearman_rho_calib": float(valid_rows["overflow_spearman_rho"].max()),
        "calibration_event":           CALIB_EVENT,
        "n_runs_calibration":          N_RUNS_CAL,
        "note": (
            "reroute_every fixed at 4 (Prague/Haneda literature). "
            "p_reschedule calibrated via overflow Spearman ρ between OPDI "
            "flight-increase signal at non-disrupted cities and ABM in-transit "
            "agent accumulation. Validated on Dec 2022 Heathrow snow event (held-out)."
        )
    }

    params_out = os.path.join(DATA_DIR, "layer_e_best_params.json")
    with open(params_out, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best params saved → {params_out}")

    # ── Held-out validation on Dec 2022 ───────────────────────────────────────
    print(f"\n[Held-out validation on {VALID_EVENT} ...]")
    val_disrupted  = EVENT_DISRUPTED_NODES.get(VALID_EVENT, get_disrupted_nodes(valid_sig))
    val_overflow   = get_overflow_targets(valid_sig, all_cities)

    print(f"  Disrupted nodes: {val_disrupted}")
    print(f"  Overflow targets: {list(val_overflow.keys())}")

    val_transit, val_stranded = run_mc_summary(
        G, val_disrupted, REROUTE_EVERY, best_p_reschedule, N_RUNS_VAL
    )
    val_rho = overflow_spearman(val_transit, val_overflow)

    # Check top-stranded city
    top5_val = sorted(val_stranded.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Validation overflow ρ: {val_rho:.4f}")
    print(f"  Top 5 stranded (Dec 2022 run with best params):")
    for city, n in top5_val:
        print(f"    {city:<15} {n:.1f} agents")

    face_valid_pass = top5_val[0][0] == "London" if top5_val else False
    print(f"\n  Face-validity check (London = top stranded): "
          f"{'PASS' if face_valid_pass else 'FAIL — check disruption signature'}")

    # Save validation results
    val_rows = []
    all_val_cities = sorted(
        set(val_stranded.keys()) | set(val_transit.keys()) | set(val_overflow.keys())
    )
    for city in all_val_cities:
        val_rows.append({
            "city":             city,
            "abm_stranded":     round(val_stranded.get(city, 0), 2),
            "abm_in_transit":   round(val_transit.get(city, 0), 2),
            "opdi_overflow_mag": round(val_overflow.get(city, 0), 4),
            "in_overflow_set":  city in val_overflow,
        })
    val_df = pd.DataFrame(val_rows).sort_values("abm_stranded", ascending=False)

    val_out = os.path.join(DATA_DIR, "layer_e_validation_dec2022.csv")
    val_df.to_csv(val_out, index=False)
    print(f"  Validation results saved → {val_out}")

    best_params["overflow_spearman_rho_validation"] = float(val_rho)
    best_params["face_validity_london_top"] = face_valid_pass
    with open(params_out, "w") as f:
        json.dump(best_params, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE — wire these into the notebook INFO_PARAMS:")
    print(f"  reroute_every = {REROUTE_EVERY}  (literature-anchored: Prague/Haneda)")
    print(f"  p_reschedule  = {best_p_reschedule:.2f}  (calibrated from Eunice overflow ρ)")
    print(f"\nCalibration Spearman ρ (Eunice overflow):   "
          f"{valid_rows['overflow_spearman_rho'].max():.4f}")
    print(f"Validation  Spearman ρ (Dec 2022 overflow): {val_rho:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
