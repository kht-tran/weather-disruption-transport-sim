"""
rerun_simulation.py
===================
Re-runs the transport ABM with inform-pathway tracking added to get_model_status().
Saves results to results.joblib.
"""

import numpy as np
import pandas as pd
from joblib import dump

from transport_network import build_network
from transport_info_abm import TransportInfoModel as TransportModel
from scenarios import build_scenarios

N_AGENTS      = 209   # Little's Law: round(SPAWN_RATE × T_avg) = round(100 × 2.087)
N_STEPS       = 72
N_RUNS        = 5
SEED          = 42
START_HOUR    = 8
REROUTE_EVERY = 4
SPAWN_RATE    = 100

INFO_PARAMS = {
    "initial_info_prob":  0.05,   # plausible range 0.03-0.15 for sudden weather event
    "internet_info_prob": 0.00084,
    "internet_growth":    0.03,
    "internet_max_prob":  0.30,
    "contact_spread_prob": 0.20,  # q_contact = 0.20 h⁻¹ (Bass imitation, transport captive-audience scale)
    "proactive_rerouting": True,
    "p_reschedule":        0.13,
}

print("Building network...")
G = build_network()
SCENARIOS = build_scenarios(G)
print(f"Network: {G.number_of_nodes()} cities, {G.number_of_edges()} edges")


def run_one(G, scenario, seed):
    model = TransportModel(G, n_agents=N_AGENTS, seed=seed,
                           start_hour=START_HOUR, spawn_rate=SPAWN_RATE,
                           reroute_every=REROUTE_EVERY, info_params=INFO_PARAMS)
    for step in range(N_STEPS):
        if step == 24 and scenario is not None:
            model.apply_disruption(
                disrupted_nodes=scenario.get("disrupted_nodes", []),
                disrupted_edges=[(u, v, m) for u, v, m in scenario.get("disrupted_edges", [])],
            )
        model.run_step()
    history = pd.DataFrame(model.history)
    stranded = model.get_stranded_counts()
    index    = model.get_stranding_index()
    return history, stranded, index


def run_mc(G, scenario):
    all_hist, all_stranded, all_index = [], {}, {}
    for i in range(N_RUNS):
        print(f"    run {i+1}/{N_RUNS}...", flush=True)
        hist, stranded, index = run_one(G, scenario, seed=SEED + i)
        all_hist.append(hist)
        for city, n in stranded.items():
            all_stranded[city] = all_stranded.get(city, []) + [n]
        for city, idx in index.items():
            all_index[city] = all_index.get(city, []) + [idx]
    combined  = pd.concat(all_hist)
    mean_df   = combined.groupby(combined.index).mean()
    std_df    = combined.groupby(combined.index).std()
    avg_strand = {c: float(np.mean(v)) for c, v in all_stranded.items()}
    avg_index  = {c: float(np.mean(v)) for c, v in all_index.items()}
    return mean_df, std_df, avg_strand, avg_index


results = {}
for key, scenario in SCENARIOS.items():
    print(f"\nScenario {key}: {scenario['name']}")
    mean_df, std_df, avg_strand, avg_index = run_mc(G, scenario)
    results[key] = {
        "mean":    mean_df,
        "std":     std_df,
        "stranded": avg_strand,
        "index":    avg_index,
        **scenario,
    }
    top3 = dict(list(sorted(avg_strand.items(), key=lambda x: -x[1]))[:3])
    print(f"  Top stranded: {top3}")

dump(results, "results.joblib")
print("\nSaved results.joblib")

# Quick sanity check: print inform-pathway totals at final step
print("\nInform-pathway breakdown at step 72 (mean across runs):")
for key in ["A", "B", "C"]:
    row = results[key]["mean"].iloc[-1]
    total_inf = row["informed"]
    for col in ["inform_initial", "inform_internet", "inform_contact",
                "inform_blocked", "inform_stranded"]:
        pct = 100 * row[col] / total_inf if total_inf > 0 else 0
        print(f"  [{key}] {col:20s}: {row[col]:6.1f}  ({pct:.1f}%)")
