"""
sensitivity_info_params_hpc.py
==============================
Single-task worker for SLURM array job.
Runs one (p_int, q_contact) combo across all 3 scenarios (5 MC runs each).

Task mapping (task_id = p_int_idx * 4 + q_contact_idx):
  p_int:     [0.10, 0.20, 0.30, 0.50]  (indices 0-3)
  q_contact: [0.05, 0.10, 0.20, 0.40]  (indices 0-3)
  16 tasks total (0-15).

Usage:
    python sensitivity_info_params_hpc.py --task_id 0

SLURM array:
    sbatch --array=0-15 submit_info_sensitivity.sh

Output: data/info_sens_task{NN}.joblib  (one file per task)
Merge:  python merge_info_sensitivity.py  (after all 16 tasks finish)
"""

import argparse
import os

import numpy as np
import pandas as pd
from joblib import dump

from transport_network import build_network
from transport_info_abm import TransportInfoModel as TransportModel
from scenarios import build_scenarios

N_STEPS       = 72
N_RUNS        = 5
SEED          = 42
START_HOUR    = 8
REROUTE_EVERY = 4
N_AGENTS      = 209
SPAWN_RATE    = 100

BASE_INFO_PARAMS = {
    "initial_info_prob":   0.05,
    "internet_info_prob":  0.00084,
    "internet_growth":     0.03,
    "internet_max_prob":   0.30,
    "contact_spread_prob": 0.20,
    "proactive_rerouting": True,
    "p_reschedule":        0.13,
}

P_INT_VALUES     = [0.10, 0.20, 0.30, 0.50]
Q_CONTACT_VALUES = [0.05, 0.10, 0.20, 0.40]


def run_one(G, scenario, seed, info_params):
    model = TransportModel(G, n_agents=N_AGENTS, seed=seed,
                           start_hour=START_HOUR, spawn_rate=SPAWN_RATE,
                           reroute_every=REROUTE_EVERY, info_params=info_params)
    for step in range(N_STEPS):
        if step == 24 and scenario is not None:
            model.apply_disruption(
                disrupted_nodes=scenario.get("disrupted_nodes", []),
                disrupted_edges=[(u, v, m) for u, v, m in scenario.get("disrupted_edges", [])],
            )
        model.run_step()
    history  = pd.DataFrame(model.history)
    stranded = model.get_stranded_counts()
    index    = model.get_stranding_index()
    return history, stranded, index


def run_mc(G, scenario, info_params):
    all_hist, all_stranded, all_index = [], {}, {}
    for i in range(N_RUNS):
        hist, stranded, index = run_one(G, scenario, SEED + i, info_params)
        all_hist.append(hist)
        for city, n in stranded.items():
            all_stranded[city] = all_stranded.get(city, []) + [n]
        for city, idx in index.items():
            all_index[city] = all_index.get(city, []) + [idx]
    combined   = pd.concat(all_hist)
    mean_df    = combined.groupby(combined.index).mean()
    avg_strand = {c: float(np.mean(v)) for c, v in all_stranded.items()}
    avg_index  = {c: float(np.mean(v)) for c, v in all_index.items()}
    return mean_df, avg_strand, avg_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    task_id       = args.task_id
    p_int         = P_INT_VALUES[task_id // 4]
    q_contact     = Q_CONTACT_VALUES[task_id % 4]

    print(f"Task {task_id}: p_int={p_int}, q_contact={q_contact}", flush=True)

    info_params = {**BASE_INFO_PARAMS,
                   "internet_max_prob":   p_int,
                   "contact_spread_prob": q_contact}

    print("Building network...", flush=True)
    G = build_network()
    SCENARIOS = build_scenarios(G)
    print(f"Network: {G.number_of_nodes()} cities, {G.number_of_edges()} edges", flush=True)

    task_results = {"p_int": p_int, "q_contact": q_contact, "scenarios": {}}

    for sc_key, scenario in SCENARIOS.items():
        print(f"  Scenario {sc_key}...", flush=True)
        mean_df, avg_strand, avg_index = run_mc(G, scenario, info_params)
        task_results["scenarios"][sc_key] = {
            "mean":     mean_df,
            "stranded": avg_strand,
            "index":    avg_index,
        }
        top1      = max(avg_strand, key=avg_strand.get)
        arr_inf   = mean_df["arrived_informed"].iloc[-1]
        total_str = sum(avg_strand.values())
        print(f"    Sc {sc_key}: top={top1}({avg_strand[top1]:.1f}), "
              f"arrived_inf={arr_inf:.0f}, total_strand={total_str:.0f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"info_sens_task{task_id:02d}.joblib")
    dump(task_results, out_path)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
