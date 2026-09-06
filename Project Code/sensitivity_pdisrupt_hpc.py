"""
sensitivity_pdisrupt_hpc.py
============================
Sensitivity analysis: sweeps disruption_prob in {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.
Fixed spawn_rate=100, n_agents=209 (identical to main run).

Uses TransportInfoModel with identical INFO_PARAMS to rerun_simulation.py.
Disruption applied at step 24 (same as main run).

Usage
-----
    python sensitivity_pdisrupt_hpc.py                    # all defaults
    python sensitivity_pdisrupt_hpc.py --n_runs 10
    python sensitivity_pdisrupt_hpc.py --p_values 0.8 0.9 1.0
    python sensitivity_pdisrupt_hpc.py --out_dir results/

SLURM one-liner:
    sbatch --time=02:00:00 --mem=8G --wrap="python sensitivity_pdisrupt_hpc.py --n_runs 10"

Outputs (in --out_dir, default: data/)
-------
    sensitivity_pdisrupt_cities.csv  — avg stranded + stranding index per
                                       (scenario, p_disrupt, city)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from transport_network import build_network
from transport_info_abm import TransportInfoModel
from scenarios import build_scenarios

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_N_AGENTS    = 209
DEFAULT_SPAWN_RATE  = 100
DEFAULT_P_VALUES    = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_N_RUNS      = 10
DEFAULT_N_STEPS     = 72
DEFAULT_START_HOUR  = 8
DEFAULT_REROUTE_EVERY = 4
DEFAULT_OUT_DIR     = "data"

INFO_PARAMS = {
    "initial_info_prob":   0.05,
    "internet_info_prob":  0.00084,
    "internet_growth":     0.03,
    "internet_max_prob":   0.30,
    "contact_spread_prob": 0.20,
    "proactive_rerouting": True,
    "p_reschedule":        0.13,
}


def run_one(G, scenario, p_disrupt, seed, n_steps, start_hour, reroute_every):
    model = TransportInfoModel(
        G, n_agents=DEFAULT_N_AGENTS, seed=seed,
        start_hour=start_hour, spawn_rate=DEFAULT_SPAWN_RATE,
        reroute_every=reroute_every, info_params=INFO_PARAMS,
    )
    for step in range(n_steps):
        if step == 24 and scenario is not None:
            model.apply_disruption(
                disrupted_nodes=scenario.get("disrupted_nodes", []),
                disrupted_edges=[(u, v, m) for u, v, m in scenario.get("disrupted_edges", [])],
                disruption_prob=p_disrupt,
            )
        model.run_step()
    return model.get_stranded_counts(), model.get_stranding_index()


def run_mc(G, scenario, p_disrupt, n_runs, n_steps, start_hour, reroute_every):
    all_stranded, all_index = {}, {}
    for i in range(n_runs):
        stranded, index = run_one(
            G, scenario, p_disrupt=p_disrupt, seed=i,
            n_steps=n_steps, start_hour=start_hour, reroute_every=reroute_every,
        )
        for city, n in stranded.items():
            all_stranded[city] = all_stranded.get(city, []) + [n]
        for city, idx in index.items():
            all_index[city] = all_index.get(city, []) + [idx]
    avg_stranded = {c: float(np.mean(v)) for c, v in all_stranded.items()}
    avg_index    = {c: float(np.mean(v)) for c, v in all_index.items()}
    return avg_stranded, avg_index


def main():
    parser = argparse.ArgumentParser(
        description="Disruption-probability sensitivity for transport ABM")
    parser.add_argument("--p_values",      nargs="+", type=float,
                        default=DEFAULT_P_VALUES)
    parser.add_argument("--n_runs",        type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--n_steps",       type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--start_hour",    type=int, default=DEFAULT_START_HOUR)
    parser.add_argument("--reroute_every", type=int, default=DEFAULT_REROUTE_EVERY)
    parser.add_argument("--scenarios",     nargs="+", default=None)
    parser.add_argument("--out_dir",       type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "sensitivity_pdisrupt_cities.csv")

    # ── Resume: skip already-completed (scenario, p_disrupt) pairs ───────────
    completed = set()
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            for _, row in existing[["scenario", "p_disrupt"]].drop_duplicates().iterrows():
                completed.add((str(row["scenario"]), round(float(row["p_disrupt"]), 6)))
            print(f"Checkpoint: {len(completed)} pair(s) already done: {sorted(completed)}")
        except Exception as e:
            print(f"Could not read checkpoint ({e}) — starting fresh.")

    print("Building network...")
    G = build_network()
    SCENARIOS = build_scenarios(G)
    if args.scenarios is None:
        args.scenarios = list(SCENARIOS.keys())
    print(f"Network: {G.number_of_nodes()} cities, {G.number_of_edges()} edges")

    total = len(args.p_values) * len(args.scenarios)
    done  = 0
    first_write = len(completed) == 0

    for p in args.p_values:
        p_key = round(p, 6)
        for scen_key in args.scenarios:
            done += 1
            if (scen_key, p_key) in completed:
                print(f"[{done}/{total}] SKIP  scenario={scen_key}  "
                      f"p_disrupt={p}  (checkpoint)", flush=True)
                continue

            scenario = SCENARIOS[scen_key]
            t0 = time.time()
            print(f"[{done}/{total}] scenario={scen_key}  p_disrupt={p}"
                  f"  n_runs={args.n_runs} ...", flush=True)

            avg_stranded, avg_index = run_mc(
                G, scenario=scenario, p_disrupt=p,
                n_runs=args.n_runs, n_steps=args.n_steps,
                start_hour=args.start_hour, reroute_every=args.reroute_every,
            )

            all_cities = set(avg_stranded) | set(avg_index)
            rows = [
                {
                    "scenario":     scen_key,
                    "p_disrupt":    p,
                    "city":         city,
                    "avg_stranded": avg_stranded.get(city, 0.0),
                    "avg_index":    avg_index.get(city, 0.0),
                }
                for city in all_cities
            ]

            mode   = 'w' if first_write else 'a'
            header = first_write
            pd.DataFrame(rows).to_csv(out_path, mode=mode, header=header, index=False)
            first_write = False

            elapsed   = time.time() - t0
            total_str = sum(avg_stranded.values())
            print(f"    done in {elapsed:.1f}s  total_stranded={total_str:.1f}"
                  f"  [saved]", flush=True)

    print(f"\nDone. Results in: {out_path}")


if __name__ == "__main__":
    main()
