# Compound Weather Disruption and Passenger Stranding in the European Rail–Air Network
### Agent-Based Simulation

## Overview
This project simulates how passengers redistribute across a 53-city European rail-air network when compound weather events disrupt multiple transport nodes at once (e.g. a storm closing both airports and rail lines in the same region). 

It combines a gravity-calibrated agent-based model with a Bass-style information diffusion layer to study *why* certain cities end up with stranded travelers — and whether simply being aware of a disruption is enough to avoid it.

## Key Results
- **Topology, not awareness, drives stranding.** When no path exists through the network, informing passengers earlier doesn't change the outcome — stranding is a structural property of the network, not an information problem.
- **Compound disruptions are qualitatively different from single-mode ones.** Disrupting rail or air alone produces little to no stranding; it's the *simultaneous* failure of both that creates real impact, supporting the interdependent-network-failure literature this model builds on.
- **Peripheral, single-mode-dependent cities are the most exposed** — even when they aren't the ones directly hit by the disruption, cities that rely on one fragile corridor absorb the most secondary stranding.
- **Validated against a real event** (Storm Ciaran, Nov 2023) as an external plausibility check, with simulated stranding patterns tracking real-world disruption severity.

---

## Requirements

Python 3.9+:
```
pip install numpy pandas matplotlib scipy networkx joblib scikit-learn jupyter statsmodels pyarrow
```

---

## Workflow

**The `data/` folder is fully included.** To run the model, go straight to Stage 3. Stages 1 and 2 document how those data files were built and are provided for full reproducibility - you do not need to re-run them.

---

### Stage 1 - Build the network data *(optional - outputs already in `data/`)*

Processes GTFS rail feeds and OPDI flight records into the edge lists used by the model.

```
python data_pipeline/process_gtfs.py        # GTFS/NeTEx feeds → data/rail_edges.csv
python data_pipeline/build_air_od.py        # OPDI parquets   → data/air_od.csv
python data_pipeline/build_rail_proxy.py    # Eurostat stats  → adds demand proxy to rail edges
python data_pipeline/build_combined_od.py   # merges air + rail → data/od_matrix_combined.csv
                                            #                     data/modal_split.csv
```

These scripts require the original source files, which are not included in this submission because of their size (several gigabytes). They can be downloaded from:
- GTFS feeds: Deutsche Bahn, SNCF, ÖBB, Renfe, SBB, Trenitalia, NS, SNCB, CP, Eurostar
- OPDI flight-list parquets: opdi.aero/flight-list-data (Jan 2022 – Mar 2026)
- Eurostat rail statistics: tran_hv_psmod

Full download URLs and feed versions for all sources are listed in the paper's Data and Calibration section (Section 4) footnotes.

---

### Stage 2 - Calibrate the demand model *(optional - outputs already in `data/`)*

Fits a dual-mode PPML gravity model (air + rail) to produce OD weights and route-level modal split.

```
jupyter notebook calibration_gravity.ipynb   # run all cells
```

Reads: `data/od_matrix_combined.csv`, `data/rail_edges.csv`
Writes: `data/gravity_weights.csv`, `data/modal_split.csv`, `data/gravity_params.json`, `data/gravity_diagnostic.png`

Then calibrate the disruption-response parameters (rerouting interval and rescheduling probability) against Storm Eunice face-validity:

```
python data_pipeline/extract_disruption_events.py   # extracts disruption signatures from OPDI
python data_pipeline/calibrate_layer_e.py           # grid search → data/layer_e_best_params.json
```

---

### Stage 3 - Run the simulation and view results

Runs three disruption scenarios (Northern Storm / Alpine Freeze / Atlantic Cascade) with 5 Monte Carlo replications each.

```
python rerun_simulation.py
```

Reads: all files in `data/`
Writes: `results.joblib`

Then open the results notebook to generate all figures, tables, and the Ciaran validation plot:

```
jupyter notebook Simulation.ipynb
```

Run all cells. The notebook generates every report figure inline and saves them to `figures/` (PDF + PNG).  
`results.joblib` is the only pre-computed input; everything else is produced inside the notebook.

---

### Stage 4 - Validate

Storm Ciaran validation is computed inside the notebook (Section 9).  
To run it as a standalone script:

```
python data_pipeline/compute_ciaran_spearman.py
```

Reads: `data/ciaran_disruption_signature.csv`, `data/layer_e_validation_ciaran.csv`
Writes: `figures/ciaran_validation_scatter.pdf`

---

### Stage 5 - Sensitivity analysis (HPC)

Sweeps spawn rate and disruption probability on the Bocconi HPC cluster.

```
# Upload scripts + data/ to HPC, then:
sbatch submit_sensitivity.sh
```

| Script | What it sweeps | Output |
|--------|---------------|--------|
| `sensitivity_spawn_hpc.py` | Spawn rate {100, 200, 500, 1000} → N_agents {209, 417, 1044, 2087} | `data/sensitivity_spawn_cities.csv` |
| `sensitivity_pdisrupt_hpc.py` | Disruption probability {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} | `data/sensitivity_pdisrupt_cities.csv` |

**All output files are already included in `data/`** — re-running this stage is not required.

The information-parameter sensitivity summary (`data/info_sensitivity_summary.csv`) sweeps $p_{\text{int}} \in \{0.10, 0.20, 0.30, 0.50\}$ and $q_{\text{contact}} \in \{0.05, 0.10, 0.20, 0.40\}$ across all three scenarios; it is generated by `sensitivity_info_params_hpc.py` + `merge_info_sensitivity.py` + `extract_info_sensitivity.py`.

---

## Core model files

| File | Role |
|------|------|
| `transport_network.py` | Builds the 53-city NetworkX graph from `data/` |
| `transport_info_abm.py` | ABM - passenger agents, routing, Bass-style awareness diffusion |
| `scenarios.py` | Disruption scenario definitions (nodes + edges closed per scenario) |
