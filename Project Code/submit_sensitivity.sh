#!/bin/bash

#SBATCH --job-name="transport_sensitivity"
#SBATCH --account=[HPC_ACCOUNT]
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-user=[HPC_ACCOUNT]
#SBATCH --mail-type=ALL

# ── Environment ───────────────────────────────────────────────────────────────
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate simulation

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p out err data

echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo ""

# ── Run 1: spawn-rate sensitivity ─────────────────────────────────────────────
echo "START spawn-rate sensitivity: $(date)"

python sensitivity_spawn_hpc.py \
    --spawn_rates 100 200 500 1000 \
    --n_runs 10 \
    --out_dir data

echo "DONE spawn-rate sensitivity: $(date)"
echo ""

# ── Run 2: disruption-probability sensitivity ─────────────────────────────────
echo "START p_disrupt sensitivity: $(date)"

python sensitivity_pdisrupt_hpc.py \
    --p_values 0.5 0.6 0.7 0.8 0.9 1.0 \
    --n_runs 10 \
    --out_dir data

echo "DONE p_disrupt sensitivity: $(date)"

echo ""
echo "Job finished: $(date)"

# ── Cleanup ───────────────────────────────────────────────────────────────────
conda deactivate
module unload /software/modules/miniconda3
