#!/bin/bash

#SBATCH --job-name="info_sens_seq"
#SBATCH --account=[HPC_ACCOUNT]
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=out/info_sens_seq_%j.out
#SBATCH --error=err/info_sens_seq_%j.err
#SBATCH --mail-user=[HPC_EMAIL]
#SBATCH --mail-type=END,FAIL

# ── Environment ───────────────────────────────────────────────────────────────
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate simulation

mkdir -p out err data

echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo ""

# ── Run 16 (p_int, q_contact) combos sequentially ────────────────────────────
# Each combo saves data/info_sens_task{NN}.joblib immediately on completion.
# Already-saved files are skipped automatically — safe to resubmit if killed.

for TASK_ID in $(seq 0 15); do
    OUT_FILE="data/info_sens_task$(printf '%02d' ${TASK_ID}).joblib"
    if [ -f "${OUT_FILE}" ]; then
        echo "[${TASK_ID}/15] SKIP — ${OUT_FILE} already exists"
        continue
    fi
    echo "[${TASK_ID}/15] $(date) — starting task ${TASK_ID}"
    python sensitivity_info_params_hpc.py --task_id ${TASK_ID} --out_dir data
    echo "[${TASK_ID}/15] $(date) — done"
    echo ""
done

# ── Merge all 16 task files into results_info_sensitivity.joblib ──────────────
echo "Merging results: $(date)"
python merge_info_sensitivity.py --task_dir data --out results_info_sensitivity.joblib
echo "Merge done: $(date)"

echo ""
echo "Job finished: $(date)"

conda deactivate
module unload /software/modules/miniconda3
