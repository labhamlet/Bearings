#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=SELD-gram-table
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=seeds_table_ssl/slurm_output_%A_%a.out
#SBATCH --array=[31,33]

cd ~/embisonics_icassp/Embisonics/seld-dcase2023-gram
export HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-trainer

SEEDS=(3)

for SEED in "${SEEDS[@]}"; do
    echo ">>> task ${SLURM_ARRAY_TASK_ID}, seed ${SEED}"
    python3 train_seldnet.py "${SLURM_ARRAY_TASK_ID}" --seed "${SEED}"
done