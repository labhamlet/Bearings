#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=EmbiAblR
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0-1
#SBATCH --constraint=scratch-node

cd ~/embisonics_icassp/Embisonics

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-trainer

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_ARENA_MAX=2

rclone copy /projects/0/prjs1261/visage/audios/audio_wds $TMPDIR/ \
    --include "*.tar" --transfers $(nproc) --checkers $(nproc) -L

ABLATIONS=(
    "loss.diffuseness=0.0"
    "loss.q=0.0"
    "route_a.n_grid=512 route_a.vmf_kappa=81.49"
)
# arm names as derived by get_identity_from_cfg, same order
ARMS=("no-psi" "no-q" "grid512" )

OVERRIDES=${ABLATIONS[$SLURM_ARRAY_TASK_ID]}
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

# ---- locate this arm's step-30000 checkpoint ----------------------------
SAVE_ROOT=/projects/0/prjs1261/experiments/sphere
RESUME_STEP=30000
CKPT=$(find "$SAVE_ROOT/Abl=$ARM" -name "step=${RESUME_STEP}*.ckpt" \
       -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$CKPT" ]; then
    echo "ERROR: no step=${RESUME_STEP}*.ckpt under $SAVE_ROOT/Abl=$ARM" >&2
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID (arm=$ARM) resuming from: $CKPT"

python3 train.py \
    $OVERRIDES \
    "model.pretrained_ckpt='$CKPT'" \
    "data.glob='${TMPDIR}/shard-{000000..000048}.tar'"