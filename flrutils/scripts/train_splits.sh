#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o train_splits_%j.log
#SBATCH --job-name=train_splits
#SBATCH --time=48:00:00

set -euo pipefail

source ~/fulianren/setup.sh || true
source .venv/bin/activate

# ── Config ─────────────────────────────────────────────────────────────────
BASE="Qwen/Qwen3-8B"
RUN_NAME="qasper_splits_train"
DATAPATH="data/splits"

# Train / val files (relative to DATAPATH)
TRAIN_FILE="train/qasper_train_t0.25.xml"
VAL_FILE="eval/qasper_eval_t0.25.xml"

# ── Train ──────────────────────────────────────────────────────────────────
echo "=== Training on splits ==="
echo "  Train: $DATAPATH/$TRAIN_FILE"
echo "  Val:   $DATAPATH/$VAL_FILE"
echo "  Base:  $BASE"

accelerate launch --num_processes 8 training/run_train_student.py \
    --base "$BASE" \
    --run_name "$RUN_NAME" \
    --datapath "$DATAPATH" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --lora_r 1024 \
    --batch_size 8 \
    --deepspeed_path config/ds2.json \
    --micro_batch_size 1 \
    --n_epochs 10 \
    --use_wandb True \
    --validate True \
    --save_during_training True \
    --save_interval 100 \
    --checkpoint_interval 100 \
    --teacher student_base \
    --logit_loss_weight 1.0 \
    --token_loss_weight 0.0

echo "=== Training complete ==="
