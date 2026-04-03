#!/bin/bash
#SBATCH -p j01
#SBATCH -o json_to_lesson_%j.log
#SBATCH --job-name=json_to_lesson
#SBATCH --time=01:00:00

set -euo pipefail

source ~/fulianren/setup.sh || true
source .venv/bin/activate

# ── Config ─────────────────────────────────────────────────────────────────
SPLITS_DIR="flrutils/splits"
OUTPUT_DIR="data/splits"
mkdir -p "$OUTPUT_DIR"/{train,test,eval}

# ── Convert JSONs to lesson XMLs ───────────────────────────────────────────
echo "=== Converting train split ==="
python curriculum/json_to_lesson.py \
    --input "$SPLITS_DIR/train/qasper__train_10_512_questions_20_train.json" \
    --output "$OUTPUT_DIR/train/qasper_train_t0.25.xml"

echo "=== Converting test split ==="
python curriculum/json_to_lesson.py \
    --input "$SPLITS_DIR/test/qasper__train_10_512_questions_20_test.json" \
    --output "$OUTPUT_DIR/test/qasper_test_t0.25.xml"

echo "=== Converting eval split ==="
python curriculum/json_to_lesson.py \
    --input "$SPLITS_DIR/eval/qasper__train_10_gt.json" \
    --output "$OUTPUT_DIR/eval/qasper_eval_t0.25.xml"

echo "=== All conversions complete ==="
ls -lh "$OUTPUT_DIR"/{train,test,eval}/
