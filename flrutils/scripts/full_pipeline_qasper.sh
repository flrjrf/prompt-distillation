#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o full_pipeline_qasper_%j.log
#SBATCH --job-name=full_pipeline
#SBATCH --time=48:00:00

source ~/fulianren/setup.sh || true
source .venv/bin/activate

DATASET_FAMILY="qasper"
DATASET="train"
BASE="Qwen/Qwen3-8B"
MODEL_SHORT="Qwen3-8B"
MAX_ITEMS_TRAIN=10
MAX_ITEMS_TEST=10
TRAIN_QUESTIONS=250
QUESTION_TEMP=1.5
LESSON_TEMP=1.5
EXAM_TEMP=0.25
VARIANT="default"

# ── Step 1: Launch vLLM server (used for steps 2, 5, 6, 7) ──────────────────
echo "=== Step 1: Launching vLLM server ==="
python -u -m vllm.entrypoints.openai.api_server \
    --model "$BASE" \
    --tensor-parallel-size 8 &
VLLM_PID=$!

echo "Waiting for vLLM server..."
until curl -sf http://localhost:8000/health > /dev/null; do
    sleep 5
done
echo "vLLM server is ready."

# # ── Step 2: Generate questions (uses API server) ────────────────────────────
# echo "=== Step 2: Generating questions ==="
# python evaluation/sample_questions.py \
#     --base "$MODEL_SHORT" \
#     --dataset_family "$DATASET_FAMILY" \
#     --dataset "$DATASET" \
#     --max_items "$MAX_ITEMS_TRAIN" \
#     --train_questions "$TRAIN_QUESTIONS" \
#     --temperature "$QUESTION_TEMP" \
#     --max_tokens 1024

# # ── Step 3: CSV → lesson XML ────────────────────────────────────────────────
# echo "=== Step 3: Converting questions CSV to lesson XML ==="
# python curriculum/csv_to_lesson.py \
#     --dataset_family "$DATASET_FAMILY" \
#     --dataset "$DATASET" \
#     --train_questions "$TRAIN_QUESTIONS" \
#     --temperature "$QUESTION_TEMP" \
#     --max_items "$MAX_ITEMS_TRAIN" \
#     --model "$MODEL_SHORT" \
#     --variant "$VARIANT"

# # ── Step 4: Create exam XML from dataset ────────────────────────────────────
# echo "=== Step 4: Creating exam XML ==="
# python curriculum/questions_to_exam.py \
#     --dataset_family "$DATASET_FAMILY" \
#     --dataset "$DATASET" \
#     --max_items "$MAX_ITEMS_TEST" \
#     --variant "$VARIANT"

# ── Step 5: Generate teacher answers for lessons (uses API server) ───────────
echo "=== Step 5: Generating teacher answers for lessons ==="
python curriculum/generate_teacher_answers.py \
    --base "$BASE" \
    --generate_lesson True \
    --dataset_family "$DATASET_FAMILY" \
    --dataset "$DATASET" \
    --variant "$VARIANT" \
    --question_model "$MODEL_SHORT" \
    --train_questions "$TRAIN_QUESTIONS" \
    --question_temperature "$QUESTION_TEMP" \
    --max_items "$MAX_ITEMS_TRAIN" \
    --lesson_temp "$LESSON_TEMP"

# ── Step 6: Generate teacher answers for exam (uses API server) ──────────────
echo "=== Step 6: Generating teacher answers for exam ==="
python curriculum/generate_teacher_answers.py \
    --base "$BASE" \
    --generate_exam True \
    --dataset_family "$DATASET_FAMILY" \
    --dataset "$DATASET" \
    --variant "$VARIANT" \
    --max_items "$MAX_ITEMS_TEST" \
    --exam_temp "$EXAM_TEMP"

# ── Step 7: Rewrite exam questions (uses API server) ────────────────────────
echo "=== Step 7: Rewriting exam questions ==="
EXAM_DATA_FILE="data/${DATASET_FAMILY}_${DATASET}_${VARIANT}_${MAX_ITEMS_TEST}_test_t${EXAM_TEMP}_qwen3-8b.xml"
python evaluation/rewrite_exam.py "$EXAM_DATA_FILE"

# ── Kill vLLM server to free GPUs for training ──────────────────────────────
echo "=== Shutting down vLLM server ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Step 8: Train student ───────────────────────────────────────────────────
echo "=== Step 8: Training student ==="
accelerate launch --num_processes 8 training/run_train_student.py \
    --run_name qasper_pipeline_10 \
    --dataset_family "$DATASET_FAMILY" \
    --dataset "$DATASET" \
    --variant "$VARIANT" \
    --base "$BASE" \
    --deepspeed_path config/ds2.json \
    --lesson_model "$BASE" \
    --exam_model "$BASE" \
    --question_model "$MODEL_SHORT" \
    --train_questions "$TRAIN_QUESTIONS" \
    --question_temperature "$QUESTION_TEMP" \
    --max_items_train "$MAX_ITEMS_TRAIN" \
    --max_items_test "$MAX_ITEMS_TEST" \
    --lesson_temp "$LESSON_TEMP" \
    --exam_temp "$EXAM_TEMP" \
    --lora_r 1024 \
    --batch_size 8 \
    --micro_batch_size 1 \
    --n_epochs 10 \
    --use_wandb True \
    --use_rewritten_val True \
    --save_during_training True \
    --save_interval 100 \
    --checkpoint_interval 100

echo "=== Pipeline complete ==="
