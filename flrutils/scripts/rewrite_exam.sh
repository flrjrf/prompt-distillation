#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o rewrite_exam_%j.log
#SBATCH --job-name=rewrite_exam
#SBATCH --time=24:00:00

source ~/fulianren/setup.sh
source .venv/bin/activate

# Launch vllm server in background
python -u -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 8 &

# Wait for the server to be ready
echo "Waiting for vllm server to start..."
until curl -sf http://localhost:8000/health > /dev/null; do
    sleep 5
done
echo "vllm server is ready. Starting exam rewriting."

python evaluation/rewrite_exam.py data/qasper_train_default_20_test_t0.25.xml
