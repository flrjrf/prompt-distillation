#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o generate_questions_%j.log
#SBATCH --job-name=gen_questions
#SBATCH --time=24:00:00

source ~/fulianren/setup.sh
source .venv/bin/activate

# Launch vllm server in background
python -u -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 8 

# Wait for the server to be ready
echo "Waiting for vllm server to start..."
until curl -sf http://localhost:8000/health > /dev/null; do
    sleep 5
done
echo "vllm server is ready. Starting question generation."

# Run question generation jobs in parallel (use & to background, wait to join)
python evaluation/sample_questions.py --dataset_family qasper --dataset train --max_tokens 1024 --train_questions 60 
