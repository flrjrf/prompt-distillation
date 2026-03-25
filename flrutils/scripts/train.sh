#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o train_output.log
#SBATCH --job-name=example_train
#SBATCH --time=24:00:00         # Set appropriate time limit

source ~/fulianren/setup.sh
source .venv/bin/activate

python evaluation_sample_questions --vllm-hostname 0.0.0.0 & python

# python training/run_train_student.py --run_name example --use_wandb True
