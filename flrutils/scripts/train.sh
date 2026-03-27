#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o train_output.log
#SBATCH --job-name=train_qasper
#SBATCH --time=01:00:00         # Set appropriate time limit

source ~/fulianren/setup.sh
source .venv/bin/activate
conda deactivate

# python evaluation_sample_questions --vllm-hostname 0.0.0.0 & python

accelerate launch training/run_train_student.py \
     --run_name qasper_train \
     --dataset_family qasper \
     --dataset train \
     --use_wandb True \
     --deepspeed_path config/ds2.json \
     --deepspeed_path_teacher config/ds2.json \
     --n_epochs 1 \
     --save_during_training True \
     --save_interval 1000
