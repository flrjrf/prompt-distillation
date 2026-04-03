#!/bin/bash
#SBATCH -G 8
#SBATCH -p j01
#SBATCH -o squadshifts_train_output.log
#SBATCH --job-name=squadshifts_train
#SBATCH --time=48:00:00         # Set appropriate time limit

source ~/fulianren/setup.sh
source .venv/bin/activate

# accelerate launch --num_processes 8 training/run_train_student.py \
#      --run_name qasper_train \
#      --dataset_family qasper \
#      --lora_r 1024 \
#      --batch_size 16 \
#      --micro_batch_size 2 \
#      --dataset train \
#      --use_wandb True \
#      --deepspeed_path config/ds2.json \
#      --n_epochs 10 \
#      --save_during_training True \
#      --save_interval 100\
#      --checkpoint_interval 100

# accelerate launch --num_processes 8 training/run_train_student.py \
#      --run_name squadshifts_train \
#      --dataset_family squadshifts \
#      --batch_size 16 \
#      --micro_batch_size 2 \
#      --lora_r 1024 \
#      --dataset nyt \
#      --use_wandb True \
#      --deepspeed_path config/ds2.json \
#      --n_epochs 10 \
#      --save_during_training True \
#      --save_interval 100 \
#      --checkpoint_interval 100 

# python training/run_train_student.py \
#      --run_name squadshifts_train \
#      --dataset_family squadshifts \
#      --batch_size 4 \
#      --micro_batch_size 4 \
#      --lora_r 1024 \
#      --dataset nyt \
#      --use_wandb True \
#      # --deepspeed_path config/ds2.json \
#      # --deepspeed_path_teacher config/ds2.json \
#      --n_epochs 10 \
#      --save_during_training True \
#      --save_interval 100 \
#      --checkpoint_interval 100 \

python training/run_train_student.py \
     --run_name qasper_train \
     --dataset_family qasper \
     --batch_size 4 \
     --micro_batch_size 4 \
     --lora_r 1024 \
     --dataset train \
     --use_wandb True \
     # --deepspeed_path config/ds2.json \
     # --deepspeed_path_teacher config/ds2.json \
     --n_epochs 10 \
     --save_during_training True \
     --save_interval 100 \
     --checkpoint_interval 100 \
