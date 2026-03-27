#!/bin/bash
#SBATCH -G 1
#SBATCH -p j01
#SBATCH -o generate_lesson_output.log
#SBATCH --job-name=generate_teacher_answers
#SBATCH --time=06:00:00         # Set appropriate time limit

source ~/fulianren/setup.sh
source .venv/bin/activate
conda deactivate

python3 curriculum/generate_teacher_answers.py --dataset_family qasper --dataset train --generate_lesson True
# python3 curriculum/generate_teacher_answers.py --dataset_family qasper --dataset train --generate_exam True --max_items 20
