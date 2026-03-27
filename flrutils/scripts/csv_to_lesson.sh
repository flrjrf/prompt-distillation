# python curriculum/csv_to_lesson.py \
#   --dataset_family=mtob \
#   --dataset=grammar_book_chunks \
#   --chunk_size=256 \
#   --chunking_method=sliding \
#   --model=Qwen3-4B-Instruct-2507 \
#   --train_questions=30 \
#   --temperature=1.5 \
#   --max_items=1000 \
#   --variant=default
python curriculum/csv_to_lesson.py \
  --dataset_family=qasper \
  --dataset=train \

