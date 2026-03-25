#!/bin/bash

# Set variables at the top
INPUT_FILE="datasets/kalamang/resources/grammar_book_combined.txt"
CHUNK_SIZE=256
MODE="sliding"
OUTPUT_FILE="flrutils/grammar_book_chunks_${CHUNK_SIZE}_${MODE}.json"

# Call chunk_text.py with the variables
CSV_FILE="${OUTPUT_FILE%.json}.csv"
python flrutils/chunk_text.py "$INPUT_FILE" "$OUTPUT_FILE" --tokens $CHUNK_SIZE --mode $MODE --output-csv "$CSV_FILE"
