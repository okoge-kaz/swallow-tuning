#!/bin/bash

INPUT_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/formatted
OUTPUT_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/training/imitation_2_oasst2_top1

mkdir -p $OUTPUT_DIR

FILES_TO_MERGE=(
  "oasst1-21k-ja-mixtral-imitation_2.jsonl"
  "oasst2-top1-en.jsonl"
)

# merge files
for FILE in "${FILES_TO_MERGE[@]}"; do
  cat "$INPUT_DIR/$FILE" >> "$OUTPUT_DIR/merged.jsonl"
done

echo "Merged dataset is saved at $OUTPUT_DIR/merged.jsonl"

# swich virtual env
source .env/bin/activate

python tools/dataset/shuffle_and_split.py \
  --input $OUTPUT_DIR/merged.jsonl \
  --output $OUTPUT_DIR
