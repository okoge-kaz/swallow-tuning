#!/bin/bash

DATASET_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/formatted

# databricks-dolly-15k
python tools/dataset/upload_dataset.py \
  --input $DATASET_DIR/databricks-dolly-15k-ja.jsonl \
  --upload tokyotech-llm/databricks-dolly-15k-ja

python tools/dataset/upload_dataset.py \
  --input $DATASET_DIR/databricks-dolly-15k-en.jsonl \
  --upload tokyotech-llm/databricks-dolly-15k-en

# oasst1-21k
python tools/dataset/upload_dataset.py \
  --input $DATASET_DIR/oasst1-21k-ja.jsonl \
  --upload tokyotech-llm/oasst1-21k-ja

python tools/dataset/upload_dataset.py \
  --input $DATASET_DIR/oasst1-21k-en.jsonl \
  --upload tokyotech-llm/oasst1-21k-en

# hh-rlhf-12k
python tools/dataset/upload_dataset.py \
  --input $DATASET_DIR/hh-rlhf-12k-ja.jsonl \
  --upload tokyotech-llm/hh-rlhf-12k-ja

# ichikara
python tools/dataset/upload_dataset.py \
  --input $DATASET_DIR/ichikara.jsonl \
  --upload tokyotech-llm/ichikara
