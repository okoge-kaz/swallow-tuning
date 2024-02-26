#!/bin/bash

DATASET_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/formatted

# databricks-dolly-15k
python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/databricks-dolly-15k-ja.jsonl \
  --output $DATASET_DIR/databricks-dolly-15k-ja.json

python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/databricks-dolly-15k-en.jsonl \
  --output $DATASET_DIR/databricks-dolly-15k-en.json

# oasst1-21k
python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/oasst1-21k-ja.jsonl \
  --output $DATASET_DIR/oasst1-21k-ja.json

python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/oasst1-21k-en.jsonl \
  --output $DATASET_DIR/oasst1-21k-en.json

# oasst2
python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/oasst2-en.jsonl \
  --output $DATASET_DIR/oasst2-en.json

python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/oasst2-top1-en.jsonl \
  --output $DATASET_DIR/oasst2-top1-en.json

# hh-rlhf-12k
python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/hh-rlhf-12k-ja.jsonl \
  --output $DATASET_DIR/hh-rlhf-12k-ja.json

# ichikara
python tools/dataset/convert_jsonl_to_json.py \
  --input $DATASET_DIR/ichikara.jsonl \
  --output $DATASET_DIR/ichikara.json
