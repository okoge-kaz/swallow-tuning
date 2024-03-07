#!/bin/bash

set -e

start=2154
end=2154
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-13b-VE-instruct/hh-rlhf-dolly-ossat-imitation-lr_2e-5-minlr_2e-6-GB128

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-13b-VE-instruct-v1.0-hh-rlhf-dolly-ossat-imitation-lr_2e-5-minlr_2e-6-GB128-iter$(printf "%07d" $i)
done
