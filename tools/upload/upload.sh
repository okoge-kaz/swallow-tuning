#!/bin/bash

set -e

start=800
end=800
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-13b-VE-NEFTune/dolly-oasst2-top1-imitation-2-3-lr_2e-5-minlr_2e-6-GB_256

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-13b-VE-NEFTune-v1.0-dolly-oasst2-top1-imitation-2-3-iter$(printf "%07d" $i)
done
