#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

ITERATION=2154

# ZeRO -> PyToech
CHECK_POINT_DIR=/bb/llm/gaf51275/llama/checkpoints/Swallow-13b-VE-instruct-v1.0/hh-rlhf-dolly-ossat-imitation-lr_2e-5-minlr_2e-6-GB128/checkpoint-${ITERATION}

python tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/model.pt \
  --debug

# PyTorch -> Hugging Face
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_PATH=${CHECK_POINT_DIR}/model.pt
OUTPUT_PATH=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-13b-VE-instruct/hh-rlhf-dolly-ossat-imitation-lr_2e-5-minlr_2e-6-GB128/${FORMATTED_ITERATION}

echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

mkdir -p $OUTPUT_PATH

BASE_MODEL_CHECKPOINT=/bb/llm/gaf51275/llama/huggingface-checkpoint/Swallow-13b-hf

python tools/checkpoint-convert/convert_ckpt.py \
  --model $BASE_MODEL_CHECKPOINT \
  --ckpt $CHECK_POINT_PATH \
  --out $OUTPUT_PATH \
  --sequence-length 4096
