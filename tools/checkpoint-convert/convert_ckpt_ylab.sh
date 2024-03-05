#!/bin/bash
#YBATCH -r epyc-7543_8
#SBATCH --job-name=convert
#SBATCH --time=2:00:00
#SBATCH --output outputs/convert/%j.out
#SBATCH --error errors/convert/%j.out
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

ITERATION=866

# ZeRO -> PyToech
CHECK_POINT_DIR=/home/kazuki/checkpoints/Swallow-70b-NEFTune/baseline-imitation_2-lr_1e-5-minlr_1e-6-GB_256/checkpoint-${ITERATION}

python tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/model.pt \
  --debug

# PyTorch -> Hugging Face
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_PATH=${CHECK_POINT_DIR}/model.pt
OUTPUT_PATH=/home/kazuki/converted_checkpoints/Swallow-70b-VE-NEFTune/baseline-imitation_2-lr_1e-5-minlr_1e-6-GB_256/${FORMATTED_ITERATION}

echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

mkdir -p $OUTPUT_PATH

BASE_MODEL_CHECKPOINT=/home/kazuki/hf_checkpoints/Swallow-70b-hf

python tools/checkpoint-convert/convert_ckpt.py \
  --model $BASE_MODEL_CHECKPOINT \
  --ckpt $CHECK_POINT_PATH \
  --out $OUTPUT_PATH \
  --sequence-length 4096
