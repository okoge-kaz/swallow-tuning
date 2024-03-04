#!/bin/bash
#$ -l rt_AF=4
#$ -l h_rt=0:7:00:00
#$ -j y
#$ -o outputs/swallow-13b/instruction/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=$(($GLOBAL_BATCH_SIZE / $MICRO_BATCH_SIZE / $NUM_GPUS))

if [[ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]]; then
  echo "Global batch size is too small for the number of GPUs"
  exit 1
fi

LR=2e-5
MIN_LR=2e-6

beta_1=0.9
beta_2=0.95

WEIGHT_DECAY=0.1
GRAD_CLIP=1

EPOCH=2
SEQ_LENGTH=4096

# checkpoint & tokenizer
TOKENIZER_DIR=/bb/llm/gaf51275/llama/huggingface-checkpoint/Swallow-13b-hf
CHECKPOINT_DIR=/bb/llm/gaf51275/llama/huggingface-checkpoint/Swallow-13b-hf
CHECKPOINT_SAVE_DIR="/bb/llm/gaf51275/llama/checkpoints/Swallow-13b-VE-instruct-v1.0/hh-rlhf-dolly-ossat-imitation-lr_${LR}-minlr_${MIN_LR}-GB${GLOBAL_BATCH_SIZE}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# dataset
DATASET_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/training/hh-rlhf-dolly-ossat-imitation

TRAIN_DATA_PATH=${DATASET_DIR}/train.jsonl
VALID_DATA_PATH=${DATASET_DIR}/val.jsonl

# deepspeed config
config_json="./deepspeed_config.json"

zero_stage=2
train_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
optimizer="Adam"
optimizer_params="{\"lr\": $LR, \"betas\": [$beta_1, $beta_2], \"eps\": 1e-6, \"weight_decay\": $WEIGHT_DECAY}"
gradient_clipping=$GRAD_CLIP
bf16="{\"enabled\": true}"

echo "{
  \"zero_optimization\": {
    \"stage\": $zero_stage
  },
  \"train_micro_batch_size_per_gpu\": $train_micro_batch_size_per_gpu,
  \"optimizer\": {
    \"type\": \"$optimizer\",
    \"params\": $optimizer_params
  },
  \"gradient_clipping\": $gradient_clipping,
  \"gradient_accumulation_steps\": $GRADIENT_ACCUMULATION_STEPS,
  \"bf16\": $bf16
}" > $config_json

# job name
JOB_NAME="Swallow-13b-VE-instruct-hh-rlhf-dolly-ossat-imitation-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}"

export WANDB_ENTITY="prj-jalm"
export WANDB_PROJECT="Llama-2-13b-instruct"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
  -x PATH \
  python train.py \
    --do_train \
    --do_eval \
    --model_name_or_path $CHECKPOINT_DIR \
    --tokenizer_name_or_path $TOKENIZER_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LR \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --adam_beta1 $beta_1 \
    --adam_beta2 $beta_2 \
    --adam_epsilon 1e-6 \
    --max_grad_norm $GRAD_CLIP \
    --weight_decay $WEIGHT_DECAY \
    --bf16 \
    --max_seq_length $SEQ_LENGTH \
    --gradient_checkpointing \
    --save_total_limit 2 \
    --logging_steps 1 \
    --report_to wandb \
    --run_name $JOB_NAME \
    --log_on_each_node False \
    --deepspeed ${config_json} \
    --save_strategy epoch \
    --save_safetensors True \
    --save_on_each_node False \
    --output_dir $CHECKPOINT_SAVE_DIR \
    --train_data_path $TRAIN_DATA_PATH \
    --val_data_path $VALID_DATA_PATH \
    --deepspeed $config_json \
    --use_flash_attention_2 True
