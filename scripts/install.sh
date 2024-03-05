#!/bin/bash
#YBATCH -r rtx6000-ada_1
#SBATCH --job-name=install
#SBATCH --time=1:00:00
#SBATCH --output outputs/install/%j.out
#SBATCH --error errors/install/%j.out
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

set -e

pip install -r requirements.txt

pip install flash-attn --no-build-isolation
