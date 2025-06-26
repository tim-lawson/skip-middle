#!/bin/bash

#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --time=1-00:00:00

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_LAYERS=${1:-12}

python -m torch.distributed.run --standalone --nproc_per_node 4 projects/skip_middle/train_fineweb.py \
  --project fineweb-baseline --logdir logs/fineweb-baseline --model.n_layers $N_LAYERS
