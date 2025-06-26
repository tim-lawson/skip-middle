#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_LAYERS=${1:-12}

lbatch -a cosc020762 -q cnu --gputype A100 -c 16 -g 4 -m 124 -t 48 \
  --cmd python -m torch.distributed.run --standalone --nproc_per_node 4 projects/skip_middle/train_fineweb.py \
  --project fineweb-nocontrol --logdir logs/fineweb-nocontrol \
  --model.n_layers $N_LAYERS --model.zero_init_masks False
