#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_LAYERS=${1:-12}
MEAN_TARGET_START=${2:-1}
MEAN_TARGET_END=${3:-0.5}
COEF_UPDATE_MULTIPLIER=${4:-0.001}

lbatch -a cosc020762 -q cnu --gputype A100 -c 16 -g 4 -m 124 -t 24 \
  --cmd python -m torch.distributed.run --standalone --nproc_per_node 4 projects/skip_middle/train_fineweb.py \
  --project fineweb-gated --logdir logs/fineweb-gated \
  --model.n_layers $N_LAYERS --model.zero_init_masks False \
  --gates proportional \
  --gates.mean_target_start $MEAN_TARGET_START \
  --gates.mean_target_end $MEAN_TARGET_END \
  --gates.coef_update_multiplier $COEF_UPDATE_MULTIPLIER
