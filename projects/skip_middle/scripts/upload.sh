#! /bin/bash

STEP=${1:-19560}
LAYERS=(2 4 6 8 10 12)

for N_LAYERS in "${LAYERS[@]}"; do
  LOGDIR="models/fineweb/fineweb-baseline-${N_LAYERS}-layers"
  REPO_ID="tim-lawson/fineweb-baseline-${N_LAYERS}-layers"

  # cp "${LOGDIR}/config.json" "${LOGDIR}/train_config.json"
  huggingface-cli upload $REPO_ID $LOGDIR .
  python -m projects.skip_middle.model.huggingface --logdir $LOGDIR --step $STEP --repo_id $REPO_ID
  huggingface-cli download $REPO_ID --local-dir $LOGDIR
done

LAYERS=(2 4 6 8 10 12)

for N_LAYERS in "${LAYERS[@]}"; do
  LOGDIR="models/fineweb/fineweb-nocontrol-${N_LAYERS}-layers"
  REPO_ID="tim-lawson/fineweb-nocontrol-${N_LAYERS}-layers"

  # cp "${LOGDIR}/config.json" "${LOGDIR}/train_config.json"
  huggingface-cli upload $REPO_ID $LOGDIR .
  python -m projects.skip_middle.model.huggingface --logdir $LOGDIR --step $STEP --repo_id $REPO_ID
  huggingface-cli download $REPO_ID --local-dir $LOGDIR
done

TARGET_ENDS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.25)

for TARGET_END in "${TARGET_ENDS[@]}"; do
  LOGDIR="models/fineweb/fineweb-gated-target-end-${TARGET_END}"
  REPO_ID="tim-lawson/fineweb-gated-target-end-${TARGET_END}"

  # cp "${LOGDIR}/config.json" "${LOGDIR}/train_config.json"
  huggingface-cli upload $REPO_ID $LOGDIR .
  python -m projects.skip_middle.model.huggingface --logdir $LOGDIR --step $STEP --repo_id $REPO_ID
  huggingface-cli download $REPO_ID --local-dir $LOGDIR
done
