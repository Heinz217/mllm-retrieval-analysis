#!/bin/bash
# Train a Top-K SAE on the last-layer hidden states of PaliGemma2-3B-Mix-224,
# using activations collected from the COCO-Caption dataset.

MODEL_PATH="./checkpoints/paligemma2-3b-mix-224"
DATA_DIR="./coco-caption"
SAVE_DIR="./sae_checkpoints/paligemma"

BATCH_SIZE=4096
HIDDEN_DIM=2304
D_SAE=20736
N_EPOCHS=5

PYTHONPATH=. python ./sae_train/paligemma_sae_train.py \
  --model_id "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --sae_batch_size $BATCH_SIZE \
  --hidden_dim $HIDDEN_DIM \
  --d_sae $D_SAE \
  --n_epochs $N_EPOCHS
