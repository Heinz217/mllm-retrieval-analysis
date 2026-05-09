#!/bin/bash
# Train a Top-K SAE on the last-layer hidden states of Chameleon-7B,
# using activations collected from the COCO-Caption dataset.

MODEL_PATH="./checkpoints/chameleon-7b"
DATA_DIR="./coco-caption"
SAVE_DIR="./sae_checkpoints/chameleon"

BATCH_SIZE=4096
HIDDEN_DIM=4096
D_SAE=32768
N_EPOCHS=5

PYTHONPATH=. python ./sae_train/chameleon_sae_train.py \
  --model_id "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --sae_batch_size $BATCH_SIZE \
  --hidden_dim $HIDDEN_DIM \
  --d_sae $D_SAE \
  --n_epochs $N_EPOCHS
