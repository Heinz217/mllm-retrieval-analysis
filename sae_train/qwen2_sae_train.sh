#!/bin/bash
# Train a Top-K SAE on the last-layer hidden states of Qwen2-VL-7B-Instruct,
# using activations collected from the COCO-Caption dataset.

MODEL_PATH="./checkpoints/Qwen2-VL-7B-Instruct"
DATA_DIR="./coco-caption"
SAVE_DIR="./sae_checkpoints/qwen2vl"

BATCH_SIZE=4096
HIDDEN_DIM=3584
D_SAE=32768
N_EPOCHS=5

PYTHONPATH=. python ./sae_train/qwen2_sae_train.py \
  --model_id "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --sae_batch_size $BATCH_SIZE \
  --hidden_dim $HIDDEN_DIM \
  --d_sae $D_SAE \
  --n_epochs $N_EPOCHS
