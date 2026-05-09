#!/bin/bash
# Zero-shot multimodal retrieval on M-BEIR with Chameleon-7B,
# evaluated with both vanilla mean-pooling (BASE) and ReAlign whitening (SHRINKAGE).

DATASET_DIR="./M-BEIR/mbeir_mscoco_task4"
MODEL_PATH="./checkpoints/chameleon-7b"
TASK=4

PYTHONPATH=. python ./retrieve/chameleon_retrieve.py \
  --model_path "$MODEL_PATH" \
  --batch_size 1 \
  --task_num $TASK \
  --dataset_path "$DATASET_DIR"
