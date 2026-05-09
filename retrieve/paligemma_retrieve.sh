#!/bin/bash
# Zero-shot multimodal retrieval on M-BEIR with PaliGemma2-3B-Mix-224,
# evaluated with both vanilla mean-pooling (BASE) and ReAlign whitening (SHRINKAGE).

DATASET_DIR="./M-BEIR/mbeir_mscoco_task4"
MODEL_PATH="./checkpoints/paligemma2-3b-mix-224"
TASK=4

PYTHONPATH=. python ./retrieve/paligemma_retrieve.py \
  --model_path "$MODEL_PATH" \
  --batch_size 1 \
  --task_num $TASK \
  --dataset_path "$DATASET_DIR"
