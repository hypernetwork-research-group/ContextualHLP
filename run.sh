#!/bin/bash

# chmod +x run.sh

EPOCHS=550
BATCH_SIZE=128
NUM_WORKERS=8

DATASETS=("IMDB") # "PATENT" "IMDB" "COURSERA" "ARXIV"
MODES=("full") # "baseline" "nodes" "node_semantic_node_structure" "node_edges" "full"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
