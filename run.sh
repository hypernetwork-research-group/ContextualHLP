#!/bin/bash

# chmod +x run.sh

EPOCHS=1000
BATCH_SIZE=16
NUM_WORKERS=8

DATASETS=("PROVA" "PATENT" "ARXIV") # "COURSERA" "PROVA" "PATENT" "ARXIV"
MODES=("full") #"baseline" "nodes" "edges" "full" "node_semantic_node_structure"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
