#!/bin/bash

# chmod +x run.sh

EPOCHS=500
BATCH_SIZE=128
NUM_WORKERS=8

DATASETS=("PATENT") # "PATENT" "IMDB" "COURSERA" "ARXIV"
MODES=("baseline" "nodes" "node_semantic_node_structure" "node_edges" "full") #"hnhn" "villain" "baseline" "nodes" "node_semantic_node_structure" "node_edges" "full"

for dataset in "${DATASETS[@]}"; do
  if dataset == "IMDB"; then
    BATCH_SIZE=512
    NUM_WORKERS=8
  elif dataset == "PATENT"; then
    BATCH_SIZE=128
    NUM_WORKERS=8
  elif dataset == "COURSERA"; then
    BATCH_SIZE=128
    NUM_WORKERS=8
  elif dataset == "ARXIV"; then
    BATCH_SIZE=256
    NUM_WORKERS=8
  fi
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
