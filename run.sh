#!/bin/bash

# chmod +x run.sh

EPOCHS=1500
BATCH_SIZE=16
NUM_WORKERS=8

DATASETS=("PATENT" "COURSERA" "PROVA" "ARXIV") # "PATENT" "COURSERA" "IMDB" "ARXIV" "PROVA" "PROVA" "ARXIV"
MODES=("baseline" "nodes" "edges" "full") # "baseline" "nodes" "edges" "full"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
