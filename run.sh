#!/bin/bash

# chmod +x run.sh
# source /path/to/venv/bin/activate

EPOCHS=1200
BATCH_SIZE=64
NUM_WORKERS=8

DATASETS=("IMDB") # "COURSERA" "IMDB" "ARXIV"
MODES=("baseline" "nodes" "edges" "full") # baseline" "nodes" "full" 

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
