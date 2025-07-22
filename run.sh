#!/bin/bash

# chmod +x run.sh

EPOCHS=500
BATCH_SIZE=16
NUM_WORKERS=8

DATASETS=("COURSERA" "IMDB" "PATENT" "ARXIV") # "COURSERA" "IMDB" "PATENT" "ARXIV"
MODES=("baseline" "nodes" "edges" "full" "node_semantic_node_structure" "node_llm_edge_llm") #"baseline" "nodes" "edges" "full" "node_semantic_node_structure" "just_node_semantic" "just_edge_semantic" "struct_llm_n" "node_llm_edge_llm"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
