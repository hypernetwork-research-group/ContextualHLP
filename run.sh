#!/bin/bash

# chmod +x run.sh

EPOCHS=250
BATCH_SIZE=128
NUM_WORKERS=8

DATASETS=("ARXIV")
MODES=("baseline" "nodes" "node_semantic_node_structure" "full") #"baseline" "nodes" "edges" "full" "node_semantic_node_structure" "just_node_semantic" "just_edge_semantic" "struct_llm_n" "node_llm_edge_llm"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
