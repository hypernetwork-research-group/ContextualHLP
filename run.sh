# chmod +x run.sh

EPOCHS=800
BATCH_SIZE=512
NUM_WORKERS=8

DATASETS=("IMDB" "COURSERA" "ARXIV")
MODES=("baseline" "nodes" "edges" "full")

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "======== Running: $dataset - $mode ========"
    python train.py --dataset "$dataset" --mode "$mode" --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
    echo ""
  done
done
