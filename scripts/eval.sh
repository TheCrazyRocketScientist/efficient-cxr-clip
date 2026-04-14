#!/bin/bash

# Configuration
PROJECT_ROOT="$(pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/model_checkpoints"
FETCH_SCRIPT="$PROJECT_ROOT/scripts/model_fetch.py"
EVAL_SCRIPT="$PROJECT_ROOT/evaluate_clip.py"

DATASETS=("mimic_cxr" "chexpert5x200" "rsna_pneumonia" "siim_pneumothorax")

if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR"/*.tar 2>/dev/null)" ]; then
    echo "Checkpoint directory is empty. Fetching renamed artifacts..."
    python3 "$FETCH_SCRIPT"
fi

for model_path in "$CHECKPOINT_DIR"/*.tar; do
    model_name=$(basename "$model_path" .tar)
    echo "Evaluating model: $model_name"

    for dataset in "${DATASETS[@]}"; do
        echo "Running $dataset zero-shot evaluation..."
        
        python3 "$EVAL_SCRIPT" \
            test.checkpoint="$model_path" \
            data_test=["$dataset"] \
            dataloader.test.batch_size=64 \
            dataloader.test.num_workers=16 \
            ++dataloader.test.pin_memory=True \
            ++dataloader.test.prefetch_factor=4
            
        if [ $? -eq 0 ]; then
            echo "Finished $dataset"
        else
            echo "Failed $dataset"
        fi
    done
done

echo "Zero-shot evaluation suite complete."