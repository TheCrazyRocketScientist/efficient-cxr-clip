#!/bin/bash

# Define your specific "Spread" pairs here: "VISION|TEXT"
# Add or remove pairs as needed to match your experiment design
PAIRS=(
    "timm/fastvit_sa12.apple_in1k|nlpie/tiny-biobert"
    "timm/fastvit_sa12.apple_in1k|nlpie/bio-mobilebert"
)

# Setup cache env vars and folders
source "$PROJECT_ROOT/scripts/set_cache.sh"

# Cache models
python3 ./scripts/cache_models.py

# Loop through the specific spread bypass
for pair in "${PAIRS[@]}"; do
    # Split the pair string into vision and text variables
    vision="${pair%%|*}"
    text="${pair##*|}"

    echo "----------------------------------------------------------------"
    echo "Targeting Spread Pair:"
    echo "Vision: $vision"
    echo "Text:   $text"
    echo "----------------------------------------------------------------"

    echo "Fetching correct PR revision for: $text"
    DYNAMIC_REV=$(python3 "$PROJECT_ROOT/scripts/grab_correct_pr.py" --model_id "$text" 2>/dev/null | tail -n 1)
    
    if [ -z "$DYNAMIC_REV" ]; then DYNAMIC_REV="main"; fi
    
    echo "Starting training for Revision: $DYNAMIC_REV"

    # Execute training for this specific pair
    python3 "$PROJECT_ROOT/train.py" \
        dataloader.train.batch_size=64 \
        dataloader.valid.batch_size=64 \
        dataloader.test.batch_size=64 \
        dataloader.train.num_workers=4 \
        dataloader.valid.num_workers=4 \
        dataloader.test.num_workers=4 \
        model.image_encoder.name="$vision" \
        model.image_encoder.source="huggingface" \
        +model.image_encoder.model_type="fastvit" \
        ++model.image_encoder.cache_dir="/kaggle/working/efficient-cxr-clip/.assets/huggingface" \
        ++model.image_encoder.local_files_only=False \
        model.text_encoder.name="$text" \
        +model.text_encoder.revision="$DYNAMIC_REV" \
        ++model.text_encoder.use_safetensors=True \
        ++model.text_encoder.local_files_only=False \
        tokenizer.pretrained_model_name_or_path="$text" \
        tokenizer.cache_dir="/kaggle/working/efficient-cxr-clip/.assets/huggingface" \
        ++tokenizer.local_files_only=False
            
done