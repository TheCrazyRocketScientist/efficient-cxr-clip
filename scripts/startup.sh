#!/bin/bash

PAIRS=(
    "timm/fastvit_sa12.apple_in1k|nlpie/tiny-biobert"
    "timm/fastvit_sa12.apple_in1k|nlpie/bio-mobilebert"
)

source "$PROJECT_ROOT/scripts/set_cache.sh"
python3 ./scripts/cache_models.py

for pair in "${PAIRS[@]}"; do
    vision="${pair%%|*}"
    text="${pair##*|}"

    echo "----------------------------------------------------------------"
    echo "Targeting Spread Pair: $vision | $text"
    echo "----------------------------------------------------------------"

    DYNAMIC_REV=$(python3 "$PROJECT_ROOT/scripts/grab_correct_pr.py" --model_id "$text" 2>/dev/null | tail -n 1)
    if [ -z "$DYNAMIC_REV" ]; then DYNAMIC_REV="main"; fi
    
    # CRITICAL FIX: Wrap the model/text names in "'$var'" 
    # This tells Hydra: "This is a literal string, don't parse the slashes"
    python3 "$PROJECT_ROOT/train.py" \
        dataloader.train.batch_size=128 \
        dataloader.valid.batch_size=128 \
        dataloader.test.batch_size=128 \
        dataloader.train.num_workers=16 \
        dataloader.valid.num_workers=16 \
        dataloader.test.num_workers=16 \
        dataloader.train.pin_memory=True \
        ++dataloader.train.persistent_workers=True \
        ++dataloader.train.prefetch_factor=4 \
        ++dataloader.valid.pin_memory=True \
        ++dataloader.valid.persistent_workers=True \
        ++dataloader.valid.prefetch_factor=4 \
        ++dataloader.test.pin_memory=True \
        ++dataloader.test.persistent_workers=True \
        ++dataloader.test.prefetch_factor=4 \
        "model.image_encoder.name='$vision'" \
        model.image_encoder.source="huggingface" \
        +model.image_encoder.model_type="fastvit" \
        ++model.image_encoder.cache_dir="/kaggle/working/efficient-cxr-clip/.assets/huggingface" \
        ++model.image_encoder.local_files_only=False \
        "model.text_encoder.name='$text'" \
        "+model.text_encoder.revision='$DYNAMIC_REV'" \
        ++model.text_encoder.use_safetensors=True \
        ++model.text_encoder.local_files_only=False \
        "tokenizer.pretrained_model_name_or_path='$text'" \
        tokenizer.cache_dir="/kaggle/working/efficient-cxr-clip/.assets/huggingface" \
        ++tokenizer.local_files_only=False
done