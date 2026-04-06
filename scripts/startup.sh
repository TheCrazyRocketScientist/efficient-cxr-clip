#!/bin/bash

VISION_VARIANTS=(
    "timm/fastvit_t8.apple_in1k"
    "timm/fastvit_s12.apple_in1k"
    "timm/fastvit_sa24.apple_in1k"
    "timm/fastvit_ma36.apple_in1k"
)

TEXT_VARIANTS=(
    "nlpie/compact-biobert"
    "nlpie/tiny-biobert"
    "nlpie/tiny-clinicalbert"
    "nlpie/distil-biobert"
    "nlpie/bio-tinybert"
    "nlpie/distil-clinicalbert"
    "nlpie/bio-mobilebert"
)

# Setup cache env vars and folders
source "$PROJECT_ROOT/scripts/set_cache.sh"

# Download dataset
kaggle datasets download -d paranjaychaudhary/cxr-bunch-512 -p $PROJECT_ROOT --unzip

# Run backtranslation
if [ -f "$PROJECT_ROOT/datasets/mimic_train.csv" ]; then
    echo "Skipping Backtranslation."
else
    echo "Starting Backtranslation."
    python3 ./text_augmentation/back_translation.py
fi

# Cache models
python3 ./scripts/cache_models.py

# Run training
for vision in "${VISION_VARIANTS[@]}"; do
    for text in "${TEXT_VARIANTS[@]}"; do
        echo "Fetching correct PR revision for: $text"
        
        # Dynamically get the best PR or main branch using your updated script
        DYNAMIC_REV=$(python3 "$PROJECT_ROOT/scripts/grab_correct_pr.py" --model_id "$text")
        
        echo "Starting training for Vision: $vision | Text: $text | Revision: $DYNAMIC_REV"
        
        # We override the model encoders, the tokenizer, and the dynamic revision
        # Using ~ to exclude datasets as per your example call
        python3 "$PROJECT_ROOT/train.py" \
            ~data_train.mimic_cxr \
            ~data_valid.mimic_cxr \
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
done