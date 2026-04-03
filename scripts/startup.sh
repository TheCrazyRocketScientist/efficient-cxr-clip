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


#setup cache env vars and folders
source "$PROJECT_ROOT/scripts/set_cache.sh"
#load env vars for kaggle,wandb and hf
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi
#download dataset
kaggle datasets download -d paranjaychaudhary/cxr-bunch-512 -p $PROJECT_ROOT/datasets --unzip
#run backtranslation, this will create dataset/mimic_train.csv
if [ -f "$PROJECT_ROOT/datasets/mimic_train.csv" ]; then
    echo "Skipping Backtranslation."
else
    echo "Starting Backtranslation."
    python3 ./text_augmentation/back_translation.py
fi
#cache models
python3 ./scripts/cache_models.py
#run training
for vision in "${VISION_VARIANTS[@]}"; do
    for text in "${TEXT_VARIANTS[@]}"; do
        echo "Starting training for Vision: $vision | Text: $text"
        
        # We override the model encoders and the tokenizer to match
        python3 $PROJECT_ROOT/train.py \
            model.image_encoder.name="$vision" \
            model.image_encoder.source="huggingface" \
            model.text_encoder.name="$text" \
            tokenizer.pretrained_model_name_or_path="$text"
            
    done
done
#end
