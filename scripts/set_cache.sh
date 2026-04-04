#!/bin/bash

# Get the directory of THIS script (e.g., .../cxr-clip/bash_scripts)
export ASSETS_DIR="$PROJECT_ROOT/.assets"

mkdir -p "$ASSETS_DIR/huggingface"
mkdir -p "$ASSETS_DIR/torch"

export HF_HOME="$ASSETS_DIR/huggingface"
export TORCH_HOME="$ASSETS_DIR/torch"

unset TRANSFORMERS_CACHE
unset HF_DATASETS_CACHE


