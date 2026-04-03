#!/bin/bash

# Get the directory of THIS script (e.g., .../cxr-clip/bash_scripts)
export ASSETS_DIR="$PROJECT_ROOT/.assets"

mkdir -p "$ASSETS_DIR/huggingface/datasets"
mkdir -p "$ASSETS_DIR/huggingface/models"
mkdir -p "$ASSETS_DIR/torch"


export HF_HOME="$ASSETS_DIR/huggingface"
export HF_DATASETS_CACHE="$ASSETS_DIR/huggingface/datasets"
export TRANSFORMERS_CACHE="$ASSETS_DIR/huggingface/models"
export TORCH_HOME="$ASSETS_DIR/torch"

