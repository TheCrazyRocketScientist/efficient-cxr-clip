#!/bin/bash

# ---SETTINGS ---
ENV_NAME="testenv"
MARKER_FILE=".setup"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_ROOT="$SCRIPT_DIR"

# ---CONDA SETUP ---
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "Error: Miniconda is not installed. Please install it first."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---IF ALREADY INITIALIZED ---
if [ -f "$MARKER_FILE" ]; then
    echo "Setup already finished. Starting..."
else
    echo "First-time Setup ,Building Environment..."

    conda create -y -n $ENV_NAME python=3.10.20
    conda activate $ENV_NAME

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt

    #get data from kaggle here
    #setup dataset folder and env variable
    
    chmod +x ./scripts/*.sh
    chmod +x ./*.sh

    touch "$MARKER_FILE"
    echo "First-time Setup Finished."
fi

#load env vars for kaggle,wandb and hf
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# --- STARTUP ---
conda activate $ENV_NAME
source ./scripts/startup.sh
