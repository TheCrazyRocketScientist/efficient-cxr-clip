# Efficient-CXR-CLIP

Efficient-CXR-CLIP is a modification of the official implementation of **"CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training"** [[arxiv]](https://arxiv.org/abs/2310.13292). 

This repository evaluates the effectiveness of lightweight, edge-optimized CLIP models for medical imaging, specifically focusing on FastViT and compact BioBERT variants (For now).

## Key Features Added
- **timm Support:** Native integration of `timm` (PyTorch Image Models) for advanced vision encoders.
- **Enhanced Logging:** Full integration with **Weights & Biases (W&B)** for experiment tracking and metric visualization.
- **Automated Pipelines:** Streamlined "First-Time Setup" that handles environment initialization and data augmentation.
- **Cache Management:** Robust handling of model and dataset caching.

## Roadmap & To-Do
- [ ] Implement Neural Architecture Search (NAS) via Hydra Optuna Bridge.
- [ ] Integrate CheXpert dataset loaders.
- [ ] Add linear probing and fine-tuning evaluation scripts.
- [ ] Log edge-specific metrics (GFLOPs, Accuracy per Joule, Inference Latency).

## Requirements
While the repository includes automation scripts, we recommend manual enforcement of the following environment:
- **Python:** 3.10.20
- **CUDA:** 12+
- **Package Manager:** Miniconda / Anaconda
- **API Access:** Valid tokens for Hugging Face, W&B, and Kaggle (replace existing temporary tokens in `.env`).

```bash
pip install -r requirements.txt
```

## Getting Started

For the first-time installation and pre-training entry point, run the initialization script:

```bash
source init.sh
```

This script manages environment checks and triggers the scripts/startup.sh workflow, which includes performing back-translation on the training split.

Only run this script.
## Supported Model Variants
We evaluate combinations of the following efficient encoders:

**Vision Encoders:**
- timm/fastvit_t8.apple_in1k

- timm/fastvit_s12.apple_in1k

- timm/fastvit_sa24.apple_in1k

- timm/fastvit_ma36.apple_in1k

**Text Encoders:**
- nlpie/compact-biobert

- nlpie/tiny-biobert

- nlpie/tiny-clinicalbert

- nlpie/distil-biobert

- nlpie/bio-tinybert

- nlpie/distil-clinicalbert

- nlpie/bio-mobilebert

## Data Preparation
This repository currently supports the MIMIC-CXR dataset using official splits.

**Training:** Preprocessed with automated back-translation.
**Structure:** Ensure CSV files are placed in the datasets/ directory.

For detailed instructions on raw data acquisition, refer to  [data preparation](datasets/README.md).

## Pre-Train model
### command line
* single gpu
    ```bash
    python train.py {--config-name default_config}
    ```
* multi gpu
    ```bash
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=45678 train.py {--config-name default_config}
    ```
## Citation
If you use this modification or the original implementation, please cite the original work:
```
@incollection{You_2023,
	doi = {10.1007/978-3-031-43895-0_10},
	url = {https://doi.org/10.1007%2F978-3-031-43895-0_10},
	year = 2023,
	publisher = {Springer Nature Switzerland},
	pages = {101--111},
	author = {Kihyun You and Jawook Gu and Jiyeon Ham and Beomhee Park and Jiho Kim and Eun K. Hong and Woonhyuk Baek and Byungseok Roh},
	title="CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training",
	booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
}
```

## License
CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training © 2023 is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1)

## Contact for Issues
Paranjay Lokesh Chaudhary (starmariner027@gmail.com)


