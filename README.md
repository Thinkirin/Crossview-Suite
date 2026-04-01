# CrossView Suite

Codebase for **CrossView Suite: Boosting Cross-view Spatial Intelligence of MLLMs with Dataset, Model and Benchmark**.

CrossView Suite studies cross-view spatial intelligence for multimodal large language models (MLLMs) through three coordinated components:

- **CrossViewSet**: a large-scale cross-view instruction dataset with mask grounding and object-level supervision.
- **CrossViewBench**: a scene-disjoint benchmark for evaluating correspondence, visibility, geometry, and physical reasoning across views.
- **CrossViewer**: a progressive **Perception -> Alignment -> Reasoning** framework for object-centric multi-view reasoning.

This repository currently contains the **CrossViewer** model code and training/evaluation scripts under [`CrossViewer/`](./CrossViewer). Suite-level dataset and benchmark assets can be organized under this repository as the project release expands.

## Highlights

- Explicit object-level cross-view alignment instead of relying only on implicit multi-image fusion
- Mask-grounded object representations for fine-grained multi-view reasoning
- Training and evaluation pipeline built around Qwen3-VL
- Configs for baseline, ablations, and global fusion variants

## Repository Layout

```text
CrossViewer-Suite/
├── README.md
└── CrossViewer/
    ├── configs/          # Training configs and ablation settings
    ├── crossviewer/      # Model definition and core modules
    ├── data/             # JSONL dataset loader and mask/object utilities
    ├── scripts/          # Training and evaluation entrypoints
    ├── run_train.sh
    ├── run_train_nohup.sh
    └── requirements.txt
```

## Current Scope

The current code snapshot focuses on the **CrossViewer** model implementation:

- Qwen3-VL-based multi-view reasoning model
- Scale-adaptive object tokenization and cross-view association
- JSONL-based training pipeline with inline mask decoding
- Multiple-choice evaluation script for validation experiments

Config path fields are resolved relative to each YAML file, so model, data, checkpoint, and log locations can stay in `configs/*.yaml` without hardcoding machine-specific absolute paths in the scripts.

## Environment

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- Qwen3-VL checkpoint
- Optional: `decord` for video-backed data loading
- Optional: `deepspeed` for large-scale training

## Installation

```bash
cd CrossViewer
pip install -r requirements.txt
pip install decord
# optional
pip install deepspeed
```

## Quick Start

1. Fill in `model.vision_encoder_path`, `data.data_root`, `data.jsonl_train`, and `data.jsonl_val` in `configs/*.yaml`.
2. Make sure the Qwen3-VL checkpoint path or model id and dataset JSONL paths are correct for your environment.
3. Launch training or evaluation from the `CrossViewer/` directory.

Example training:

```bash
cd CrossViewer
torchrun --nproc_per_node=4 --master_port=12355 scripts/train.py --config configs/default.yaml
```

Example evaluation:

```bash
cd CrossViewer
python scripts/eval_mc.py --config configs/default.yaml --ckpt /path/to/checkpoint
```

## Data Expectations

The training pipeline expects JSONL annotations configured in `configs/*.yaml`. The dataset loader supports:

- multi-view samples stored in JSONL format
- per-view images and object masks
- inline COCO-style RLE masks
- object-centric question answering supervision

## Paper Summary

According to the paper, CrossView Suite includes:

- **1.6M** training samples in **CrossViewSet**
- **17K** benchmark questions in **CrossViewBench**
- **17** fine-grained task types spanning correspondence, occlusion, geometry, and physical reasoning

CrossViewer is designed to improve cross-view spatial reasoning by combining mask-grounded perception, explicit alignment, and LLM-based reasoning.

## Status

- Current repository content mainly covers the model side: `CrossViewer`
- Dataset and benchmark release details can be added under this suite later
- Citation and release metadata can be finalized after the paper submission/publication stage
