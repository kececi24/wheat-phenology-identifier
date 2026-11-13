# DINOv3 BBCH Linear Probe (Single-Day Ablation)

This directory contains the single-day non-temporal deep learning baseline. It extracts dense DINOv3 patch descriptors from a single target photo, applies hierarchical patch and tile attention pooling, directly concatenates thermal time (GDD) and coordinates, and classifies BBCH intervals via a linear head without temporal sequence modeling.

## Documentation

Full architectural specifications, ablation roles, and inference instructions are documented in the [`docs/`](../docs/README.md) hub:

- **Linear Architecture and Ablation Role**: [Baselines and Adapters Guide - Section 3](../docs/baselines-and-adapters.md#3-dinov3-linear-single-day-pipeline-dinov3_bbch_linear)
  - Dense DINOv3 ViT-B/16 feature extraction with unfrozen final block and norm.
  - Direct concatenation of visual embeddings ($d=768$) and 4 metadata scalars.
  - Single-day classification head ($\text{LayerNorm} \to \text{Dropout} \to \text{Linear}$).
  - Quantitative ablation role proving the benefit of temporal sequence modeling.
- **Inference Instructions**: [Tools and Workflows Guide - Section 5.3](../docs/tools-and-workflows.md#3-single-day-linear-inference-dinov3_bbch_linearinfer_single_imagepy)
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `linear_phenology.py` | Single-image dataset, metadata concatenation, and linear classifier architecture | [Baselines Guide](../docs/baselines-and-adapters.md#3-dinov3-linear-single-day-pipeline-dinov3_bbch_linear) |
| `run_linear_training.py` | Training and evaluation runner for single-day LOSO folds | [Baselines Guide](../docs/baselines-and-adapters.md#3-dinov3-linear-single-day-pipeline-dinov3_bbch_linear) |
| `infer_single_image.py` | Standalone single-image inference helper from exported checkpoints | [Tools Guide](../docs/tools-and-workflows.md#3-single-day-linear-inference-dinov3_bbch_linearinfer_single_imagepy) |
