# DINOv3 BBCH Temporal Pipeline

This directory contains the primary production causal temporal deep learning pipeline for daily wheat phenological stage classification using Meta DINOv3 ViT-B/16 representations, dense patch descriptors, hierarchical attention pooling, and gated multimodal metadata fusion.

## Documentation

Comprehensive architecture, training, and operational documentation is centralized in the [`docs/`](../docs/README.md) hub:

- **Architecture & Modeling**: [DINOv3 Temporal Pipeline Guide](../docs/dinov3-pipeline.md)
  - Overlapping tiling (224x224, stride 112), dense 2x2 patch grids, hierarchical patch & tile attention, causal temporal Transformer (21-day window), and gated residual metadata fusion.
- **Training Protocol**: [Training and Evaluation Guide](../docs/training-and-evaluation.md)
  - Leave-One-Station-Out (LOSO) cross-validation, fold-safe backbone fine-tuning, hybrid ordinal CDF loss, and monotonic Viterbi decoding.
- **Inference & Utilities**: [Tools and Workflows Guide](../docs/tools-and-workflows.md)
  - Single-image and causal temporal cached-window deployment scripts.
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `multiscale_phenology.py` | Core model architectures, attention poolers, datasets, and hybrid loss functions | [Pipeline Guide](../docs/dinov3-pipeline.md) |
| `precompute_multiscale_embeddings.py` | High-throughput dense feature cache builder (float16) | [Pipeline Guide](../docs/dinov3-pipeline.md#6-embedding-cache-format-and-layout) |
| `run_multiscale_training.py` | Main training and evaluation runner for temporal Transformer | [Training Guide](../docs/training-and-evaluation.md#1-cross-validation-leave-one-station-out-loso-vs-random-folds) |
| `finetune_dinov3_backbone.py` | Fold-safe backbone fine-tuning unfreezing block 11 and final norm | [Training Guide](../docs/training-and-evaluation.md#2-fold-safe-backbone-fine-tuning-and-leakage-prevention) |
| `finetuned_loso_runner.py` | Multi-stage LOSO orchestration (fine-tune -> cache -> train) | [Training Guide](../docs/training-and-evaluation.md#automated-3-stage-pipeline-per-fold) |
| `run_finetuned_loso.sh` | Shell launcher for automated LOSO experiment sweeps | [Training Guide](../docs/training-and-evaluation.md#automated-3-stage-pipeline-per-fold) |
| `infer_cached_window.py` | Causal deployment inference from historical image sequence | [Tools Guide](../docs/tools-and-workflows.md#2-realistic-cached-window-causal-deployment-infer_cached_windowpy) |
| `metadata_sanity_check.py` | Station coverage, date interval, and camera availability auditor | [Tools Guide](../docs/tools-and-workflows.md) |
| `configs/` | Flat YAML configuration files for standard and LOSO training | [Training Guide](../docs/training-and-evaluation.md) |
