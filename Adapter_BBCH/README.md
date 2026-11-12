# Adapter BBCH: Parameter-Efficient Fine-Tuning (PEFT)

This directory contains the parameter-efficient fine-tuning (PEFT) study on single raw crop images. It inserts lightweight trainable adapter modules (LoRA, AdaptFormer bottleneck, and Transition Prior) into the last two transformer blocks of a frozen DINOv3 ViT-B/16 backbone across 100%, 50%, and 25% label budgets without temporal modeling or metadata inputs.

## Documentation

Comprehensive architecture, method formulations, and benchmark comparisons are documented in the [`docs/`](../docs/README.md) hub:

- **PEFT Architecture and Methods**: [Baselines and Adapters Guide - Section 1](../docs/baselines-and-adapters.md#1-single-image-parameter-efficient-fine-tuning-adapter_bbch)
  - Frozen Baseline (`none`), LoRA (`lora`), AdaptFormer parallel bottleneck (`standard`), and SVD-derived biological Transition Prior (`transition`).
  - Label budget subsampling regimes (100%, 50%, 25%).
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `block_model.py` | DINOv3 block hooks and parallel adapter injection modules | [Baselines & Adapters Guide](../docs/baselines-and-adapters.md#target-architecture-and-insertion-points) |
| `adapter_model.py` | SVD orthonormal transition prior estimator and regularizer | [Baselines & Adapters Guide](../docs/baselines-and-adapters.md#evaluated-peft-methods) |
| `run_adapter_training.py` | Main training and evaluation runner across LOSO folds | [Baselines & Adapters Guide](../docs/baselines-and-adapters.md#label-budget-sampling) |
| `orchestrator.sh` | Sequential runner across full, 50%, and 25% budget configurations | [Baselines & Adapters Guide](../docs/baselines-and-adapters.md#label-budget-sampling) |
| `config/` | YAML configuration files (`config.yaml`, `config50.yaml`, `config25.yaml`) | [Baselines & Adapters Guide](../docs/baselines-and-adapters.md#label-budget-sampling) |
