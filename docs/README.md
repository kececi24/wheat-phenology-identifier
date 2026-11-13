# Wheat Phenology Identifier Documentation

Welcome to the central documentation portal for the **Wheat Phenology Identifier** project. This repository implements an end-to-end computer vision and deep learning system designed for daily in-field causal wheat phenological stage classification from in-situ fixed agricultural cameras.

The project centers on the **BBCH scale** (Biologische Bundesanstalt, Bundessortenamt und CHemische Industrie) and emphasizes strict temporal causality, high spatial resolution via tiling, foundation vision models (DINOv3), hierarchical attention pooling, and leak-free Leave-One-Station-Out (LOSO) cross-validation.

---

## Documentation Index

The documentation is organized into modular guides covering specific components of the system:

| Guide | Description |
| --- | --- |
| [Project Overview](project-overview.md) | High-level problem statement, agricultural context, BBCH scale milestone progression, causal daily formulation, and core architectural design decisions. |
| [Dataset and Preprocessing](dataset-and-preprocessing.md) | Station filesystem hierarchy, camera streams (1X macro vs 10X micro), label parsing, transition soft targets, and magnification verification. |
| [DINOv3 Temporal Pipeline](dinov3-pipeline.md) | High-resolution image tiling (224x224 stride 112), ViT-B/16 DINOv3 backbone, 2x2 dense patch grid, hierarchical attention pooling, causal temporal Transformer (21/31 days), and gated metadata fusion. |
| [Training and Evaluation](training-and-evaluation.md) | Leave-One-Station-Out (LOSO) protocol, fold-safe backbone fine-tuning, training-only coverage filtering, hybrid ordinal loss, validation checkpointing, and monotonic Viterbi decoding. |
| [Baselines and Adapters](baselines-and-adapters.md) | Parameter-efficient fine-tuning (LoRA, AdaptFormer, Transition Prior in `Adapter_BBCH`), tabular metadata + ExG baseline (`Baseline_BBCH`), single-day linear probe (`DINOv3_BBCH_Linear`), and historical CNN/3D CNN/DINOv2 comparisons. |
| [Tools and Workflows](tools-and-workflows.md) | Practical utilities: dataset cleaning, SIFT/RANSAC magnification auditing, distributed FSDP2 execution, single-image/window inference scripts, and assisted milestone labeling. |

---

## Repository Layout

The repository is structured into active research/production pipelines, shared utilities, and historical archives:

```text
wheat-phenology-identifier/
├── DINOv3_BBCH/             # Maintained causal temporal DINOv3 pipeline (primary system)
│   ├── configs/             # Flat YAML configurations for standard and LOSO training
│   ├── finetune_dinov3_backbone.py # Leakage-safe backbone adaptation
│   ├── finetuned_loso_runner.py    # Multi-stage LOSO orchestration
│   ├── precompute_multiscale_embeddings.py # High-throughput dense feature cache builder
│   ├── run_multiscale_training.py  # Temporal Transformer training and evaluation
│   ├── multiscale_phenology.py     # Core models, datasets, transforms, and loss functions
│   └── infer_cached_window.py      # Production deployment inference from cached windows
├── Adapter_BBCH/            # Single-image parameter-efficient fine-tuning (PEFT) study
│   ├── config/              # YAML configs for full, 50%, and 25% label budgets
│   ├── block_model.py       # DINOv3 transformer-block wrappers (LoRA, AdaptFormer, Transition)
│   ├── adapter_model.py     # Training-only transition prior estimation via SVD
│   └── run_adapter_training.py # Single-image PEFT training runner
├── Baseline_BBCH/           # Lightweight non-deep-learning baseline (ExG + tabular metadata)
│   ├── config.yaml          # Default configuration for multiprocessing baseline
│   └── run_metadata_exg_training.py # HistGradientBoostingClassifier runner with rolling stats
├── DINOv3_BBCH_Linear/      # Non-temporal single-day DINOv3 linear probe
│   ├── linear_phenology.py  # Dense feature extraction + metadata concatenation + linear head
│   └── run_linear_training.py # Single-day fine-tuning and evaluation runner
├── dataset_cleaner/         # Data inspection, bad image filtering, and reorganization tools
├── scripts/                 # Maintenance scripts (e.g., SIFT/RANSAC magnification audit)
├── docs/                    # Modular documentation guides (this directory)
├── archive/                 # Preserved historical experiments (CNN, CNN3D, DINOv2, ViT)
├── labeling_bbch_iso_dates.csv # Canonical 15-station BBCH milestone date annotations
└── labeling_bbch_iso_dates.xlsx# Excel source with milestone dates and field notes
```

---

## Quickstart Guide

### 1. Environment Setup

Ensure Python 3.10+ is available with PyTorch (CUDA recommended for training and embedding caching):

```bash
# Clone the repository
git clone <repo-url>
cd wheat-phenology-identifier

# Install core dependencies
pip install -r requirements.txt
pip install -r DINOv3_BBCH/requirements.txt
```

> [!NOTE]
> Hugging Face access to Meta's DINOv3 models (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is gated. Accept the model license on Hugging Face and log in via `huggingface-cli login` or supply your token via the `HF_TOKEN` environment variable.

### 2. Sanity Check Metadata and Station Coverage

Before running precomputations or training, verify that image paths, timestamps, camera streams, and BBCH milestone intervals align across all stations:

```bash
python DINOv3_BBCH/metadata_sanity_check.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --camera AUTO \
  --stream micro
```

This utility checks available station-years, verifies camera availability (K1/K2), confirms date ranges, and warns of any missing files or corrupt headers.

### 3. Precompute Dense DINOv3 Embeddings

For fast temporal Transformer training, precompute dense patch and tile descriptors using the frozen DINOv3 ViT-B/16 backbone:

```bash
python DINOv3_BBCH/precompute_multiscale_embeddings.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov3_bbch_cache \
  --image-backbone facebook/dinov3-vitb16-pretrain-lvd1689m \
  --camera AUTO \
  --stream micro \
  --tile-streams micro \
  --tile-pooling attention \
  --tile-size 224 \
  --tile-stride 112 \
  --max-tiles 0 \
  --dense-features \
  --dense-grid-size 2 \
  --dense-include-cls \
  --embedding-dtype float16 \
  --batch-size 256 \
  --image-batch-size 16 \
  --num-workers 16 \
  --prefetch-factor 4 \
  --device cuda
```

This caches compact 2x2 dense patch grids plus CLS tokens (5 descriptors per 224x224 tile) in FP16 (~5.4 GB for the complete dataset).

### 4. Run Leave-One-Station-Out (LOSO) Training

Train the causal temporal Transformer across balanced Leave-One-Station-Out folds using the precomputed embedding cache:

```bash
python DINOv3_BBCH/run_multiscale_training.py \
  --config DINOv3_BBCH/configs/train_loso.yaml \
  --embedding-cache results_dinov3_bbch_cache/vit_embeddings.pt \
  --out-dir results_dinov3_loso \
  --device cuda
```

Key training parameters configured in [`configs/train_loso.yaml`](../DINOv3_BBCH/configs/train_loso.yaml) (and variants such as [`configs/train_loso_dsp.yaml`](../DINOv3_BBCH/configs/train_loso_dsp.yaml)):
- **Causal window**: 21 days (`--window-days 21`, `--window-mode causal`).
- **Loss**: Hybrid loss (`--loss hybrid` = soft label cross-entropy + ordinal CDF loss with weight 0.5).
- **Metadata fusion**: Gated residual MLPs (`--temporal-feature-fusion gated` with `--use-days-since-planting` and `--use-weather-metadata`) for days since sowing and cumulative GDD.
- **Checkpoint selection**: Best model selected strictly by validation macro-F1 (`--checkpoint-metric macro_f1`).

### 5. Run Single-Day or Causal Window Inference

To predict the phenological stage for an in-field crop photo on a given target date:

```bash
python DINOv3_BBCH/infer_cached_window.py \
  --checkpoint results_dinov3_loso/fold_1/best_model.pt \
  --image-path data/02.02/2015/K1/10X/02_02-2015_05_15-10_00-K1-10X.jpeg \
  --image-dir data/02.02/2015/K1/10X \
  --target-date 2015-05-15 \
  --planting-date 2014-10-20 \
  --station-code 02.02 \
  --device cuda
```

The script extracts the preceding 20 days of causal history from `--image-dir`, applies the hierarchical patch and tile pooler, fuses normalized days-since-sowing and GDD, and outputs predicted BBCH probabilities and the argmax class.

---

## Reproducibility and Experiment Principles

To maintain scientific integrity across all experiments:
1. **Station-Disjoint Splits**: Folds must always be grouped by physical station (`--fold-group-by station`). Never split randomly across individual dates or overlapping causal windows.
2. **Leakage Prevention**: Feature normalizations, transition priors, and training-only coverage filters must be computed exclusively from the training stations of each fold.
3. **Validation-Only Selection**: Model checkpoints, learning rate schedules, and decoding hyperparameters must be selected on validation folds before evaluating held-out test stations.
4. **Primary Metrics**: Report exact accuracy and macro-F1 as primary metrics. Soft date-window scores (e.g., 3-day or 5-day tolerance) provide secondary biological context.

---

## References and Visual Assets

- **Pipeline Architecture Schematic**: [Raster PNG](figures/dinov3_bbch_network_pipeline.png) | [Vector PDF](figures/dinov3_bbch_network_pipeline.pdf)
- **Foundational Literature Reference**: [Automated Image Labeling for Precision Agriculture: Scaling Crop Phenology (PDF)](references/Final%20Automated%20Image%20Labeling%20for%20Precision%20Agriculture%20Scaling%20Crop%20Phenology.pdf)

