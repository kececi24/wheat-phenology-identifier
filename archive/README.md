# Historical Research Archives

This directory preserves earlier model generations and experimental prototypes developed during project evolution. These pipelines are retained for scientific provenance and baseline reproducibility.

For the active production pipeline, see [`DINOv3_BBCH/`](../DINOv3_BBCH/) and the [DINOv3 Temporal Pipeline Guide](../docs/dinov3-pipeline.md).

## Documentation

A comprehensive comparative retrospective and evolutionary analysis of these archived pipelines is available in the documentation hub:

- **Historical Model Comparisons**: [Baselines and Adapters Guide - Section 4](../docs/baselines-and-adapters.md#4-historical-pipeline-comparisons-archive)
  - 2D CNN backbones (`CNN/`) and station-overfitting failure modes.
  - Spatiotemporal 3D CNNs (`CNN3D/`) and memory/regularization trade-offs.
  - DINOv2 single-tile baseline (`DINOv2_BBCH/`) and dual-camera fusion findings (`DINOv2_Gated/`, `DINOv2_CrossAttention/`).
  - Architectural progression summary table.
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Archived Experiment Directories

| Directory | Architecture Paradigm | Primary Finding / Role | Documentation Reference |
| :--- | :--- | :--- | :--- |
| `CNN/` | 2D CNN (EfficientNetV2, ConvNeXt) + Causal Transformer | Overfit heavily to background soil and mounting structures | [Baselines Guide](../docs/baselines-and-adapters.md#historical-2d-cnn-archivecnn) |
| `CNN3D/` | Spatiotemporal 3D CNN on raw video volume | High VRAM footprint; fragile regularizer balance | [Baselines Guide](../docs/baselines-and-adapters.md#historical-3d-cnn-archivecnn3d) |
| `DINOv2_BBCH/` | Frozen DINOv2-Base + Tile Attention + Causal Transformer | Proved foundation vision superiority over CNNs | [Baselines Guide](../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants) |
| `DINOv2_BBCH_Gated/` | Dual-camera (1X macro + 10X micro) gated fusion | Proved 1X stream adds noise without gain | [Baselines Guide](../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants) |
| `DINOv2_CrossAttention/` | Dual-stream cross-attention mechanism | Complex cross-attention did not exceed 10X micro | [Baselines Guide](../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants) |
| `DINOv2_Gated/` | Early dual-camera gated fusion pilot | Predecessor to DINOv2 BBCH gated pipeline | [Baselines Guide](../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants) |
| `DINOv2/` | Initial DINOv2 exploration with legacy label mapping | Earliest foundation model prototype | [Baselines Guide](../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants) |
| `ViT2/`, `ViT3/` | Early Vision Transformer (DeiT/ViT) prototypes | Early Transformer exploration before foundation DINO | [Baselines Guide](../docs/baselines-and-adapters.md#4-historical-pipeline-comparisons-archive) |
| `old/VTMM/` | Multiscale vision and temporal prototype | Legacy research prototype | [Baselines Guide](../docs/baselines-and-adapters.md#4-historical-pipeline-comparisons-archive) |

## Archived Data and Report Artifacts

| File | Description | Documentation Reference |
| :--- | :--- | :--- |
| `bad_images_report.csv` | Initial dataset quality inspection report recording blurred, dark, and defective frames | [Tools Guide](../docs/tools-and-workflows.md#1-dataset-quality-cleaning-and-repair-dataset_cleaner) |
| `labeling.xlsx` | Intermediate historical labeling table with serial date formats | [Dataset Guide](../docs/dataset-and-preprocessing.md#5-iso-label-files-and-metadata-schema) |
| `labeling_iso_dates.xlsx`, `labeling_iso_dates_new.xlsx` | Earlier iteration workbooks preceding final canonical 15-station ISO table | [Dataset Guide](../docs/dataset-and-preprocessing.md#5-iso-label-files-and-metadata-schema) |
| `labeling_iso_dates.csv` | Early draft ISO date table with formula evaluation references | [Dataset Guide](../docs/dataset-and-preprocessing.md#5-iso-label-files-and-metadata-schema) |
