# Dataset Quality Cleaning and Repair

This directory contains automated image quality inspection, in-place JPEG end-marker repair, corrupted/blurred frame isolation, and 5-fold ensemble 1X/10X magnification classification utilities.

## Documentation

Full operational workflows, command-line arguments, defect detection math, and threshold tables are documented in the [`docs/`](../docs/README.md) hub:

- **Quality Cleaning and Repair Workflow**: [Tools and Workflows Guide - Section 1](../docs/tools-and-workflows.md#1-dataset-quality-cleaning-and-repair-dataset_cleaner)
  - Single-image, station-level, and root dataset inspection commands.
  - In-place stream repair appending missing `0xFFD9` EOI markers (`--repair-jpeg`).
  - Spatial side-patch Laplacian blur calculation (excluding central pole columns).
  - 5-fold vision ensemble magnification audit and safe folder relocation.
- **Dataset Structure & Integrity Rules**: [Dataset and Preprocessing Guide](../docs/dataset-and-preprocessing.md)
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `clean_dataset.py` | Standalone script for corrupted JPEG repair, exposure checks, and side-patch blur detection | [Tools Guide](../docs/tools-and-workflows.md#image-quality-cleaning-clean_datasetpy) |
| `train_image_sorter.py` | Trains 5-fold stratified vision ensemble on verified trusted folders for 1X vs 10X separation | [Tools Guide](../docs/tools-and-workflows.md#magnification-sorting-ensemble) |
| `image_prediction_sorter.py` | Audits candidate folders and safely moves verified misfiled images to sibling directories | [Tools Guide](../docs/tools-and-workflows.md#magnification-sorting-ensemble) |
| `config.yaml` | Configuration for trusted folders, ensemble architectures, and voting thresholds | [Tools Guide](../docs/tools-and-workflows.md#magnification-sorting-ensemble) |
