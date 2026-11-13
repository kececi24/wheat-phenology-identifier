# Ground-Truth Milestone Labels and Phenological Annotations

This directory contains canonical milestone dates and developmental stage ground-truth annotations for the wheat phenology identification pipeline across all monitored field stations.

## Documentation

Full schema details, milestone interval definitions, biological BBCH stage mappings, and soft transition target formulations are documented in the central [`docs/`](../docs/README.md) hub:

- [Dataset and Preprocessing Guide - Section 5: ISO Label Files and Metadata Schema](../docs/dataset-and-preprocessing.md#5-iso-label-files-and-metadata-schema)
- [Dataset and Preprocessing Guide - Section 4: BBCH Milestone Interval Definitions](../docs/dataset-and-preprocessing.md#4-bbch-milestone-interval-definitions)
- [Central Documentation Hub](../docs/README.md)

## Files Overview

| File | Format | Description | Documentation Reference |
| :--- | :--- | :--- | :--- |
| [`labeling_bbch_iso_dates.csv`](labeling_bbch_iso_dates.csv) | CSV | **Canonical reference table** containing 15 physical stations (24 harvest seasons) with ISO-8601 (`YYYY-MM-DD`) milestone dates, duration audit totals, preferred camera IDs, and expert field comments. | [Metadata Schema](../docs/dataset-and-preprocessing.md#table-schema-and-columns) |
| [`labeling_bbch_iso_dates.xlsx`](labeling_bbch_iso_dates.xlsx) | Excel (`.xlsx`) | Formatted workbook of the canonical milestone dates including Excel date formatting, duration audit formulas, and observational comments. | [Label Files](../docs/dataset-and-preprocessing.md#5-iso-label-files-and-metadata-schema) |
| [`Fenolojik_Evreler_v3.xlsx`](Fenolojik_Evreler_v3.xlsx) | Excel (`.xlsx`) | Historical raw Turkish field observation spreadsheet containing early unparsed date records and field notations. Retained for historical provenance. | [Baselines Guide](../docs/baselines-and-adapters.md#4-historical-pipeline-comparisons-archive) |

## Usage

In training and evaluation workflows, pass the canonical CSV path to `--label-path`:

```bash
# Verify station-year coverage and stage distribution
python DINOv3_BBCH/metadata_sanity_check.py \
  --label-path labels/labeling_bbch_iso_dates.csv \
  --data-path data \
  --camera AUTO \
  --stream micro
```
