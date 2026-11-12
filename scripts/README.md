# Repository Scripts and Verification Utilities

This directory contains standalone operational utilities for dataset auditing, phenology landmark curve extraction, and physical camera shift detection.

## Documentation

Full usage guides and operational commands are centralized in the [`docs/`](../docs/README.md) hub:

- **Auditing and Verification Workflows**: [Tools and Workflows Guide](../docs/tools-and-workflows.md)
  - Classical SIFT/RANSAC magnification auditing (`audit_image_magnification.py`).
  - Distributed multi-station verification and curve analysis.
- **Dataset Preprocessing & Quality Standards**: [Dataset and Preprocessing Guide](../docs/dataset-and-preprocessing.md)
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `audit_image_magnification.py` | Read-only classical SIFT/AKAZE + RANSAC homography magnification auditor | [Tools Guide](../docs/tools-and-workflows.md#2-classical-1x10x-magnification-auditing) |
| `verify_all_stations.py` | Extracts Green Chromatic Coordinate (GCC) / ExG curves and phenology landmarks | [Tools Guide](../docs/tools-and-workflows.md#1-excess-green-landmark-verification-verify_all_stationspy) |
| `analyze_details.py` | Audits phenological curve anomalies against milestone dates | [Tools Guide](../docs/tools-and-workflows.md#2-station-detail-analysis-analyze_detailspy) |
| `detect_shifts.py` | Detects physical camera rotation, servicing shifts, and wind drift | [Tools Guide](../docs/tools-and-workflows.md#3-camera-shift-detection-detect_shiftspy) |
| `summarize_all_verifications.py` | Aggregates station-level landmark verification summaries | [Tools Guide](../docs/tools-and-workflows.md#4-verification-summary-summarize_all_verificationspy) |
