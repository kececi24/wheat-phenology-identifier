# Classical 1X/10X Image Magnification Audit

This document is a navigation pointer for the classical computer vision magnification audit script (`audit_image_magnification.py`).

Complete execution workflows, command-line arguments, and classification status definitions are documented in the [`docs/`](../docs/README.md) hub:

- [Tools and Workflows Guide - Section 2: Classical Magnification Auditing](../docs/tools-and-workflows.md#2-classical-1x10x-magnification-auditing)
  - SIFT / AKAZE feature matching with RANSAC homography scale estimation.
  - Multi-worker parallel station audits and discrepancies-only reporting.
  - Verification status codes (`CONSISTENT`, `LIKELY_MISFILED`, `LIKELY_FILENAME_ERROR`, `FILENAME_FOLDER_MISMATCH`).
- [Central Documentation Hub](../docs/README.md)
