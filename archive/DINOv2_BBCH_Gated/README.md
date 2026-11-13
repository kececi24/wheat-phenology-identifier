# Historical DINOv2 BBCH Dual-Camera Gated Pipeline (Archived)

This directory contains the historical dual-camera experiment fusing 1X wide-angle canopy imagery with 10X micro close-up imagery via learned residual gating for BBCH stage classification.

## Documentation and Findings

Extensive empirical evaluations demonstrated that the dual-camera setup did not improve generalization over the 10X micro stream alone, as the 1X stream introduced uninformative sky, horizon, and installation variations. Full details are documented in the [`docs/`](../../docs/README.md) hub:

- [Baselines and Adapters Guide - Dual-Camera Findings](../../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants)
- [Project Overview - Single Micro Stream Focus](../../docs/project-overview.md#1-close-up-10x-micro-stream-focus)
- [Central Documentation Hub](../../docs/README.md)

For the active production pipeline, see [`DINOv3_BBCH/`](../../DINOv3_BBCH/) and the [DINOv3 Temporal Pipeline Guide](../../docs/dinov3-pipeline.md).
