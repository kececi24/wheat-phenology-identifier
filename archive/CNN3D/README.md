# Historical 3D CNN Phenology Pipeline (Archived)

This directory contains the historical spatiotemporal 3D Convolutional Neural Network (3D CNN) pipeline, which treated multi-week daily canopy sequences as short video tensors $[B, C, T, H, W]$ ($T = 15 \dots 31$ days).

## Documentation and Findings

Detailed analysis of the 3D CNN experiment and its empirical failure modes (high GPU memory footprint and unstable regularizer trade-offs) is documented in the [`docs/`](../../docs/README.md) hub:

- [Baselines and Adapters Guide - Historical 3D CNN](../../docs/baselines-and-adapters.md#historical-3d-cnn-archivecnn3d)
- [Evolutionary Comparison Table](../../docs/baselines-and-adapters.md#5-architectural-progression-summary)
- [Central Documentation Hub](../../docs/README.md)

For the active production system, see [`DINOv3_BBCH/`](../../DINOv3_BBCH/) and the [DINOv3 Temporal Pipeline Guide](../../docs/dinov3-pipeline.md).
