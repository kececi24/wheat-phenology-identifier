# Historical 2D CNN Phenology Pipeline (Archived)

This directory contains the historical 2D Convolutional Neural Network (CNN) pipeline, utilizing Torchvision encoders (EfficientNetV2, ConvNeXt) coupled with a causal temporal Transformer over 31-day observation windows.

## Documentation and Findings

Detailed analysis of the 2D CNN experiment and its empirical failure modes (severe station-level background overfitting) is documented in the [`docs/`](../../docs/README.md) hub:

- [Baselines and Adapters Guide - Historical 2D CNN](../../docs/baselines-and-adapters.md#historical-2d-cnn-archivecnn)
- [Evolutionary Comparison Table](../../docs/baselines-and-adapters.md#5-architectural-progression-summary)
- [Central Documentation Hub](../../docs/README.md)

For the active production system, see [`DINOv3_BBCH/`](../../DINOv3_BBCH/) and the [DINOv3 Temporal Pipeline Guide](../../docs/dinov3-pipeline.md).
