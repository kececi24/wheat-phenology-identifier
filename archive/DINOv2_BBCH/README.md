# Historical DINOv2 BBCH Phenology Pipeline (Archived)

This directory contains the historical DINOv2 BBCH temporal pipeline. It utilized frozen `facebook/dinov2-base` embeddings with tile-level attention pooling and a 31-day causal temporal Transformer.

## Documentation and Findings

Comprehensive documentation of the DINOv2 BBCH implementation, its role as the foundation vision predecessor, and why dense DINOv3 patch grids replaced single-vector tile pooling is available in the [`docs/`](../../docs/README.md) hub:

- [Baselines and Adapters Guide - Historical DINOv2 Pipeline](../../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants)
- [Evolutionary Comparison Table](../../docs/baselines-and-adapters.md#5-architectural-progression-summary)
- [Central Documentation Hub](../../docs/README.md)

For the active production pipeline, see [`DINOv3_BBCH/`](../../DINOv3_BBCH/) and the [DINOv3 Temporal Pipeline Guide](../../docs/dinov3-pipeline.md).
