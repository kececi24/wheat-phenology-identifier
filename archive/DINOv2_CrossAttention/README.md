# Historical DINOv2 Cross-Attention Dual-Camera Pipeline (Archived)

This directory contains the historical experiment utilizing bidirectional cross-attention mechanisms between 1X wide-angle and 10X close-up camera streams.

## Documentation and Findings

Empirical testing confirmed that cross-attention between camera views added model complexity and computational overhead without outperforming single-stream 10X micro models on held-out physical stations. Full comparative analysis is documented in the [`docs/`](../../docs/README.md) hub:

- [Baselines and Adapters Guide - Historical Comparisons](../../docs/baselines-and-adapters.md#historical-dinov2-pipeline-archivedinov2_bbch-variants)
- [Evolutionary Comparison Table](../../docs/baselines-and-adapters.md#5-architectural-progression-summary)
- [Central Documentation Hub](../../docs/README.md)

For the active production pipeline, see [`DINOv3_BBCH/`](../../DINOv3_BBCH/) and the [DINOv3 Temporal Pipeline Guide](../../docs/dinov3-pipeline.md).
