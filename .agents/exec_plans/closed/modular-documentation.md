# Modular Repository Documentation ExecPlan

## Goals
Create a comprehensive, modular documentation suite for the wheat-phenology-identifier repository in `docs/`.
Each document must cover a specific topic/feature/work, be concise yet detailed and self-contained, and cross-reference related components.

## Target Structure
- `docs/README.md`: Central documentation portal, index, repo architecture diagram, quickstart.
- `docs/project-overview.md`: Phenological problem formulation, causal daily prediction, BBCH scale, system architecture.
- `docs/dataset-and-preprocessing.md`: Station layout, cameras (K1/K2, AUTO), close-up 10X micro stream, BBCH milestone tables, image verification and SIFT audit.
- `docs/dinov3-pipeline.md`: Tiling, DINOv3 backbone, hierarchical patch/tile attention, precomputed embedding caches, causal temporal Transformer, gated metadata fusion (days-since-sowing, weather/GDD, location).
- `docs/training-and-evaluation.md`: LOSO vs random fold strategy, fold-safe fine-tuning, leakage prevention rules, hybrid loss, metrics, monotonic Viterbi sequence decoding.
- `docs/baselines-and-adapters.md`: Single-image Adapter_BBCH PEFT, Baseline_BBCH ExG, DINOv3_BBCH_Linear probe, archive baselines (DINOv2, CNN, CNN3D).
- `docs/tools-and-workflows.md`: Dataset cleaner, magnification audit scripts, distributed FSDP2 execution, single-image and window inference.

## Verification & Acceptance
- All documents exist under `docs/`.
- Content is technically precise and aligns with `.agents/summary.md` and repo source code.
- Swarm verifier checks cross-references, technical consistency, and formatting.

## Outcome
Successfully authored and verified the complete modular documentation suite in `docs/`:
- `docs/README.md`
- `docs/project-overview.md`
- `docs/dataset-and-preprocessing.md`
- `docs/dinov3-pipeline.md`
- `docs/training-and-evaluation.md`
- `docs/baselines-and-adapters.md`
- `docs/tools-and-workflows.md`
Verified by the swarm verifier agent. All files meet modularity, technical depth, and conciseness criteria.

