# DINOv3 PEFT capacity and loss

Last updated: 2026-11-10
Status: results analyzed; capacity/loss ablations pending

## Sources and handoff

[Task](thread://01a17d92-9799-7d20-ab45-a70cdac381c8?hostId=local) ·
[Adapter pipeline](../../../Adapter_BBCH/README.md) ·
[November 4 analysis](../../../results/adapter_block_analysis_2026-11-04/REPORT.md)

- The raw-image study compares frozen DINOv3, Q/V LoRA, a standard bottleneck,
  and a transition-initialized adapter, with shared pooling/head and no
  temporal model or metadata. The three label budgets used the same 16
  station-held-out folds and 2,605 test image-days. All 3,424 saved-artifact
  checks passed; this was an analysis of existing runs.
- Pooled macro-F1 at 25% / 50% / 100% labels: frozen **79.12 / 79.08 / 78.47%**;
  LoRA **79.65 / 78.91 / 79.08%**; standard **79.13 / 79.08 / 78.16%**;
  transition **78.94 / 78.95 / 77.83%**. Transition lost to frozen at each
  budget. LoRA's gains are small and inconsistent; the single-seed results
  and station-bootstrap intervals do not establish a reliable advantage.
- Next: hold folds, label subsets, and seeds fixed and compare frozen
  width-128 vs width-32 heads under hybrid loss vs soft-target CE
  (`ordinal_ce_weight: 1.0`). Also consider a mean-pooling/linear control:
  learned pooling accounts for about 401k trainable parameters, more than
  the width-128 head (101k). Keep LoRA as a comparison and defer temporal
  integration. Any seed replication should keep fold and label seeds fixed.

## Immediate next action

Prepare the width 128/32 × hybrid/soft-target CE ablation on the frozen baseline.

## Preflight and blockers

Before launching, confirm the raw-image adapter runtime, backbone weights, label snapshot, and fixed split/label seeds.

## Validation and completion criteria

Compare the four matched runs by validation macro-F1 and report held-out station results without selecting on test. Record any subsequent pooling or seed ablations separately.
