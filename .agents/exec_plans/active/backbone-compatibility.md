# Backbone/config compatibility

Last updated: 2026-11-10
Status: config routing confirmed; runs pending

## Sources and handoff

[Task](thread://01a1a115-c60c-7d82-aaa7-ef00b7bcdb1a?hostId=local) ·
[Temporal pipeline README](../../../DINOv3_BBCH/README.md)

- Both LOSO launchers forward `--config-dir`; select one directory per
  invocation and optionally choose YAML stems with `--configs`. Discovery
  is nonrecursive. Configs require `pipeline`, `finetune`, `precompute`,
  and `training`; use separate result roots. Relative paths are resolved
  from the repository root. Local comparison folders include `vit-base`,
  `deits`, and `beit`.
- Dense extraction requires a supported spatial-token layout and compatible
  input/patch sizes. DINOv2, original DINO, standard ViT, non-distilled DeiT,
  and BEiT are candidates from inspection, not confirmed end-to-end here.
  Distilled DeiT and models with a different prefix or no CLS need special
  handling. Global descriptors can come from the pooler; disabling dense
  precompute does not change the fine-tuning script's dense forward path.
- Next: dry-run each config directory, then smoke-test extraction and
  fine-tuning on one fold before committing a full comparison. Changing the
  backbone requires a new cache and temporal checkpoint.

## Immediate next action

Dry-run the local vit-base, deits, and beit config directories; select one fold for a smoke test.

## Preflight and blockers

Dry-run checks config routing only. A live smoke requires accessible checkpoint weights, compatible runtime, data, and a separate result root per model.

## Validation and completion criteria

For each candidate, verify the input/patch shape, spatial-token order, finite features, fine-tuning block discovery, and cache/checkpoint compatibility before launching full LOSO.
