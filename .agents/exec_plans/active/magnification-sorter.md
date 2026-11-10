# Magnification sorter

Last updated: 2026-11-10
Status: training trial reported working; inference in progress

## Sources and handoff

[Task](thread://01a00057-7a69-7483-9a36-d53109f89540?hostId=local) ·
[Trainer](../../../dataset_cleaner/train_image_sorter.py) ·
[Config](../../../dataset_cleaner/config.yaml) ·
[Single-image predictor](../../../dataset_cleaner/image_prediction_sorter.py)

- Trusted folders supply 1X/10X labels; filenames are independent evidence.
  DeiT-Tiny uses RGB, a whole-image 224×224 resize, tensor conversion, and
  processor normalization. Station-year GroupKFold keeps nearby captures
  together; each fold saves its best validation-loss checkpoint and reports.
- The user reported the remote training trial worked. The local YAML still
  specifies 3 folds, 2 epochs, and patience 3; a complete five-model run has
  not been confirmed here. The local single-image predictor no longer
  references undefined AMP settings, but still has placeholder input paths.
- Next: smoke-test one actual fold and a verified image on the machine with
  the weights. Then verify five checkpoints from one run and vote on images
  outside all five training/validation sets. Compare the vote separately to
  the folder and filename, and report discrepancies for review. K1/K2
  classification remains separate and needs trusted camera references.

## Immediate next action

Smoke-test one fold on the machine with the saved weights, using a visually verified image.

## Preflight and blockers

Requires a usable PyTorch/Transformers/Pillow environment, one local fold checkpoint and processor, and a readable verified image. Full five-fold weights are not confirmed.

## Validation and completion criteria

The saved fold returns the expected 1X/10X label and a finite score using the training transform. Before folder scanning, test 3/5 majority voting and verify that the audit reports disagreements without moving images.
