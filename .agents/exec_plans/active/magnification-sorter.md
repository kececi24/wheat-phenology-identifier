# Magnification sorter

Last updated: 2026-11-10
Status: five-fold audit and CSV apply implemented; production validation in progress

## Sources and handoff

[Task](thread://01a00057-7a69-7483-9a36-d53109f89540?hostId=local) ·
[Trainer](../../../dataset_cleaner/train_image_sorter.py) ·
[Config](../../../dataset_cleaner/config.yaml) ·
[Single-image predictor](../../../dataset_cleaner/image_prediction_sorter.py)

- Trusted folders supply 1X/10X labels; filenames are independent evidence.
  DeiT-Tiny uses RGB, a whole-image 224×224 resize, tensor conversion, and
  processor normalization. Station-year GroupKFold keeps nearby captures
  together; each fold saves its best validation-loss checkpoint and reports.
- The original three-fold weights remain under `magnification_model`.
  The five-fold YAML writes to `magnification_ensemble_model` and publishes
  the saved config after training completes. Fold summaries now report the
  metrics from the saved epoch.
- The audit writes per-fold votes and confidence, the folder label, the
  filename label, and proposed sibling-folder moves to a CSV. The default
  threshold is 3/5. Trusted folders are protected in both directions. The
  apply command uses reviewed MOVE rows and preserves basenames. It skips
  changed sources and occupied destinations.
- Scratch tests verified voting, CSV apply, stale-source detection, and
  destination collision handling. The existing three folds also returned
  finite scores on a sample image. A separate CUDA smoke run trained five
  folds on ten copied trusted images and audited all ten successfully.
  K1/K2 classification remains separate.

## Immediate next action

Verify all five production folds, then run the read-only audit and review its CSV.

## Preflight and blockers

Local PyTorch, CUDA, Transformers, Pillow, and the cached DeiT backbone are
available. The dataset has 4,375 trusted images and 41,694 scan candidates.
The production run saved the first epoch of fold 1 and continued into epoch 2.
The completion marker is written only after all five folds finish.

## Validation and completion criteria

Verify the five saved folds belong to the same completed run, then check
the audit's status totals and sample its proposed moves before applying.
