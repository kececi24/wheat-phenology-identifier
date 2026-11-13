# Wheat Phenology Project Handoff

Last updated: 2026-11-10

## Start Here

Read [AGENTS.md](AGENTS.md) and the [current plan](current_plan.md) before
resuming work. Immediate next steps live in the current plan, and durable
workstream context lives under [active ExecPlans](exec_plans/active/).
Completed workstreams are recorded under [closed ExecPlans](exec_plans/closed/).
This summary keeps the project's shared technical and historical reference.

`DINOv3_BBCH/` is the maintained temporal pipeline; `Adapter_BBCH/` is the
single-image adapter study. `Baseline_BBCH/` and `DINOv3_BBCH_Linear/` remain
separate comparisons. Legacy DINOv2, ViT, and CNN baselines are under
`archive/`. Use the current pipeline README and YAML files for launch commands.

## Historical reference

The remaining sections preserve earlier designs, commands, and results.
July-era counts, paths, and verification statements describe those runs;
the current plan, active ExecPlans, and current READMEs take precedence.

## Project Goal

Predict daily wheat phenological stage from fixed-camera crop images. Deployment
must be causal: predictions for a target day may use that day and previous days,
but never future images.

The strongest tested approach is:

1. Use 10X close-up images (`micro` stream).
2. Tile each full-resolution image to retain small plant structures.
3. Encode tiles with `facebook/dinov3-vitb16-pretrain-lvd1689m`. The established
   reference keeps it frozen; the new fold-safe experiment fine-tunes only the
   final backbone block on each fold's training stations before caching.
4. Exclude DINOv3 register tokens and retain a compact 2x2 patch grid plus CLS
   for each tile.
5. Learn hierarchical attention: dense patches to tile, then tiles to daily image.
6. Feed a causal daily-embedding sequence to a temporal Transformer. The current
   strongest station-complete experiment uses 21 days; 31 days remains a useful
   controlled baseline.
7. Fuse days-since-sowing through a gated residual MLP.
8. Optionally fuse weather/GDD and latitude/longitude/elevation as separately
   controlled metadata experiments.
9. Train with hybrid soft-label cross entropy plus ordinal CDF loss.
10. Select checkpoints using validation macro-F1 only, then evaluate test once.

## Current BBCH Labels

Canonical label files:

- `labels/labeling_bbch_iso_dates.csv`
- `labels/labeling_bbch_iso_dates.xlsx`

The model derives intervals from these milestone columns:

| Class | Interval |
| --- | --- |
| `BBCH0` | sowing to emergence |
| `BBCH1` | emergence to tillering |
| `BBCH2` | tillering to stem elongation |
| `BBCH3` | stem elongation to heading |
| `BBCH5` | heading to flowering |
| `BBCH6_7` | flowering to maturity |
| `BBCH8` | maturity to harvest |

`OffSeason` is optional. The recommended in-season experiment uses
`--exclude-offseason`; OffSeason rows are then retained as causal context but are
not prediction targets.

Important label behavior:

- Class order is ordinal and must not be alphabetically sorted.
- Transition days use soft targets (currently 60/40 at a boundary).
- Numeric duration columns in the label table are audit information; milestone
  dates define labels.
- A row previously marked as having poor photographs may still exist. Verify the
  table before each definitive run rather than silently filtering validation/test.

## Data Layout and Camera Selection

Expected image layout:

```text
data/<station>/<year>/K1|K2/1X|10X/*.jpeg
```

Example:

```text
data/02.02/2015/K1/10X/02_02-2015_03_04-10_00-K1-10X.jpeg
```

Current behavior:

- `--camera AUTO` reads the label table's `kamera`/camera value when present.
- If no camera is specified, it prefers K1 and falls back to an available camera.
- Stations that begin with K2 are supported.
- `--stream micro` uses 10X only and matches the expected deployment setup.
- Missing dates are represented by masked temporal positions. Nearest-image
  fallback does not count as exact window coverage.
- Corrupt images are handled defensively in the current pipelines, but data should
  still be manually audited.

Always run the metadata sanity check before caching or training:

```bash
python DINOv2_BBCH/metadata_sanity_check.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --camera AUTO \
  --stream micro
```

Inspect class counts, station-years, camera choices, date ranges, and path coverage.

### Classical 1X/10X audit

`scripts/audit_image_magnification.py` is a read-only magnification audit for
new or mixed datasets. It checks filename/folder agreement, then uses SIFT
matching, RANSAC homographies, and same-scale registration against nearby trusted
images from both streams. It writes a CSV with evidence and never moves files.
Usage and status definitions are in `scripts/IMAGE_MAGNIFICATION_AUDIT.md`.

A verified `02.03`/2015 pilot processed 916 images. It correctly distinguished
two likely filename errors from a separate strongly supported folder error: the
file `02_03-2015_12_19-10_00-K1-10X.jpeg` in `K1/10X` matched the same-day 1X
frame with 653 RANSAC inliers and estimated scale 1.000. Review audit output
before changing any files, and rerun embedding caches after approved moves.

## Architecture Details

### Image encoder

- Stable baseline backbone: frozen `facebook/dinov2-base`.
- Current leading backbone: frozen
  `facebook/dinov3-vitb16-pretrain-lvd1689m`.
- Default tile size: 224 x 224.
- Recommended stride: 112 for 50% overlap.
- `tile-pooling attention` preserves all tile embeddings in the cache and trains a
  learned attention pooler.
- Tile positional information is included before attention pooling.
- Float16 embedding storage reduces RAM and disk usage.
- `DINOv3_BBCH` uses compact dense descriptors. The default 2x2 patch grid plus
  CLS stores five descriptors per tile. Patch attention is followed by tile
  attention, preserving more local crop structure than one global vector per tile.
- The completed DINOv3 Base dense cache was 5.4 GB. DINOv2 caches cannot be reused
  for DINOv3 because dimensions and cache structure differ.
- `DINOv3_BBCH/finetune_dinov3_backbone.py` provides leakage-safe, fold-specific
  image-backbone adaptation. It trains on the fold's training stations, selects
  the checkpoint using its validation stations, and evaluates the test station
  only after selection. The current recipe unfreezes the final DINOv3 block and
  final norm while retaining dense 2x2 patch features plus CLS.
- A fine-tuned backbone cache is fold-specific. Never use one backbone/cache for
  every LOSO fold: that would expose validation/test stations to supervised
  backbone updates. Metadata-only changes still do not require image recaching,
  but changing the backbone weights does.

### Temporal model

- Current leading tested window: 21 causal days (previous 20 plus target day).
  Retain 31 days as the established longer-context baseline and compare under
  identical LOSO folds before treating 21 days as universally superior.
- Recommended backend: temporal Transformer with CLS aggregation.
- LSTM/GRU alternatives exist but the short LSTM/weather experiment did not show a
  clear improvement over the Transformer. Do not assume it is superior without a
  controlled A/B test.
- The default `gated` metadata fusion preserves the full visual embedding. It adds
  learned residual projections instead of concatenating every scalar and
  compressing the visual representation through a small bottleneck.
- Days-since-sowing and weather use separate MLPs and separate sigmoid gates. Both
  gates default to `0.1`, allowing the visual/temporal representation to dominate
  initially while training decides whether each metadata source is useful.
- Metadata inputs are already normalized; the scalar cumulative-GDD branch does
  not apply input `LayerNorm`, which would erase a one-dimensional signal.
- `--temporal-feature-fusion legacy` recreates the old concatenation architecture
  and parameter names for old-checkpoint compatibility. Use `gated` for new runs.

### Metadata

- Days since sowing is normalized and supplied per temporal position through the
  calendar metadata MLP.
- Weather feature sets are explicit:
  - `cumulative` (default): normalized cumulative GDD only;
  - `daily_cumulative`: normalized daily GDD and cumulative GDD;
  - `full`: the legacy ten-value temperature/precipitation/GDD vector.
- Cumulative GDD is computed over the calendar timeline, including dates without
  images, rather than only summing observed image dates.
- Meteostat station-family identifiers are canonicalized so `01`, `02`, and `06`
  do not lose their leading zeros during cache merges. If `tavg` is absent from a
  cache, it is derived from `(tmin + tmax) / 2` where possible.
- Four weather missingness indicators are recorded before local interpolation:
  `weather_tavg_missing`, `weather_tmin_missing`, `weather_tmax_missing`, and
  `weather_prcp_missing`.
- If weather is unavailable, weather values are zero and indicators are one.
- Missingness indicators remain available in the `full` feature set; the default
  `cumulative` branch intentionally feeds only cumulative GDD.
- Optional location metadata consists of normalized city-family latitude,
  longitude, and elevation. It passes through a separate MLP and learnable gated
  residual connection after temporal aggregation. Unknown stations fail in strict
  training mode instead of silently receiving fabricated coordinates.
- Training history, validation/test metrics, and TensorBoard record the learned
  `temporal_feature_gate`, `weather_feature_gate`, and `location_gate` when present.
- Checkpoints store the fusion type, exact feature columns/dimensions, selected
  weather set, MLP sizes, and gate initializations. Resume validates these values.
- `infer_cached_window.py` and `infer_single_image.py` reconstruct these settings
  from checkpoints. Checkpoints without the new fields default to legacy/full
  reconstruction and remain usable for inference.

### Cache and retraining implications

- Existing DINOv2 image embedding caches can be reused when changing calendar,
  weather/GDD, or location/elevation metadata because fusion occurs in the temporal
  classifier, not the frozen image encoder.
- DINOv3 tiled precomputation now decodes/tiles images in parallel, packs tiles
  from multiple source images into full GPU batches, uses pinned asynchronous
  transfers, and exposes `--image-batch-size` plus `--prefetch-factor`. The
  fold-safe runner defaults to cache tile batch 256, image batch 16, 32 workers,
  and prefetch factor 4, all overridable through `CACHE_*` environment variables.
- The new gated metadata architecture requires training a new temporal classifier
  from scratch. Do not resume an old legacy-fusion training checkpoint as a gated
  run.
- Old checkpoints are still supported for inference through automatic legacy
  reconstruction.

### Loss and metrics

Recommended loss:

```text
hybrid = soft cross entropy + ordinal CDF distance
```

Primary experiment-selection metric: validation macro-F1.

Also report:

- exact accuracy
- plus/minus-one-stage accuracy
- date-window accuracy
- mean absolute stage error
- quadratic weighted kappa
- per-class precision, recall, F1, and support
- confusion matrices

High plus/minus-one accuracy with modest exact accuracy means most errors are
adjacent-stage boundary errors, not complete biological failures.

`date-window accuracy` is a soft score, not binary accuracy. For tolerance `T`,
a prediction receives `max(0, 1 - distance/T)`, where distance is the number of
days from the image date to the predicted class interval. With `T=7`, distances
of 1 through 6 days score 6/7 through 1/7, and distance 7 scores zero. The
tolerance applies symmetrically on either side of the predicted interval; it is
not split into three days before, one exact day, and three days after. Report
exact accuracy and macro-F1 as primary metrics and clearly label this value as a
secondary soft date-window score.

## Leakage Rules

These are non-negotiable:

- Split by physical station for deployment-generalization claims:
  `--fold-group-by station`.
- Never randomly split individual images or overlapping windows. Neighboring days
  are highly correlated and would leak.
- Train only on the training groups.
- Select `best_model.pt` using validation only.
- Evaluate test only after checkpoint selection.
- Training-only coverage filters and embedding augmentation/SMOTE must never alter
  validation or test data.
- Do not choose a random seed because its test results look better.
- Do not tune hyperparameters, monotonic penalties, or data exclusions on test.

## Fold Generation

`DINOv3_BBCH` now supports both the historical random-combination sampler and an
explicit LOSO strategy:

```text
--fold-strategy random|loso
```

LOSO behavior:

- requires `--fold-group-by station`;
- creates one fold per physical station present in generated metadata;
- uses exactly one test station per fold;
- rotates validation stations through a seeded circular ordering;
- with two validation stations, every station is validated exactly twice;
- guarantees train/validation/test disjointness;
- ignores `--folds`, since complete test coverage determines fold count;
- stores `fold_strategy` in split information and checkpoints, and validates it
  during resume;
- writes all assignments to `fold_assignments.csv` before training.

Use `--expected-stations 15` for a definitive 15-station experiment. It fails
before training if label/image discovery exposes a different count. The fold seed
changes fold order and balanced validation pairings, not eventual test coverage.
Do not search for a favorable seed.

The completed eight-fold DINOv3 run demonstrated this limitation directly. It
used 13 physical station groups and two test groups per fold, but only eight
stations appeared in any test split. `02.03`, `02.05`, `02.06`, `06.01`, and
`26.11` were never tested, while some other stations were tested two or three
times. Its aggregate metrics are therefore not a station-complete estimate.

## Coverage Filtering

Implemented training-only controls:

- `--min-train-stage-days N`: remove a station-stage from training when it has fewer
  than N usable target-image days.
- `--min-train-window-coverage-days N`: remove a training sample when fewer than N
  exact days exist in its causal window.

Validation and test remain complete and are stratified by coverage. This prevents
inflated metrics from deleting difficult deployment cases.

Start with these A/B tests rather than applying a large threshold blindly:

```text
baseline:        stage=0, window=0
coverage only:   stage=0, window=20
combined:        stage=8, window=20
```

Some real BBCH stages are naturally short. A 20-day stage threshold can remove
valid data and should not be the default.

## Monotonic Sequence Decoder

Optional post-processing is implemented with:

```text
--exclude-offseason --monotonic-decoding viterbi
```

Behavior:

- Decode each station-year independently in chronological order.
- Permit staying or advancing, never moving backward.
- Consecutive dates may advance one class; longer image gaps permit larger forward
  jumps so missing images do not force decoder lag.
- Preserve raw predictions in `pred_idx`/`pred_label`.
- Save constrained predictions in `decoded_pred_idx`/`decoded_pred_label`.
- Produce separate monotonic confusion matrices and per-class reports.
- Keep checkpoint selection based on raw validation metrics.
- Tune `--monotonic-advance-penalty` using validation only; start at 0.

Monotonic decoding requires OffSeason to be excluded because OffSeason occurs both
before sowing and after harvest and is not monotonic in the current class order.

## Recommended Commands

These are historical launch recipes. Check the current YAML-based entrypoints
in `DINOv3_BBCH/README.md` first; legacy DINOv2 paths are now under `archive/`.

### Precompute embeddings

Run once for a fixed image set, backbone, tiling configuration, and augmentation
configuration:

```bash
python DINOv2_BBCH/precompute_multiscale_embeddings.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch_cache \
  --image-backbone facebook/dinov2-base \
  --camera AUTO \
  --stream micro \
  --tile-streams micro \
  --tile-pooling attention \
  --tile-size 224 \
  --tile-stride 112 \
  --max-tiles 0 \
  --batch-size 64 \
  --num-workers 4 \
  --embedding-dtype float16 \
  --augment-views 1 \
  --augment-streams micro \
  --amp auto \
  --pretrained \
  --device cuda
```

The cache can be reused when changing temporal layers, loss, dropout, folds,
weather, or training epochs. Recompute it when changing images, backbone, tiling,
preprocessing, stream, or embedding augmentation.

### Recommended training baseline

```bash
python DINOv2_BBCH/run_multiscale_training.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch \
  --embedding-cache results_dinov2_bbch_cache/vit_embeddings.pt \
  --epochs 40 \
  --folds 8 \
  --fold-group-by station \
  --validation-groups 2 \
  --test-groups 2 \
  --fold-seed 42 \
  --seed 42 \
  --window-days 31 \
  --window-mode causal \
  --min-train-stage-days 8 \
  --min-train-window-coverage-days 20 \
  --stream micro \
  --camera AUTO \
  --batch-size 32 \
  --accumulation-steps 1 \
  --num-workers 0 \
  --temporal-model transformer \
  --temporal-aggregation cls \
  --temporal-layers 2 \
  --temporal-heads 8 \
  --temporal-norm-first \
  --temporal-ffn-multiplier 2 \
  --loss hybrid \
  --ordinal-ce-weight 0.5 \
  --checkpoint-metric macro_f1 \
  --dropout 0.2 \
  --weight-decay 3e-4 \
  --lr 1e-4 \
  --lr-scheduler cosine \
  --warmup-ratio 0.05 \
  --use-days-since-planting \
  --temporal-feature-fusion gated \
  --temporal-feature-hidden-dim 32 \
  --temporal-feature-gate-init 0.1 \
  --use-location-metadata \
  --location-feature-hidden-dim 16 \
  --location-gate-init 0.1 \
  --use-weather-metadata \
  --weather-feature-set cumulative \
  --weather-feature-gate-init 0.1 \
  --weather-cache results_dinov2_bbch_cache/meteostat_weather_cache.csv \
  --exclude-offseason \
  --monotonic-decoding viterbi \
  --monotonic-advance-penalty 0 \
  --device cuda \
  --log-interval 50
```

For a fair metadata ablation, keep folds and all other settings identical. Compare
no weather against `cumulative`, then optionally `daily_cumulative`; do not jump to
`full` unless testing the old broad weather vector deliberately. Test location in a
separate paired run as well.

### Fold-safe DINOv3 fine-tuning

`DINOv3_BBCH/run_finetuned_loso.sh` automates the required three stages separately
for every LOSO fold:

```text
training-station image fine-tuning
  -> fold-specific dense embedding cache
  -> fold-specific temporal training and final held-out test
```

For the current 13-station truncated table:

```bash
LABEL_PATH=labeling_truncated3.csv \
EXPECTED_STATIONS=13 \
FOLD_START=1 \
FOLD_END=13 \
RESULT_ROOT=results_dinov3_finetuned_loso \
bash DINOv3_BBCH/run_finetuned_loso.sh
```

For the intended definitive run, use the full canonical 15-station table and set
`EXPECTED_STATIONS=15`/`FOLD_END=15`. The runner saves adapted backbones under
`backbones/fold_N/backbone`, caches under `caches/fold_N`, and temporal results
under `temporal/run_fold_N`. `--only-fold N` preserves the original LOSO fold
identity when each temporal run is launched separately. The default runner can
delete completed multi-gigabyte caches to control disk use; use `KEEP_CACHES=1`
only when those caches are needed for later ablations.

The Meta DINOv3 repository is gated. Accept its Hugging Face terms and make sure
the same Linux user/Python environment can see the token. `HF_TOKEN` overrides a
saved login. A local model directory supplied through `IMAGE_BACKBONE` is the
most reliable option for long unattended runs.

### Resume interrupted training

```bash
python DINOv2_BBCH/run_multiscale_training.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch \
  --embedding-cache results_dinov2_bbch_cache/vit_embeddings.pt \
  --resume-checkpoint results_dinov2_bbch/last_checkpoint.pt \
  --epochs 40 \
  --device cuda
```

Resume with all architecture, fold, class, feature, and cache settings matching the
original run. The full original command is safer than relying on defaults.

In particular, match `--temporal-feature-fusion`, weather feature set and gate,
calendar gate, and location settings. These are validated against the checkpoint.

## Output Interpretation

Key files:

- `fold_assignments.csv`: inspect before trusting cross-validation coverage.
- `all_val_metrics.csv`: compare hyperparameter configurations here.
- `all_test_metrics.csv`: final held-out results, not a tuning source.
- `all_test_predictions.csv`: raw and optional decoded per-date predictions.
- `all_test_confusion_matrix.png`: raw aggregate confusion matrix.
- `all_test_monotonic_confusion_matrix.png`: constrained result when enabled.
- `all_val_coverage_metrics.csv` and `all_test_coverage_metrics.csv`: behavior under
  sparse windows/stages.
- `fold_*/training_coverage_filter.json`: exactly what training filters removed.
- `tensorboard/`: epoch curves and fold summaries.
- `fold_*/best_model.pt`: validation-selected model for that fold.
- `last_checkpoint.pt`: resumable optimizer/scaler/scheduler state.

Expected failure patterns seen previously:

- Training accuracy around 90-99% while validation is 60-70%: substantial
  station/camera/year overfitting.
- Large fold variance: station domain shift and data-quality differences dominate.
- Early-stage confusion with late-stage imagery: soil/stubble/background similarity,
  label-boundary ambiguity, weak visual evidence, or metadata dependence.
- PS/BBCH adjacent confusion: often biologically reasonable; inspect MAE and
  plus/minus-one accuracy alongside exact accuracy.

Recent controlled evidence:

- Correcting the weather-cache merge improved validation macro-F1 slightly in the
  observed run but did not improve held-out test macro-F1 over the buggy/mostly
  absent-weather run. The old full weather vector therefore has no demonstrated
  generalization benefit yet.
- This result motivated the compact cumulative-GDD branch and separate learned
  gate. Treat it as a hypothesis to test against a strict no-weather baseline, not
  as an established improvement.
- Location/elevation is likewise optional. A gate near its `0.1` initialization
  suggests the model found limited additional value; always inspect learned gates
  and held-out station metrics rather than assuming metadata helps.

## Historical CNN Baselines

`CNN/` and `CNN3D/` are retained as historical comparison pipelines. They are
useful for controlled backbone and raw-video experiments, but neither has
supplanted the frozen DINO feature pipelines. Do not mix their label definitions
or checkpoints with the current BBCH models.

### CNN: EfficientNet/ConvNeXt plus temporal Transformer

The `CNN/` pipeline was converted from the original ViT-oriented copy into a
torchvision CNN implementation:

- default image backbone: `efficientnet_v2_s`;
- optional backbones include `efficientnet_v2_m` and `convnext_tiny`;
- causal temporal windows, normally micro/10X only;
- full-image end-to-end training or frozen-backbone precomputed embeddings;
- temporal Transformer with `cls`, `mean`, or target-day aggregation;
- optional normalized days-since-planting feature for every temporal position;
- missing days represented through the temporal mask;
- train/validation/test station-group folds, with validation-selected checkpoints
  and test evaluation only after selection;
- configurable AdamW weight decay and dropout;
- ordinal CDF loss or soft-target cross entropy;
- AMP, gradient accumulation, cosine decay with linear warmup, fused AdamW,
  pinned-memory loaders, persistent workers, optional `torch.compile`, resumable
  `last_checkpoint.pt`, and TensorBoard logging;
- per-fold and aggregate confusion matrices and prediction/metric CSV files.

The validation and test group counts are file-level constants in
`CNN/run_multiscale_training.py`, currently two validation and two test groups.
The optional diversity constraints are enabled, requiring a multi-group
validation/test split to contain more than one physical station-code prefix when
possible. This reduces same-family splits such as `02.02` plus `02.03`, but it
does not solve the broader random-fold coverage limitation described above.

The cached CNN workflow is substantially faster because the frozen image encoder
runs once. Augmenting cached vectors cannot reproduce end-to-end image
augmentation unless augmented image views are explicitly encoded during
precomputation. Full-image training can augment dynamically and fine-tune the
backbone, but costs much more time and VRAM.

No complete CNN result directory or `all_test_metrics.csv` is present in this
local checkout. The conversation-level evidence was qualitative: training
accuracy frequently reached roughly 90-99% while validation stayed around
60-70%, indicating strong station/camera/year overfitting. Exact CNN aggregate
metrics should be added here only from preserved result files.

### CNN3D: raw-image spatiotemporal CNN

`CNN3D/` was cleaned so it now contains only the raw-image 3D CNN experiment. Old
cached-vector and 2D CNN plus temporal-Transformer paths were removed. Its active
flow is:

```text
causal image window [time, channels, height, width]
  -> 3D convolutions over time and space
  -> pooled visual feature
  -> optional days-since-planting MLP
  -> stage classifier
```

Implemented behavior:

- single micro or macro stream through `SingleStream3DCNN`, or paired streams
  through `MultiScale3DCNN`;
- shifted `labeling3.csv` classes: `OffSeason`, then `PS1` through `PS7`;
- label-table camera selection through `kamera`/`Camera`, including K2 where
  requested;
- CSV and Excel label input, including common separators and Turkish/Windows
  encodings;
- corrupt or unidentified images produce a warning and a zero frame instead of
  terminating a long DataLoader run;
- optional normalized days-since-planting metadata; weather and coordinates are
  not fused in CNN3D;
- causal missing-day masks and deterministic validation/test preprocessing;
- AMP, gradient accumulation, warmup plus cosine scheduling, fused AdamW,
  optional `torch.compile`, TensorBoard, resumable checkpoints, and end-of-run
  plots;
- `infer_image.py` supports a real previous-days folder or repeated-target-image
  diagnostic inference and restores image size from the checkpoint.

Training-only augmentation was added with random resized crop, horizontal flip,
small rotation, color jitter, mild blur, and random erasing. One random transform
seed is shared by all frames in a sample, preventing artificial frame-to-frame
camera motion. Validation and test remain deterministic. All augmentation
strengths and image size are exposed as CLI options and saved in checkpoints.

Observed CNN3D progression:

1. The larger/less-regularized raw 3D CNN grossly overfit the small
   station-split dataset.
2. A strong-regularization run used a 15-day window, base channels 8, feature
   dimension 128, dropout 0.5, days since planting, and the default strong
   augmentation recipe. The user reported that this swung too far and underfit.
3. The proposed middle-ground configuration is a 21-day causal window, base
   channels 16, feature dimension 192, dropout 0.35, effective batch size 16,
   and milder augmentation: crop minimum 0.85, rotation 3 degrees, color jitter
   0.12, blur probability 0.05, and erasing probability 0.05.

The middle-ground run has not yet been reported as completed. No CNN3D result
directory or aggregate metrics CSV is present in this checkout, so current
CNN3D evidence remains qualitative. For the next run, compare train and
validation curves before changing capacity again:

- both low: reduce regularization or increase capacity;
- train high and validation low: increase regularization or reduce capacity;
- validation rising at the final epoch: extend epochs while retaining
  validation checkpointing.

CNN3D is expected to need more data and stronger regularization than frozen
DINO embeddings because it learns spatial and temporal visual filters from
scratch. It should be treated as an architecture ablation, not the current
deployment candidate.

## Fold-Safe Fine-Tuned DINOv3 LOSO Result

Remote result directory:

```text
/home/can/blg521/results_dinov3_finetuned_loso/
```

This experiment used `labeling_truncated3.csv`, 13 physical stations, 21-day
causal windows, dense DINOv3 features, and the same 10-train/2-validation/1-test
LOSO structure as the frozen 13-station reference. For each fold, supervised
image-level fine-tuning saw only the ten training stations. It unfroze the final
of 12 DINOv3 blocks plus the final norm, selected the adapted backbone by
validation macro-F1, precomputed a new fold-specific dense cache, and then trained
the temporal classifier. This is the correct leakage-safe architecture for
backbone fine-tuning under station-held-out evaluation.

The saved aggregate directory was internally inconsistent:

- `summary.json`, aggregate predictions, confusion, and history contained folds
  2 through 13 only;
- `all_test_metrics.csv` contained folds 1 through 13 and had been modified later;
- fold 1 remained valid in `temporal/run_fold_1`, so its predictions were manually
  combined with folds 2 through 13 for the complete result below.

This arose because the shell runner aggregates only `FOLD_START..FOLD_END`; a
resumed run beginning at fold 2 does not automatically include an already
completed fold 1. Do not quote the directory's stale 12-fold `summary.json`.

Recomputed results over all 2,166 unique held-out prediction rows:

| Metric | Result |
| --- | ---: |
| exact accuracy | 0.8638 |
| aggregate macro-F1 | 0.8457 |
| weighted F1 | 0.8615 |
| station-balanced exact accuracy | 0.8644 +/- 0.0516 |
| station-balanced fold macro-F1 | 0.8142 +/- 0.0885 |
| plus/minus-one-stage accuracy | 1.0000 |
| five-day soft date-window score | 0.9073 |
| mean absolute stage error | 0.1362 |
| quadratic weighted kappa | 0.9824 |

Aggregate per-class F1:

| Class | F1 |
| --- | ---: |
| `BBCH0` | 0.9382 |
| `BBCH1` | 0.8978 |
| `BBCH2` | 0.7602 |
| `BBCH3` | 0.8742 |
| `BBCH5` | 0.7389 |
| `BBCH6_7` | 0.7764 |
| `BBCH8` | 0.9344 |

Per-station LOSO exact accuracy and macro-F1:

| Test station | Exact | Macro-F1 |
| --- | ---: | ---: |
| `02.02` | 0.9389 | 0.9102 |
| `27.02` | 0.9359 | 0.8980 |
| `02.04` | 0.9313 | 0.9212 |
| `27.05` | 0.9068 | 0.8917 |
| `27.04` | 0.8925 | 0.8691 |
| `02.05` | 0.8496 | 0.7302 |
| `02.06` | 0.8495 | 0.7916 |
| `01.01` | 0.8444 | 0.8417 |
| `01.05` | 0.8378 | 0.8117 |
| `02.03` | 0.8363 | 0.8013 |
| `06.01` | 0.8270 | 0.7475 |
| `06.02` | 0.8092 | 0.7604 |
| `26.11` | 0.7784 | 0.6107 |

All errors remained adjacent. The dominant confusions were `BBCH2 -> BBCH1`
(52), `BBCH2 -> BBCH3` (52), `BBCH6_7 -> BBCH5` (31), `BBCH0 -> BBCH1`
(28), and `BBCH5 -> BBCH6_7` (26). The result improved BBCH5 F1 substantially,
but BBCH2 remains the most important transition-boundary problem. `26.11`
continues to have extremely sparse BBCH3/BBCH5 support and the weakest station
macro-F1.

Compared superficially with the frozen `labeling_truncated2.csv` LOSO result,
fine-tuning changed:

| Metric | Frozen | Fine-tuned |
| --- | ---: | ---: |
| exact accuracy | 0.8560 | 0.8638 |
| aggregate macro-F1 | 0.8343 | 0.8457 |
| station-mean exact accuracy | 0.8555 | 0.8644 |
| station-mean macro-F1 | 0.8017 | 0.8142 |
| exact fold standard deviation | 0.0683 | 0.0516 |
| macro-F1 fold standard deviation | 0.1164 | 0.0885 |

The largest apparent gains were `06.01` (+8.1 exact points, +16.6 macro-F1),
`27.04` (+7.5/+7.9), and `02.05` (+5.3/+7.1). Six of 13 stations improved.
However, this is not a clean backbone A/B: `labeling_truncated3.csv` revised
milestone dates for `01.01`, `02.03_2013`, `02.05`, and `06.01`. Those include
several stations with the largest changes. Paired station-level differences were
not statistically conclusive, so do not attribute the gain to fine-tuning yet.

Backbone fine-tuning itself was stable:

- all folds ran eight image-level epochs;
- the mean best epoch was 5.1, with a range of 3 to 7;
- mean selected train macro-F1 was 0.8765;
- mean selected validation macro-F1 was 0.8457;
- final validation macro-F1 averaged 0.8356.

The temporal stage still overfit after checkpoint selection:

- mean best epoch: 12.5, range 3 to 22;
- selected train macro-F1: 0.9625;
- selected validation macro-F1: 0.8815;
- selected test macro-F1: 0.8142;
- epoch-30 train macro-F1: 0.9872;
- epoch-30 validation macro-F1: 0.8543.

Thirty temporal epochs are sufficient; retain validation-selected checkpoints
instead of increasing epoch count. Location and calendar gates again remained
near 0.1, so their benefit remains unproven.

The next decisive experiment is a frozen-versus-fine-tuned run using exactly the
same `labeling_truncated3.csv`, LOSO assignments, filters, temporal settings,
metadata, and seeds. Keep the current one-block/eight-epoch adaptation recipe
until that controlled comparison demonstrates a repeatable gain. Only then test
two unfrozen blocks, lower backbone learning rates, or longer fine-tuning.

## Frozen DINOv3 21-Day LOSO Reference

Remote result directory:

```text
/home/can/blg521/results_dinov3_bbch_21_LOSO/
```

This is the strongest and most scientifically useful result so far, but it is a
13-station LOSO experiment rather than the intended 15-station experiment.
`labeling_truncated2.csv` contains 13 station codes and omits `11.03` and
`26.12`; `labeling_bbch_iso_dates.csv` contains all 15. Generated metadata had
2,892 rows, 14 station-years, and 13 physical stations because `02.03` has two
years. The LOSO scheduler correctly generated 13 folds with 10 train, two
validation, and one unique test station per fold. Every included station was
tested once and validated exactly twice.

Aggregate results over 2,166 unique held-out prediction rows:

| Metric | Result |
| --- | ---: |
| exact accuracy | 0.8560 |
| aggregate macro-F1 | 0.8343 |
| station-balanced exact accuracy | 0.8555 +/- 0.0683 |
| station-balanced fold macro-F1 | 0.8017 +/- 0.1164 |
| plus/minus-one-stage accuracy | 1.0000 |
| five-day soft date-window score | 0.9003 |
| mean absolute stage error | 0.1440 |
| quadratic weighted kappa | 0.9815 |

Aggregate per-class F1:

| Class | F1 |
| --- | ---: |
| `BBCH0` | 0.9376 |
| `BBCH1` | 0.8930 |
| `BBCH2` | 0.7666 |
| `BBCH3` | 0.8583 |
| `BBCH5` | 0.6644 |
| `BBCH6_7` | 0.7828 |
| `BBCH8` | 0.9375 |

Per-station LOSO exact accuracy and macro-F1:

| Test station | Exact | Macro-F1 |
| --- | ---: | ---: |
| `02.02` | 0.9667 | 0.9484 |
| `02.04` | 0.9437 | 0.9337 |
| `27.02` | 0.9423 | 0.9298 |
| `27.05` | 0.9255 | 0.9143 |
| `01.01` | 0.8778 | 0.8750 |
| `02.03` | 0.8683 | 0.8650 |
| `02.06` | 0.8387 | 0.7532 |
| `27.04` | 0.8172 | 0.7898 |
| `26.11` | 0.8108 | 0.6533 |
| `01.05` | 0.8108 | 0.7884 |
| `02.05` | 0.7970 | 0.6594 |
| `06.02` | 0.7763 | 0.7297 |
| `06.01` | 0.7459 | 0.5814 |

All aggregate mistakes remained adjacent. The dominant confusions were
`BBCH2 -> BBCH1` (53), `BBCH2 -> BBCH3` (40), `BBCH5 -> BBCH6_7` (35), and
`BBCH6_7 -> BBCH5` (29). `BBCH5` remains the main class bottleneck. `06.01` is
the clearest station-domain problem; `02.05` mapped all 11 BBCH0 samples to
BBCH1; `26.11` had extremely sparse BBCH3/BBCH5 support, making its station
macro-F1 unstable.

Coverage remains important. Test windows meeting the 12-day threshold achieved
0.8681 exact accuracy, versus 0.7885 below it. Validation/test were correctly
left unfiltered. Training folds kept about 1,395 samples on average while
excluding about 233 for insufficient window coverage, 38 for stage support, and
142 for missing targets.

Checkpoint behavior confirms overfitting after convergence:

- mean best epoch: 11.9, range 3 to 22;
- selected train macro-F1: 0.9602;
- selected validation macro-F1: 0.8641;
- epoch-30 train macro-F1: 0.9870;
- epoch-30 validation macro-F1: 0.8279.

Keep validation-selected checkpoints. Twenty epochs captures almost every fold,
but 30 with early stopping remains reasonable. Location and calendar gates again
stayed near their 0.1 initialization, so metadata benefit is still unproven.

The previous 21-day eight-fold result had exact accuracy 0.8500 and aggregate
macro-F1 0.8335 over only eight tested stations with repeated appearances. The
LOSO improvement is encouraging but not a clean architecture A/B because train
and validation composition changed. The next definitive run should use the full
15-station BBCH label table with `--expected-stations 15`.

Recommended LOSO fold arguments:

```bash
--fold-strategy loso \
--fold-group-by station \
--expected-stations 15 \
--validation-groups 2 \
--test-groups 1 \
--fold-seed 42
```

## Historical Eight-Fold DINOv3 BBCH Run

Remote result directory:

```text
/home/can/blg521/results_dinov3_bbch/
```

Core configuration:

- frozen DINOv3 ViT-B/16 LVD-1689M backbone;
- micro/10X stream only;
- 224 tiles, stride 112, all tiles;
- compact dense 2x2 patch grid plus CLS;
- float16 cache (5.4 GB);
- causal 31-day Transformer, two layers, CLS temporal aggregation;
- hybrid loss with ordinal CE weight 0.5;
- checkpoint selected by validation macro-F1;
- days since sowing and location metadata through gates initialized at 0.1;
- no weather, no OffSeason targets, no monotonic decoding;
- 20 epochs, eight random station-group folds;
- minimum stage support 8 and training-window coverage 6.

Aggregate held-out results over 2,326 prediction rows:

| Metric | Result |
| --- | ---: |
| exact accuracy | 0.8306 |
| macro-F1 | 0.8146 |
| plus/minus-one-stage accuracy | 1.0000 |
| mean absolute stage error | 0.1694 |
| quadratic weighted kappa | 0.9774 |
| seven-day soft date-window score | 0.8943 |

Unweighted fold summaries:

- test exact accuracy: mean 0.8325, standard deviation 0.0539;
- test macro-F1: mean 0.8086, standard deviation 0.0767;
- validation macro-F1: mean 0.8164, standard deviation 0.0339;
- selected epoch: mean 10.1, range 2 to 19;
- selected-checkpoint training accuracy: mean 0.9551;
- selected-checkpoint validation accuracy: mean 0.8335.

Validation and test macro-F1 agree closely, indicating that checkpoint selection
transferred reasonably. The roughly 12-point train/validation accuracy gap still
shows substantial station-domain overfitting.

Aggregate per-class F1:

| Class | F1 |
| --- | ---: |
| `BBCH0` | 0.9149 |
| `BBCH1` | 0.8851 |
| `BBCH2` | 0.7563 |
| `BBCH3` | 0.8294 |
| `BBCH5` | 0.7031 |
| `BBCH6_7` | 0.7294 |
| `BBCH8` | 0.8837 |

Every aggregate confusion-matrix error was to an immediately adjacent class.
The remaining problem is primarily transition placement, especially around
`BBCH2`, `BBCH5`, and `BBCH6_7`, rather than catastrophic stage-order errors.
The perfect plus/minus-one metric is consequently saturated and is no longer
useful for selecting among strong models.

Observed station-year exact accuracy among stations that appeared in test:

| Station-year | Exact accuracy |
| --- | ---: |
| `27.02_2016` | 0.9295 |
| `02.04_2013` | 0.9219 |
| `27.05_2013` | 0.8975 |
| `02.02_2014` | 0.8667 |
| `27.04_2016` | 0.8199 |
| `01.05_2016` | 0.8063 |
| `01.01_2014` | 0.7630 |
| `06.02_2013` | 0.6612 |

`06.02_2013` is the clearest domain-shift problem and explains much of the weak
performance in folds 1 and 5. Fold 5 BBCH2 recall fell to 0.0526, with nearly all
BBCH2 samples predicted as BBCH1 or BBCH3.

Coverage-stratified exact accuracy:

| Exact days in 31-day window | Accuracy |
| --- | ---: |
| 0-9 | 0.7083 |
| 10-19 | 0.7521 |
| 20-25 | 0.8998 |
| 26-31 | 0.8285 |

The bins are confounded by station and class composition, so they are not a
causal estimate. Nevertheless, a training threshold of 6 removed only about
eight low-coverage samples in inspected folds and was effectively very weak.
The next controlled threshold should be 12 or 16, not 20 by default.

Post-hoc soft date-window results derived from the prediction rows:

| Metric | Score |
| --- | ---: |
| exact | 0.8306 |
| 3-day soft tolerance | 0.8573 |
| 5-day soft tolerance | 0.8774 |
| 7-day soft tolerance | 0.8943 |

Seven days adds about 6.4 percentage points over exact classification. Use three
days for a stricter secondary summary when label dates are considered reliable.
Changing this tolerance does not change the training loss in the current command,
and checkpoint selection used macro-F1, so it does not explain the learned model.

Both learned metadata gates remained essentially at their 0.1 initialization.
This does not prove that the metadata MLP outputs had no effect, but it provides
no evidence that training learned to increase or suppress either source. Run a
matched no-metadata ablation before attributing the DINOv3 result to metadata.

Recommendations recorded before LOSO was implemented:

1. Use LOSO folds so every physical station is tested. This is now implemented;
   the remaining requirement is to run the full 15-station label table.
2. Keep the same DINOv3 cache and use 30 epochs; validation checkpointing protects
   folds that peak early while allowing folds 5 and 6 more time.
3. Compare training-window coverage 6 versus 12 or 16.
4. Use exact accuracy/macro-F1 as primary metrics and three-day soft score as
   secondary.
5. Run identical no-metadata versus days/location metadata ablations.
6. Compare DINOv3 dense against DINOv2 using identical labels, folds, filters,
   metadata, and training hyperparameters before claiming a backbone improvement.

## Experiment Discipline

For every run, preserve:

- exact command;
- code version or archive;
- label file checksum/name;
- embedding cache configuration;
- seed and fold seed;
- fold assignments;
- metadata sanity output;
- validation-selected epoch;
- per-fold and aggregate metrics;
- confusion matrices;
- notes about excluded bad data.

Change one major factor at a time. Recommended experiment order:

1. Run frozen versus one-block fine-tuned DINOv3 using the same
   `labeling_truncated3.csv`, all 13 LOSO assignments, and otherwise identical
   settings. This isolates backbone adaptation from the revised milestone dates.
2. Run the selected backbone policy with the full 15-station canonical label
   table and `--expected-stations 15`; both completed truncated-table runs cover
   only 13 physical stations.
3. Run no-weather versus cumulative-GDD A/B, then daily+cumulative if warranted.
4. Compare coverage thresholds.
5. Test location/elevation in a separate paired ablation.
6. Tune dropout, weight decay, and temporal depth using validation macro-F1.
7. Tune monotonic advance penalty on validation.
8. Only then compare more unfrozen blocks, larger backbones, or multimodal
   streams.

Do not infer improvement from one favorable fold. Prefer mean, standard deviation,
worst-station performance, and per-class recall across the same assignments.

## Deployment Notes

- Use `infer_cached_window.py` for realistic cached/tiled inference.
- Give it the target image and a folder containing previous dates.
- It resolves the previous causal window from filename dates.
- Missing previous days remain masked unless explicitly repeated.
- `infer_single_image.py` repeats one image and is diagnostic, not equivalent to a
  true temporal deployment window.
- A final deployment model should be retrained on all development stations only
  after hyperparameters are frozen. Preserve a genuinely external station or season
  for the final unbiased evaluation.

## Metadata + ExG Baseline

`Baseline_BBCH/` is a lightweight non-DINO comparison pipeline. It reuses
DINOv3's BBCH metadata generation, weather/GDD calculation, location table, and
balanced station-level LOSO folds, but replaces learned image embeddings with
direct ExG greenness statistics. A histogram gradient-boosted classifier receives
coordinates/elevation, weather and missingness, daily/cumulative GDD, days since
sowing, current ExG summaries, and strictly causal rolling summaries.
ExG extraction and independent LOSO folds run in process pools using all detected
CPU cores by default; `--exg-workers` and `--fold-workers` override the counts.

Run all folds with:

```powershell
.\.venv\Scripts\python.exe Baseline_BBCH\run_metadata_exg_training.py `
  --label-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --out-dir results_metadata_exg_bbch `
  --stream micro `
  --window-days 21 `
  --validation-groups 2 `
  --expected-stations 15
```

The implementation has focused tests for green-image ExG, corrupt-image
missingness, and causal rolling features. No complete training result has been
recorded yet; do not quote baseline accuracy until the full LOSO run finishes.

## DINOv3 Linear Single-Day Pipeline

`DINOv3_BBCH_Linear/` is an isolated non-temporal DINOv3 experiment. Each image
is tiled, encoded with a partially fine-tuned DINOv3 backbone, pooled with dense
patch/tile attention, and concatenated with normalized cumulative GDD plus
latitude/longitude/elevation. `LayerNorm -> Dropout -> Linear` produces the
seven-class BBCH prediction. There is no temporal window, temporal Transformer,
or RNN.

The training CLI supports one fold or all balanced station-level LOSO folds,
validation macro-F1 checkpoint selection, full requested metrics, and optional
Hugging Face backbone export. `infer_single_image.py` reconstructs the complete
model from `best_model.pt` and computes cumulative GDD from planting date through
the target date. Focused synthetic tests cover attention masks, direct metadata
fusion, partial-backbone gradients, LOSO invariants, checkpoint reconstruction,
and inference probabilities. No full LOSO result has been recorded yet.

## Station-Year Assisted Labeling

`demo/label_station_year.py` uses `demo/best_model.pt` and
`demo/best_finetune.pt` by default to propose milestone dates for a complete
station-year. It resolves a 10X camera folder, encodes every dated image once,
caches dense embeddings for reuse, predicts observed target days with the
checkpoint's 21-day causal window, applies sowing-anchored monotonic Viterbi
decoding, and writes daily probabilities plus reviewable milestone and
canonical-row proposals. See `demo/STATION_YEAR_LABELING.md`.

Sowing and harvest can be supplied through `--sowing-date` and
`--harvest-date`. Sowing is both a fixed output boundary and the model's
days-since-sowing input. The seven-class checkpoint can propose emergence
through maturity, but cannot independently identify harvest because it has no
post-harvest class. Outputs are always marked `REVIEW_REQUIRED`; the tool never
edits canonical label files. Unknown station families can provide explicit
latitude, longitude, and elevation. A real two-date CUDA smoke test verified
local fine-tuned-backbone reconstruction, temporal inference, CSV/JSON output,
and embedding-cache reuse.

## Historical Open Questions

1. Reconcile the two differing `2 - Emergence` dates between the canonical CSV and
   XLSX label files, then rerun the full test suite.
2. Complete a controlled frozen-versus-fine-tuned DINOv3 A/B using the same
   `labeling_truncated3.csv`; the existing comparison is confounded by four
   revised station-year milestone schedules.
3. Complete a true 15-station LOSO run using `labeling_bbch_iso_dates.csv`; the
   current truncated tables expose 13 physical stations and omit `11.03` and
   `26.12`.
4. Quantify whether cumulative GDD helps unseen stations rather than merely fitting
   geography/season.
5. Quantify location/elevation value under identical folds, with and without
   weather, and inspect the learned location gate.
6. Audit bad-image and questionable-label rows with explicit, versioned exclusion
   files.
7. Determine whether overlap stride 112 materially beats stride 224 for the added
   storage/runtime cost.
8. Compare DINOv2 Base, DINOv3 Base dense, and DINOv3 Large only under identical,
   station-complete folds.
9. Consider supervised auxiliary attributes (emergence, heading, flowering,
   senescence) if reliable annotations can be created; these can teach visual cues
   more directly than stage labels alone.
10. Evaluate model calibration and uncertainty for human-review routing.

## Maintenance Rule

Update this handoff whenever any of these change:

- canonical model folder;
- class definitions or labeling file;
- data/fold policy;
- recommended commands;
- validation/test methodology;
- major architecture or metadata features;
- verified tests;
- open priority for the next session.
