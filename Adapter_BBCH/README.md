# Transition-subspace adapter: first BBCH pilot

This is an **unvalidated research experiment**, not a claim of improvement over
LoRA or the temporal models. The initial implementation operates after frozen
DINOv3 feature pooling:

```text
current 10X image -> tiles -> original pretrained frozen DINOv3
 -> mean dense descriptors within each tile -> mean tiles
 -> parameter-free LayerNorm -> residual adapter -> small MLP -> 7 BBCH classes
```

No temporal windows, calendar/weather/location inputs, or metadata fusion are
used. Station/year and stage labels are used only for training splits, label
subsampling, and basis construction. Inference needs one image's cached features.

The shared label builder, LOSO scheduler, ordinal loss, and frozen feature
precomputation are imported/reused from `DINOv3_BBCH/`. Existing model files are
not modified. Run the commands below from the repository root. 

## 1. Dependencies and label/fold audit

```bash
python -m pip install -r Adapter_BBCH/requirements.txt

python Adapter_BBCH/run_adapter_training.py \
  --config Adapter_BBCH/config.yaml \
  --dry-run
```

Dry-run prints station/stage counts and fold count without loading features,
downloading a backbone, training, or writing experiment files. Do not hard-code
the older handoff's station count: inspect the actual discovery first, then
optionally pass `--expected-stations N` to subsequent commands.

This pipeline excludes OffSeason and targets only current-day images. It does
not apply historical temporal-window coverage or station-stage training filters.
All control arms receive exactly the same images and folds.

## 2. Build a frozen feature cache once

You may reuse an existing cache **only if its backbone is the original public
pretrained DINOv3**, with no supervised updates. A cache from a fine-tuned LOSO
fold must not be shared across folds. This runner checks the cache's recorded
backbone/pretraining metadata and rejects local backbone paths or explicit
fine-tuning markers; those checks cannot detect falsified cache metadata.

Otherwise use the shared precomputation script:

```bash
python DINOv3_BBCH/precompute_multiscale_embeddings.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_adapter_frozen_cache \
  --image-backbone facebook/dinov3-vitb16-pretrain-lvd1689m \
  --camera AUTO --stream micro \
  --preplant-days 0 --postharvest-days 0 \
  --transition-days 2 --date-tolerance-days 5 \
  --tile-streams micro --tile-pooling attention \
  --tile-size 224 --tile-stride 112 --max-tiles 0 \
  --vit-image-size 224 \
  --dense-features --dense-grid-size 2 --dense-include-cls \
  --embedding-dtype float16 --amp auto \
  --batch-size 128 --image-batch-size 8 --num-workers 8 \
  --device cuda
```

First extraction may require Hugging Face model access. Here `--tile-pooling
attention` tells the shared extractor to store separate tile features; **this
pilot does not train an attention pooler**. Dense descriptors (CLS plus compact
patch grid) and tiles are averaged with fixed weights. This deliberately keeps
basis and inference coordinates identical, at the cost of learned spatial
selection. No spatial region is treated as an independent labeled image.

Augmented cache views are not used by this first pilot. Do not compare it with a
baseline using a different pooling scheme as evidence for the adapter alone.

## 3. Configure and run one fold with all five controls

Edit `Adapter_BBCH/config.yaml` to set the cache path, output directory, folds,
adapter settings, label budget, and training settings. Set `label_percentage`
to a percentage greater than 0 and at most 100; the supplied file defaults to
100%. The legacy `label_fraction` setting remains supported, where `0.25` means
25%. The supplied file contains all training options and uses the same defaults
as the CLI examples.

```bash
python Adapter_BBCH/run_adapter_training.py \
  --config Adapter_BBCH/config.yaml
```

Configuration precedence is: parser defaults, then YAML, then explicit CLI
options. An explicit `--label-percentage` or legacy `--label-fraction` overrides
either label-budget spelling in YAML. A YAML file cannot specify both settings,
and a CLI command cannot specify both options. YAML loading is explicit through
`--config`; CLI-only invocation remains supported. Relative data/cache/output
paths are resolved from the current working directory, so run from the
repository root, or use absolute paths.

The file is a flat mapping. Keys use underscores (`fold_id`) or hyphens
(`fold-id`). `methods` is a YAML list: `[all]` or `[transition, random, pca]`.
Booleans use `true`/`false`; `expected_stations: null` leaves the station-count
check disabled. Unknown/duplicate keys, invalid types/choices, and invalid budgets
are rejected. Resolved settings, including CLI overrides, are saved in
`run_config.json` and each selected checkpoint.

For a one-off override without editing the file:

```bash
python Adapter_BBCH/run_adapter_training.py \
  --config Adapter_BBCH/config.yaml \
  --fold-id 0 --methods transition random pca \
  --label-percentage 25 --out-dir results_adapter_budget25 --device cuda
```

For backward compatibility, the percentage option can instead be written
`--label-fraction 0.25`.

If YAML enables a boolean, `--no-dry-run` or `--no-ignore-status-csv` can disable
it. This configuration applies to the adapter training runner; shared DINOv3
feature precomputation in section 2 retains its existing CLI.

The in-memory classifier is small; CPU is sufficient for this phase, and
`--device cuda` is supported. The backbone does not run during adapter training.
Use a fresh output directory for every invocation; existing experiments are
never overwritten or implicitly resumed. `--methods transition random pca` selects
individual arms. Defaults run one fold and all arms.

| Method | Adapter input projection | Trainable adapter weights |
|---|---|---|
| `none` | No adapter; identical MLP head | 0 |
| `transition` | Fixed basis from training station-year stage differences | `D*r` |
| `random` | Fixed orthonormal random basis | `D*r` |
| `pca` | Fixed principal directions of normalized training features | `D*r` |
| `learned` | Same initialization as random, but projection is learned too | `2*D*r` |

All arms use the same head initialization, training-label subset, batch ordering,
loss, and validation checkpoint criterion. The fixed-basis controls match
trainable parameter counts exactly. The learned bottleneck has more parameters;
it is an architectural control, not a parameter-matched one. Adapter up-projections
start at zero, so every arm starts with identical predictions. The learned
down-projection begins receiving gradients after the up-projection changes.

`learned` is a nonlinear residual bottleneck adapter, **not backbone LoRA**.
Ordinary Q/V LoRA needs a separate end-to-end image-training experiment; it cannot
be trained from these frozen final embeddings. No LoRA comparison is claimed here.

## Basis estimation

For every fold, split stations before selecting training labels or fitting bases:

1. Apply the same parameter-free LayerNorm to every image vector used in either
   basis estimation or model inference.
2. Average selected training images within each station-year and hard BBCH stage.
   A prototype needs `--min-stage-images` images. The classifier still uses the
   original soft boundary targets, remapped to the seven in-season classes.
3. Subtract adjacent stage prototypes within a station-year. Missing stages are
   not bridged; e.g., BBCH1 is never compared directly to BBCH3.
4. Average valid differences across years within each physical station, then
   average equally across stations. Require `--min-transition-stations` distinct
   training stations for each transition.
5. Stack supported mean transitions as columns of `D` and take its left singular
   vectors. Freeze the first `r` vectors as rows of the input projection `A`.
6. Train `h' = h + B GELU(A h)` and the MLP. Here `h` is already normalized.

There are at most six supported mean directions for seven classes. If their
numerical rank is below the requested rank, all fixed controls use that same
effective rank. No random directions are added to fill the transition basis.
If no nonzero supported transition exists, the run fails with an explanation.
The basis report records station support, mean norms, variation between stations,
singular values, and the effective rank. This first version does not apply a
variance penalty or reliability weighting.

Two limitations matter for interpretation:

- The residual adapter restricts the inputs to its correction; the MLP still sees
  the original full feature. This does not enforce station invariance.
- With complete balanced prototypes and full rank, adjacent differences span the
  same space as centered class prototypes. This pilot therefore cannot by itself
  establish a uniquely ordinal mechanism. Before making that claim, add a
  supervised class-prototype contrast control and compare truncated ranks.

## Full LOSO and label-budget experiments

After the one-fold pipeline check, use `--fold-id 0` with a new `--out-dir` for
complete station-held-out evaluation. Keep `--fold-seed 42` fixed across methods
and budgets; repeat with predefined training seeds, not seeds picked by test score.

Use `--label-percentage 10`, `25`, `50`, and `100` in separate output directories.
Both prototype estimation/PCA and supervised training see **only** that selected
training subset. Validation and test remain complete. For a fixed seed, selected
subsets are nested across budgets. Sampling is stratified by station-year and
stage, retaining at least one example per group; it uses the full labels to define
strata and is not an annotation-acquisition policy. The sampler selects
`ceil((requested percentage / 100) × group size)` images per group. Group rounding
and the minimum of one image can therefore increase realized coverage, especially for
rare stages; the requested percentage is not an exact global sample count. The
requested and realized percentages, selected and full training-image counts, and
sampling policy are saved in split reports, basis reports, per-method metrics, and
checkpoints. Sparse subsets may require a predeclared lower
`--min-stage-images`; compare the same setting across methods and never fill
prototype gaps using withheld labels.

The nested sampler replaces the earlier independent random draws for budgets
below 100%; the full-data selection is unchanged. Reports include its policy
version and effective per-fold sampling seed (`seed + fold_id`). Aggregate
training-image counts sum the fold-specific sets and include repeated images
across LOSO folds; they are not unique annotation counts.

Checkpoints are selected by validation macro-F1 across all seven fixed classes.
Test is evaluated once per method/fold after selection. Do not choose rank,
support thresholds, label exclusions, or other settings from held-out test scores.
One-fold results validate the workflow; they do not establish generalization.

## Outputs

```text
out_dir/
  run_config.json                 # args, label SHA256, cache SHA256/settings
  image_metadata.csv
  fold_assignments.csv
  fold_1/
    split_info.json               # split stations/counts and requested/realized label coverage
    selected_training_labels.csv
    basis_report.json
    bases.pt
    transition/                   # and none/random/pca/learned
      best_model.pt               # selected model, basis, config, provenance
      history.csv
      val_metrics.json
      val_predictions.csv         # selected epoch, not the final epoch
      test_predictions.csv
      test_metrics.json
      confusion_matrix.csv
      per_class_metrics.csv
  all_test_metrics.csv
  combined_test_predictions.csv
  aggregate_<method>.json           # requested and realized coverage by method
  comparison_summary.csv             # requested/realized coverage for budget comparisons
```

The comparison includes pooled accuracy/macro-F1, station-mean/std/min metrics,
stage MAE, quadratic kappa, and plus/minus-one accuracy. All seven classes are
included in macro-F1 even when a station has no examples of one class; keep this
definition fixed across comparisons. Partial runs are marked `complete_loso=false`.
The run does not currently support resume or aggregate prior output directories.

Cache matching uses the portable `station/year/camera/scale/filename` suffix,
so Windows- and Linux-rooted paths can match. Missing cached target images fail
the run; validation/test are never silently dropped. Inference verifies the
extraction configuration against the checkpoint. Cache metadata is descriptive;
retain the original extraction command/code and cache SHA256 for provenance.

## Cached single-image inference

Use the same backbone and extraction settings for any new image cache:

```bash
python Adapter_BBCH/infer_cached_image.py \
  --checkpoint results_adapter_bbch_pilot/fold_1/transition/best_model.pt \
  --embedding-cache results_adapter_frozen_cache/vit_embeddings.pt \
  --image-path data/02.02/2015/K1/10X/ACTUAL_FILENAME.jpeg
```

Replace the example with a real image present in the cache. This first version
does not provide a raw-image inference wrapper; the shared extractor creates its
features. Adapter inference itself uses no labels, dates, or station metadata.

## Optional remote unit and synthetic training checks

```bash
python -m unittest discover -s Adapter_BBCH/tests -p 'test_*.py' -v
```

Tests cover station/year balancing, unsupported transitions, selected-label
sampling, frozen/trainable gradients, zero-residual initialization, checkpoint
reconstruction, cache and split guards, and a two-epoch synthetic run through all
five methods. They do not download a backbone or validate real-data accuracy.
