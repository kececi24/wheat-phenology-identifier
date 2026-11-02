# DINOv3 transformer-block adapters for BBCH

This experiment trains single-image classifiers from **raw images**. It inserts
PEFT into the last two pretrained DINOv3 transformer blocks. There is no temporal
model, calendar input, weather input, or station metadata input. Station/year and
labels are used only for splitting, label-budget sampling and training-only prior
estimation.

| Method | Visual encoder changes | Default adapter weights, ViT-B / two blocks |
| --- | --- | ---: |
| `none` | Frozen pretrained encoder | 0 |
| `lora` | Rank-3 LoRA in Q and V of each selected block | 18,432 |
| `standard` | Trainable width-6 parallel FFN bottleneck | 18,432 |
| `transition` | Width-6 parallel FFN bottleneck with transition prior | 18,432 |

All four use the same learned dense-patch/tile pooling and MLP:
LayerNorm -> Linear(768,128) -> GELU -> Dropout -> Linear(128,7).
The head and pooling receive identical initial weights for a given fold/seed.
Frozen backbone parameters, including all norms and pretrained LayerScale, stay
frozen and in evaluation mode; new pooling/head parameters train normally.

The standard branch is a minimal **AdaptFormer-style parallel bottleneck**, not
an exact reproduction of all AdaptFormer settings. LoRA implements
`Wx + (alpha/r) B A x` with frozen `W`, trainable `A`, initially zero `B`, and
Q/V targets only. No extra `peft` dependency is needed for these explicit modules.
See [LoRA](https://arxiv.org/abs/2106.09685),
[AdaptFormer](https://arxiv.org/abs/2205.13535), and
[HF DINOv3](https://huggingface.co/docs/transformers/model_doc/dinov3).

## Transition adapter

Before adapting each fold, the **unmodified pretrained encoder** processes only
the selected training images with deterministic tiling and no augmentation.
For each adapted block, a temporary hook reads its actual `norm2` output, the
input to its pretrained FFN. CLS and patch activations are averaged per tile
(register tokens excluded), then tiles are averaged per image. No additional
normalization is applied to these image summaries.

Training-only stage prototypes are averaged within station-year. Differences
between adjacent stages are averaged across years within station and then equally
across physical stations. SVD of these supported mean differences yields a
separate orthonormal `A0` for each block, in that block's coordinates. With seven
classes the maximum transition rank is six. Missing boundaries are never bridged;
if supported rank is below the requested width the run fails with an explanation.
It does not fill missing directions with noise.

For normalized tokens $z$, the adapter computes:

$$
\begin{aligned}
A &= A_0 + E \\
\delta &= B \, \operatorname{GELU}(A z) \\
\mathbf{y}_{\text{block}} &= \mathbf{y}_{\text{orig}} + \delta \\
\mathcal{L} &= \mathcal{L}_{\text{hybrid\_BBCH}} + \lambda_{\text{prior}} \cdot \frac{1}{L} \sum_{l=1}^{L} \|E_l\|_F^2
\end{aligned}
$$

`A0` is a frozen buffer, `E` starts at zero, and `B` starts at zero. Standard
adapters use a random trainable down-projection and the same zero up-projection.
Both branches are added outside the frozen FFN's LayerScale/drop-path. Every
PEFT arm therefore initially preserves the pretrained encoder function. These
are hypotheses to evaluate; accuracy superiority and novelty are not established.
The four arms compare full recipes. Separate random/PCA priors and a no-prior-loss
ablation would be needed to isolate transition information from regularization.

## Run on the remote machine

After copying/syncing this folder:

```bash
python -m pip install -r Adapter_BBCH/requirements.txt

# Metadata/fold inspection: no download, model load, or output writes.
python Adapter_BBCH/run_adapter_training.py \
  --config Adapter_BBCH/config/config.yaml --dry-run

# Recommended initial check: one fold, all four methods, full labels.
python Adapter_BBCH/run_adapter_training.py \
  --config Adapter_BBCH/config/config.yaml \
  --fold-id 1 --out-dir results_adapter_bbch_blocks_smoke

# Complete LOSO, full labels.
python Adapter_BBCH/run_adapter_training.py --config Adapter_BBCH/config/config.yaml

# Complete LOSO, quarter / half labels.
python Adapter_BBCH/run_adapter_training.py --config Adapter_BBCH/config/config25.yaml
python Adapter_BBCH/run_adapter_training.py --config Adapter_BBCH/config/config50.yaml

# Existing orchestrator runs all three YAML files sequentially.
bash Adapter_BBCH/orchestrator.sh
```

DINOv3 requires access to the gated public model and an authenticated HF token.
The runner resolves the pretrained HF commit and pins it across folds, methods,
and inference. Local supervised backbone paths are rejected for this experiment.
The resolved commit is recorded in checkpoints; frozen public weights are not
duplicated in every checkpoint.

Paths in YAML are relative to the working directory. Explicit CLI options override
YAML. `methods: [all]` expands to the four methods above. The old `random`, `pca`
and `learned` method names belong to the preserved cached pilot and are rejected
by this runner. `embedding_cache` is also rejected: internal adapters cannot train
from final cached embeddings.

The supplied configs run full LOSO (`fold_id: 0`) at 25%, 50%, and 100%, each to a
fresh output directory. The default CLI without a config runs one fold. For clean
training-seed repeats, keep both `fold_seed` and `label_seed` fixed and change only
`seed` and `out_dir`. `label_seed + fold_id` determines nested label subsets;
`seed + fold_id` determines initialization and training randomness. Use the same
label seed for matched budget comparisons. Tune hyperparameters,
including `prior_lambda`, on validation stations only.

## Memory and input policy

All arms use 224-pixel tiles, stride 112, **at most eight deterministically
selected tiles per image**, and compact 2x2 patch features plus CLS. The tile cap
is identical for prior estimation, training, validation and inference. This
differs from the earlier all-tile frozen-cache pilot, so these new metrics should
be compared within this experiment. `max_tiles: 0` retains all tiles and may need
substantially more memory. With fewer than eight tiles all available tiles are
used. Training augmentation is off in the supplied configs.

Default image batch size is 2 with 8 accumulation steps (16 images per full
optimizer update). `tile_batch_size: 8` chunks encoder calls, but autograd still
retains the adapted-layer graphs until backward; chunking does not remove that
memory cost. For OOM, first lower `batch_size` and increase `accumulation_steps`;
use the same settings for every method. Keep tile geometry/cap fixed across arms.
`amp: bf16` is optional on supported CUDA GPUs; `fp16` enables gradient scaling;
`none` is FP32. This runner is single-device eager PyTorch: no DDP, compile, or
gradient checkpointing. The block wrapper relies on the HF DINOv3 `norm2`/Q/V
interfaces and fails on unsupported models rather than silently adapting elsewhere.

## Leakage, budgets and outputs

LOSO still holds out physical stations, with separate validation stations.
Only selected training labels enter prior construction, adapter fitting and
pooling/head fitting. Validation/test remain complete. Label subsets are nested
for a fixed label seed using the existing station/year/stage-stratified permutation
policy. Ceil rounding and at least one image per stratum can raise realized
coverage above the requested percentage. Because this sampling uses stage labels
to define strata, it simulates label budgets rather than a label-blind acquisition
policy. Unreadable images fail the run; they are never silently removed.

Validation macro-F1 over all seven classes selects the checkpoint; test is
evaluated once afterward. Each method writes `history.csv`, `best_model.pt`,
validation/test predictions and metrics, confusion matrices and per-class metrics.
Each fold records station splits, the exact selected training labels, per-layer
basis support and basis tensors. Root outputs include `run_config.json`,
`all_test_metrics.csv`, `combined_test_predictions.csv`, `comparison_summary.csv`
and `aggregate_<method>.json`. Parameter counts and requested/realized label
coverage are recorded. Counts summed across LOSO training folds repeat images;
they are not unique annotation counts.

Checkpoints contain trained pooling/head/adapter tensors plus transition-prior
buffers, model configuration and the frozen base's HF commit. They are a new
format and cannot be opened by the old cached-feature inference tool.

```bash
python Adapter_BBCH/infer_image.py \
  --checkpoint results_adapter_bbch_blocks_100/fold_1/transition/best_model.pt \
  --image data/02.02/2015/K1/10X/example.jpeg --device cuda
```

Inference needs just an image, the checkpoint, and access to the pinned base
weights. Labels, dates and station identifiers are not model inputs.

## Optional remote tests and old pilot

No training or tests were run locally for this change. New synthetic tests use a
tiny randomly initialized DINOv3 with no downloads; they check zero-initialization,
frozen-weight gradients, adapter budgets, intermediate-coordinate estimation,
checkpoint reconstruction and config behavior. Run them before the remote pilot:

```bash
python -m unittest discover -s Adapter_BBCH/tests -p 'test_block*.py' -v
```

The earlier post-pooling pilot remains available as `run_cached_training.py`,
with original configs copied under `config_cached/`, old inference through
`infer_cached_image.py`, and instructions in [README_cached.md](README_cached.md).
The earlier tests now import the cached runner explicitly.
