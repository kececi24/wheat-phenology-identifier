# Frozen DINOv3 loss comparison

This pilot changes the training objective, using original pretrained DINOv3
features and an identical small image classifier in every arm. It does not
fine-tune a ViT, use adapters, or require a temporal window at inference.
The transition objective is a research starting point, not a novelty claim.

## Research question and experiments

Does the ordering of crop development provide useful supervision beyond
independent image labels, especially when only some images are labeled?
We compare four losses at 25%, 50%, and 100% label coverage. This gives
$4 \times 3 = 12$ method/budget combinations per LOSO fold with one training
seed, not independent seed repetitions.

| Method | Objective |
| --- | --- |
| `hard_ce` | Cross entropy against the canonical hard stage |
| `soft_ce` | Cross entropy against the existing transition-softened target |
| `ordinal` | Mean squared difference between predicted and target cumulative probabilities, across the first six thresholds |
| `transition` | Soft CE on selected labels plus `path_weight` times the mean season path loss |

All use fixed mean pooling of cached descriptors and tiles, parameter-free
LayerNorm, and a 32-unit GELU MLP with dropout. The ordinal control changes only
the loss; it is not CORAL and does not introduce a different classifier.
The ordinal loss uses class ranks, not numerical distances between BBCH codes.

The following sections define the implemented objectives and their differences.

## Image model and notation

The class order is:

$$
(\mathrm{BBCH0},\mathrm{BBCH1},\mathrm{BBCH2},\mathrm{BBCH3},
\mathrm{BBCH5},\mathrm{BBCH6\_7},\mathrm{BBCH8}).
$$

Indices $0,\ldots,K-1$, with $K=7$, encode this order. They do not represent
equal biological distances or equal stage durations. Let $q_i$ be an image's
soft target distribution and $y_i=\arg\max_c q_{i,c}$ its hard label.
$\mathcal L$ denotes the selected labeled training images.

Frozen DINOv3 descriptors are averaged within tiles, then across tiles, to
produce an image vector $z_i$. The common trainable head computes:

$$
h_i=\operatorname{Dropout}_{0.2}
\left(\operatorname{GELU}(W_1\operatorname{LN}(z_i)+b_1)\right),
\qquad
p_i=\operatorname{softmax}(W_2h_i+b_2).
$$

LayerNorm has no trainable affine parameters. The default hidden width is 32.
With 768-dimensional features this head has 24,839 trainable parameters.
Dropout is disabled for evaluation. Backbone and pooling are fixed.

## Supervised control losses

### Hard-label CE

$$
\mathcal L_{\mathrm{hard}}=
-\frac{1}{|\mathcal L|}\sum_{i\in\mathcal L}\log p_{i,y_i}.
$$

Each selected image supplies one exact class target. Class order and temporal
relationships are not explicitly part of this objective.

### Soft-label CE

$$
\mathcal L_{\mathrm{soft}}=
-\frac{1}{|\mathcal L|}
\sum_{i\in\mathcal L}\sum_{c=0}^{K-1}q_{i,c}\log p_{i,c}.
$$

The existing label builder makes adjacent-stage mixtures near milestone dates,
controlled by `transition_days` (default 2). For example, a target may assign
0.6 to one stage and 0.4 to its neighbor. Comparing soft CE with hard CE tests
the benefit of this label softening on the same selected images.

### Ordinal CDF loss

Define predicted and target cumulative probabilities:

$$
F_i(k)=\sum_{c=0}^{k}p_{i,c},
\qquad
Q_i(k)=\sum_{c=0}^{k}q_{i,c}.
$$

The implemented ordinal objective is:

$$
\mathcal L_{\mathrm{ordinal}}=
\frac{1}{|\mathcal L|(K-1)}
\sum_{i\in\mathcal L}\sum_{k=0}^{K-2}
\left(F_i(k)-Q_i(k)\right)^2.
$$

The final threshold is omitted because both cumulative probabilities equal
one there. Moving probability farther from its target generally affects more
thresholds. This control uses the same soft targets and softmax classifier;
it is pure squared CDF loss, with no CE mixture. Its loss scale differs from CE.

## Transition objective

### Allowed stages and admissible histories

Let season $s$ contain $T_s$ chronological training images. At position $t$:

$$
A_{s,t}=
\begin{cases}
\{c:q_{s,t,c}>0\}, & \text{if the image has a selected label},\\
\{0,\ldots,K-1\}, & \text{otherwise}.
\end{cases}
$$

Unselected target values are not consulted when making this mask. Soft CE
uses target probability values, while the path constraint uses support only:
0.6/0.4 and 0.9/0.1 allow the same two states.

An admissible history respects these masks and never moves backward:

$$
\mathcal A_s=
\left\{\mathbf y:
y_t\in A_{s,t}\ \forall t,\quad
y_t\le y_{t+1}\ \text{for }t<T_s
\right\}.
$$

### Marginal probability and combined loss

The image head predicts observations independently. These probabilities define
a factorized distribution over histories. We reward the total probability
assigned to admissible histories:

$$
Z_s=\sum_{\mathbf y\in\mathcal A_s}
\prod_{t=1}^{T_s}p_{s,t,y_t},
\qquad
\mathcal L_{\mathrm{path},s}=-\frac{1}{T_s}\log Z_s.
$$

For the set $\mathcal S$ of training seasons with selected anchors:

$$
\boxed{
\mathcal L_{\mathrm{transition}}=
\mathcal L_{\mathrm{soft}}+
\lambda\frac{1}{|\mathcal S|}
\sum_{s\in\mathcal S}\mathcal L_{\mathrm{path},s}
}
$$

$\lambda$ is `path_weight`, default 1.0. Supervised CE averages over selected
images; the path term averages length-normalized losses over anchored seasons.
These weightings differ. AdamW weight decay is applied separately to the head
in every arm. Setting `path_weight: 0` recovers the soft CE objective.

### Dynamic programming

We compute the path sum without enumerating histories. Let $a_t(c)$ denote
log probability mass of admissible prefixes ending in stage $c$:

$$
a_1(c)=
\begin{cases}
\log p_{s,1,c}, & c\in A_{s,1},\\
-\infty, & \text{otherwise},
\end{cases}
$$

$$
a_t(c)=
\begin{cases}
\log p_{s,t,c}+
\operatorname{LSE}_{j\le c} a_{t-1}(j), & c\in A_{s,t},\\
-\infty, & \text{otherwise},
\end{cases}
\qquad
\log Z_s=\operatorname{LSE}_c a_{T_s}(c).
$$

$\operatorname{LSE}$ is log-sum-exp. Unreachable states are omitted before
differentiable reductions, avoiding invalid all-negative-infinity gradients.
An empty admissible set raises an error.

### Example and limiting case

Suppose Monday is anchored to BBCH2 and Friday to BBCH3, with unlabeled
observations between them. Several crossing times are admissible. The loss
sums their probabilities rather than selecting an exact crossing day.
These brackets follow from selected observations; the implementation does not
add measured transition-date intervals.

If every image has a selected one-hot label and the labels are monotone,
there is exactly one admissible history:

$$
\mathcal L_{\mathrm{path},s}
=-\frac{1}{T_s}\sum_{t=1}^{T_s}\log p_{s,t,y_t}.
$$

This is season-averaged hard CE. New temporal supervision arises mainly when
sparse or soft anchors leave multiple histories possible.

### Assumptions and single-image inference

Paths can stay in a stage or advance, including skipping stages across missing
observations. They cannot regress. At selected labeled images, allowed stages
are the nonzero support of `active_target`; at unselected images, all seven
stages are allowed. The loss sums paths in log space. It does not impose a
constant developmental speed or a separate duration model.

Soft CE retains the target probabilities (for example 0.6/0.4); the path term
uses their support only. Existing `transition_days` is an annotation heuristic,
**not a measured uncertainty interval**. This implementation does not invent
additional crossing-date brackets. At 100% labels with `transition_days: 0`,
there is one admissible path when labels are monotone: the path term reduces to
season-normalized hard CE. Do not interpret that setting as new supervision.

Sequences are grouped by physical station and annotation year/season, never
across stations. The shared builder excludes preplant/postharvest targets.
This assumes one growing cycle per station/year; multiple cycles must receive
separate season identifiers before using this loss. Contradictory constraints
and duplicate sequence dates fail explicitly. Unanchored seasons are omitted
from the path objective. Dates/stations are training organization information,
not classifier inputs.

At inference, the prediction is simply:

$$
\hat y(x)=\arg\max_c p_\theta(c\mid x).
$$

No temporal decoding is applied. Therefore monotone predictions on unseen
sequences are encouraged through training, but are not guaranteed.

## What each comparison establishes

| Comparison | Question | Qualification |
| --- | --- | --- |
| Soft CE vs hard CE | Does transition label softening help? | Same labeled observations |
| Ordinal vs soft CE | Does cumulative ordinal supervision help? | Loss and gradient scales differ |
| Transition vs soft CE | Do admissible developmental histories help? | Reduced budgets also introduce unlabeled training observations |
| 25% vs 50% vs 100% | How does performance depend on label coverage? | Nested, label-stratified simulated budgets |

Each LOSO fold holds out one physical station for test and, by default, two
other stations for validation. Remaining stations provide training data.
Only training labels are subsampled; validation/test labels remain complete.
All arms share cache, head size, initial weights, dropout seed, selected labels,
optimizer settings and epoch limit. All training images are forwarded in the
common batch, but CE/ordinal gradients use selected positions only.

Defaults are AdamW with learning rate $10^{-3}$, weight decay
$3\times10^{-4}$, at most 200 full-batch updates, and validation patience 25.
Checkpoints maximize validation macro-F1 over the fixed seven classes.
The selected checkpoint is then evaluated on test once.

Assess both pooled macro-F1 and mean-station macro-F1: pooled scores combine
all test images, while mean-station scores give each test station equal
weight. Outputs also include accuracy, stage-rank MAE, accuracy within one
stage, quadratic weighted kappa, per-class scores and confusion matrices.
Raw loss values across the four objectives are not directly comparable.

This initial study tests usefulness on this dataset. Repeated training seeds
with fixed folds/label subsets, validation-only tuning of $\lambda$, and a
baseline with equal access to unlabeled images are needed for stronger claims.

## Run remotely

From the repository root:

```bash
python -m pip install -r Loss_BBCH/requirements.txt
python Loss_BBCH/run_training.py --config Loss_BBCH/config/config.yaml --dry-run
python -m pytest Loss_BBCH
python Loss_BBCH/run_training.py --config Loss_BBCH/config/config.yaml
```

Edit `label_path`, `data_path`, and `embedding_cache` first. Reuse an original
frozen DINOv3 cache, such as `results_adapter_frozen_cache/vit_embeddings.pt`.
See [the existing cache extraction instructions](../Adapter_BBCH/README_cached.md#2-build-a-frozen-feature-cache-once)
if it does not exist. Do not use supervised/fold-fine-tuned caches. The loader
checks recorded provenance and fails on missing images rather than dropping
held-out rows. This folder depends on the existing `Adapter_BBCH` and
`DINOv3_BBCH` modules; copy it as part of the repository.

YAML defaults can be overridden by explicit CLI arguments. Paths are relative
to the working directory. `fold_id: 0` runs all LOSO folds; use `--fold-id 1`
and a new output directory for a first remote trial. Fresh output directories
are required to avoid mixing stale results; there is no resume implementation.

The supplied 25%, 50%, and 100% configs share fold/label/training seeds. Run:

```bash
python Loss_BBCH/run_training.py --config Loss_BBCH/config/config25.yaml
python Loss_BBCH/run_training.py --config Loss_BBCH/config/config50.yaml
python Loss_BBCH/run_training.py --config Loss_BBCH/config/config.yaml
```

Training uses **one full-batch optimizer update per epoch** in all arms. The
default 200 epochs is therefore not comparable to 200 minibatch epochs in
previous experiments. Only cached features and the small head need gradients.
All methods share initialization, selected labels and validation selection.
Checkpoint selection uses validation macro-F1 over the fixed seven classes;
the test station is evaluated once after selection, without sequence decoding.

Outputs include `comparison_summary.csv` (pooled and mean-station metrics),
`combined_test_predictions.csv`, per-fold checkpoints and histories,
`selected_training_labels.csv`, and `sequence_anchor_audit.csv`. The run config
records source/label/cache hashes and runtime versions. Per-fold validation
predictions preserve the selected checkpoint evidence.

The supplied tests cover exact path enumeration and gradients, the hard-label
CE limit, contradictory constraints, hidden-label invariance, season separation,
YAML validation, and a synthetic four-arm training/checkpoint integration run.
They require the project dependencies but do not download a backbone.

## Interpretation and experimental safeguards

- The default is the public DINOv3 model already used in the project, not a
  newly initialized or fine-tuned vision backbone.
- Label coverage uses nested stratification by station/season/stage. It can
  exceed the nominal fraction and is a simulated budget, not label-blind
  annotation acquisition. In-season eligibility also comes from annotations.
- At reduced coverage, the transition arm uses additional **unlabeled images
  from training stations**. CE/ordinal use selected images only. A gain would
  not isolate the loss from access to unlabeled data; add a matched
  semi-supervised baseline before making that claim.
- The path term averages normalized losses over anchored seasons; supervised
  losses average over selected images. It does not station-balance both terms.
- For seed repeats, change `seed` while keeping `fold_seed` and `label_seed`
  fixed. Tune `path_weight` on validation only. Do not select it using test F1.
- Unconstrained monotonicity alone allows constant predictions. Anchors and
  soft CE are retained to counter that degeneracy; they do not guarantee gains.
- Ordered labels and probabilistic path marginalization have established
  precedents (CTC and constrained sequence models). This is a controlled pilot.

