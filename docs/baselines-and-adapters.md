# Baselines, Adapters, and Historical Comparisons

This document provides a technical guide to the baseline pipelines, parameter-efficient fine-tuning (PEFT) adapters, and historical model progressions in the repository.

To isolate the contributions of temporal Transformer modeling, foundation vision representations, metadata fusion, and parameter adaptation, the repository includes several distinct paradigms:

| Paradigm | Directory | Temporal Model | Visual Input | Metadata | Primary Role |
| --- | --- | --- | --- | --- | --- |
| **Causal Temporal DINOv3** | `DINOv3_BBCH/` | 21-day Causal Transformer | Dense DINOv3 (tiled, 2x2 grid + CLS) | Gated Sowing, GDD, Lat/Lon/Elev | **Primary Production System** |
| **Single-Image PEFT** | `Adapter_BBCH/` | None (Single-day) | Raw Image (DINOv3 + PEFT in last 2 blocks) | None (Vision only) | Parameter-efficient adaptation & transition priors |
| **Single-Day Linear Probe** | `DINOv3_BBCH_Linear/` | None (Single-day) | Dense DINOv3 (tiled, partial fine-tune) | Direct GDD + Lat/Lon/Elev | Non-temporal deep learning ablation |
| **Metadata + ExG Baseline** | `Baseline_BBCH/` | 21-day Causal Rolling Stats | None (Illumination-tolerant ExG statistics) | Weather, GDD, Calendar, Lat/Lon/Elev | Non-deep-learning tabular/vegetation control |
| **Historical 2D CNN** | `archive/CNN/` | 31-day Causal Transformer | EfficientNetV2 / ConvNeXt | Concatenated days since planting | Legacy 2D CNN benchmark |
| **Historical 3D CNN** | `archive/CNN3D/` | Spatiotemporal 3D Convolutions | Raw Video Tensor $[T, C, H, W]$ | Concatenated days since planting | Raw spatiotemporal video benchmark |
| **Historical DINOv2** | `archive/DINOv2_BBCH/` | 31-day Causal Transformer | Frozen DINOv2 (1 vector per tile) | Gated Calendar, Weather, Coordinates | Foundation model predecessor |

---

## 1. Single-Image Parameter-Efficient Fine-Tuning (`Adapter_BBCH`)

The `Adapter_BBCH/` experiment investigates parameter-efficient transfer learning on **single raw images**, operating without temporal windows, weather features, or calendar signals. It evaluates how effectively small, trainable adapter modules inserted into a frozen foundational vision backbone can adapt to agricultural phenology classification under varying annotation budgets.

### Target Architecture and Insertion Points

Adapters are inserted into the **last two transformer blocks** of the pretrained Meta DINOv3 ViT-B/16 backbone (`facebook/dinov3-vitb16-pretrain-lvd1689m`):
- All pretrained backbone parameters (attention weights, feed-forward layers, LayerNorms, LayerScale parameters) remain strictly frozen and in evaluation mode.
- Input images are tiled into $224 \times 224$ patches (stride 112) with a deterministic cap of at most **8 tiles per image**.
- Patch features within each tile are adaptively pooled to a $2 \times 2$ grid plus CLS (5 descriptors per tile).
- All adapter arms share the exact same downstream architecture:
  $$\text{HierarchicalDenseTilePooler} \longrightarrow \operatorname{LayerNorm}(768) \longrightarrow \operatorname{Linear}(768, 128) \longrightarrow \operatorname{GELU} \longrightarrow \operatorname{Dropout}(0.2) \longrightarrow \operatorname{Linear}(128, 7)$$

### Evaluated PEFT Methods

```text
(A) Frozen Baseline (none)
    x ──► [ Frozen DINOv3 Block ] ─────────────────────────► y

(B) LoRA (lora)
    x ──► [ Frozen W_q, W_v ] ──(+)──► Attention Output
               ▲                  │
               └── (alpha/r) B·A ─┘   (A: Kaiming, B: Zero)

(C) AdaptFormer Bottleneck (standard)
    norm2(z) ──► [ A (width 6) ] ──► GELU ──► [ B (init 0) ] ──(+)──► y_block
                 (Random Kaiming down-projection)

(D) Transition Prior Bottleneck (transition)
    norm2(z) ──► [ A0 + E ] ───────► GELU ──► [ B (init 0) ] ──(+)──► y_block
                 (A0: SVD Orthonormal Basis, E: Trainable deviation)
```

1. **`none` (Frozen Backbone Baseline)**:
   - 0 adapter parameters.
   - Evaluates the zero-shot representation capability of frozen DINOv3 features when only the pooling and classification head are trained.

2. **`lora` (Low-Rank Adaptation)**:
   - Injected into the Query ($W_q$) and Value ($W_v$) projection layers of the multi-head attention mechanism within each of the last two blocks.
   - Forward pass:
     $$W x + \frac{\alpha}{r} B A x$$
   - Rank $r = 3$, scale $\alpha = 3.0$. $A \in \mathbb{R}^{3 \times 768}$ is initialized via Kaiming uniform; $B \in \mathbb{R}^{768 \times 3}$ is initialized to zero.
   - Total adapter parameters for ViT-B across two blocks: **18,432 weights**.

3. **`standard` (AdaptFormer-Style Parallel Bottleneck)**:
   - A parallel bottleneck branch added alongside the frozen multi-layer perceptron (FFN) block.
   - Hooked to the output of `norm2` (the exact input to the pretrained FFN):
     $$\delta = B \cdot \operatorname{GELU}(A z)$$
   - Down-projection matrix $A \in \mathbb{R}^{6 \times 768}$ is initialized via Kaiming uniform.
   - Up-projection matrix $B \in \mathbb{R}^{768 \times 6}$ is initialized to zero, ensuring the adapter is an exact identity mapping at initialization.
   - Added outside the frozen block's LayerScale and drop-path: $\mathbf{y} = \mathbf{y}_{\text{orig}} + \delta$.
   - Total adapter parameters: **18,432 weights**.

4. **`transition` (Transition Prior Adapter)**:
   - Explores whether biological stage-transition directions can guide PEFT parameter space.
   - **Offline Orthonormal Prior Estimation ($A_0$)**: Prior to training, the unmodified pretrained encoder extracts `norm2` activations across training-station images. Stage prototypes $\mu_s$ are computed per station-year. Differences between adjacent supported stages ($\Delta_s = \mu_{s+1} - \mu_s$) are averaged across years and physical stations. Singular Value Decomposition (SVD) of this difference matrix produces an orthonormal basis matrix $A_0 \in \mathbb{R}^{6 \times 768}$ for each block.
   - **Formulation**:
     $$A = A_0 + E, \quad \delta = B \cdot \operatorname{GELU}((A_0 + E) z)$$
     where $A_0$ is a frozen buffer, $E \in \mathbb{R}^{6 \times 768}$ is a trainable deviation initialized to zero, and $B$ is initialized to zero.
   - **Regularized Loss**:
     $$\mathcal{L} = \mathcal{L}_{\text{hybrid}} + \lambda_{\text{prior}} \cdot \frac{1}{L} \sum_{l=1}^L \|E_l\|_F^2$$
     penalizing deviation from the estimated biological transition subspace.

### Label Budget Sampling

`Adapter_BBCH` evaluates these methods across three simulated annotation budgets:
- **100% Labels**: Full training set.
- **50% Labels**: Stratified sub-sampling keeping 50% of station-year-stage images.
- **25% Labels**: Severe low-data regime keeping 25% of images.

Configurations are managed via [`Adapter_BBCH/config/config.yaml`](../Adapter_BBCH/config/config.yaml), [`config50.yaml`](../Adapter_BBCH/config/config50.yaml), and [`config25.yaml`](../Adapter_BBCH/config/config25.yaml).

---

## 2. Metadata + ExG Baseline (`Baseline_BBCH`)

The `Baseline_BBCH/` pipeline provides a rigorous, non-deep-learning reference to answer: *How much predictive performance can be achieved purely from weather, thermal time, calendar progression, coordinates, and classical color vegetation indices without any learned neural image representations?*

### Feature Engineering

The pipeline extracts a comprehensive tabular feature vector for each daily observation:

```text
Raw Daily Photo  ──►  Illumination-Tolerant ExG  ──►  Daily ExG Stats (9)  ──┐
                                                                             ├──► Causal Rolling (21d) ──► HistGradientBoosting
Weather Data     ──►  Temperatures & Rain (4)    ──►  Missing Flags (4)    ──┤    Mean, Std, Min, Max,     Classifier
Thermal Time     ──►  Daily & Cumulative GDD (2) ────────────────────────────┤    Delta (x_t - x_0)
Calendar & Space ──►  Days Since Sowing (1)      ──►  Lat / Lon / Elev (3) ──┘
```

1. **Illumination-Tolerant Excess Greenness ($ExG$)**:
   - For each daily image, RGB channels are normalized by total pixel luminance:
     $$r = \frac{R}{R + G + B + \epsilon}, \quad g = \frac{G}{R + G + B + \epsilon}, \quad b = \frac{B}{R + G + B + \epsilon}$$
     $$ExG = 2g - r - b$$
   - Distribution summary statistics are computed:
     - `exg_mean`, `exg_std`: Central tendency and spread of canopy greenness.
     - `exg_p10`, `exg_p50`, `exg_p90`: Robust 10th, 50th, and 90th percentiles.
     - `exg_positive_fraction`: Fraction of pixels where $ExG > 0.05$ (proxy for vegetation canopy coverage).
     - `green_dominance_fraction`: Fraction of pixels where green strictly exceeds red and blue ($G > R \land G > B$).
     - `brightness_mean`: Average unnormalized pixel intensity.
     - `exg_missing`: Binary indicator flagging unreadable/missing images.

2. **Causal Rolling Statistics (21-Day Window)**:
   - For every daily observation, strictly causal rolling window summaries are computed over the preceding 21 calendar days for `exg_mean`, `exg_std`, `exg_positive_fraction`, `green_dominance_fraction`, `weather_tavg_norm`, `weather_prcp_norm`, `weather_gdd_norm`, and `weather_gdd_cum_norm`.
   - Five aggregation statistics are extracted per source variable:
     $$\text{mean}, \quad \text{std}, \quad \text{min}, \quad \text{max}, \quad \text{delta} = (x_t - x_{t-21})$$
   - The `delta` metric explicitly captures the temporal rate of change (e.g., green-up velocity during tillering or canopy browning rate during ripening).

3. **Geographic and Meteorological Context**:
   - Normalized coordinates: `lat_norm`, `lon_norm`, `elevation_norm`.
   - Daily weather and missingness: `weather_tavg_norm`, `weather_tmin_norm`, `weather_tmax_norm`, `weather_prcp_norm`, and four missingness flags.
   - Thermal time: daily Growing Degree Days (`weather_gdd_norm`) and cumulative GDD (`weather_gdd_cum_norm`).
   - Calendar time: normalized days since planting ($(\text{date} - \text{planting}) / 365.0$).

### Classification Algorithm and Execution

- **Model**: Scikit-learn's `HistGradientBoostingClassifier`, natively handling continuous tabular values and missing feature indicators.
- **Validation-Based Stopping**: Number of boosting iterations is tuned per fold using validation macro-F1.
- **High-Throughput Parallelism**: ExG extraction across thousands of images is parallelized using CPU process pools (`--exg-workers`), and independent LOSO folds train in parallel (`--fold-workers`).

---

## 3. DINOv3 Linear Single-Day Pipeline (`DINOv3_BBCH_Linear`)

The `DINOv3_BBCH_Linear/` pipeline acts as the single-day deep learning counterpart to `DINOv3_BBCH`.

### Architecture and Data Flow

```text
Single Target Image (10X) ──► Overlapping Tiles (224x224, stride 112)
                                      │
                                      ▼
                        DINOv3 ViT-B/16 (Final Block & Norm Unfrozen)
                                      │
                                      ▼
                        Dense Descriptors (2x2 Patch Grid + CLS)
                                      │
                                      ▼
                        Hierarchical Patch & Tile Attention Pooling
                                      │
                                      ▼
                        Single Visual Embedding (d = 768)
                                      │
                                      ├── Concat Normalized Cumulative GDD (1)
                                      └── Concat Normalized Lat, Lon, Elev (3)
                                      │
                                      ▼
                        LayerNorm(772) ──► Dropout(0.2) ──► Linear(772, 7)
```

### Purpose and Ablation Role

- **No Temporal Window**: Unlike `DINOv3_BBCH`, there is no 21-day or 31-day temporal sequence, no causal mask, and no temporal Transformer.
- **Direct Concatenation**: Visual features ($d=768$) and 4 metadata scalars (`weather_gdd_cum_norm`, `lat_norm`, `lon_norm`, `elevation_norm`) are directly concatenated into a 772-dimensional vector.
- **Linear Classifier**: The classifier head consists of a single linear projection:
  $$\hat{\mathbf{y}} = \operatorname{Linear}(\operatorname{Dropout}(\operatorname{LayerNorm}(\mathbf{v} \,\|\, \mathbf{m})))$$
- **Ablation Utility**: Comparing `DINOv3_BBCH_Linear` directly against `DINOv3_BBCH` under identical LOSO folds isolates the precise quantitative benefit of **causal temporal sequence modeling** over static single-day classification.

---

## 4. Historical Pipeline Comparisons (`archive/`)

The `archive/` directory preserves earlier model generations developed during project evolution.

### Historical 2D CNN (`archive/CNN/`)

- **Visual Encoders**: Torchvision CNN backbones, primarily `efficientnet_v2_s`, `efficientnet_v2_m`, and `convnext_tiny`.
- **Temporal Modeling**: Causal temporal Transformer with CLS sequence aggregation over 31-day causal windows.
- **Execution Modes**: Supported both frozen precomputed embedding caching and end-to-end full-image backpropagation.
- **Observed Behavior & Failure Modes**:
  - Training accuracy frequently reached 90–99%, while validation accuracy plateaued around 60–70%.
  - CNN inductive biases (local receptive fields, translation equivariance) proved vulnerable to station-level visual shifts (soil color, row textures, camera mounting angles).
  - The CNN backbones overfit heavily to non-phenological background textures, motivating the transition to self-supervised foundation vision models (DINOv2 / DINOv3).

### Historical 3D CNN (`archive/CNN3D/`)

- **Concept**: Treat the daily image window as a spatiotemporal video volume $[B, C, T, H, W]$ ($T = 15$ to $31$ days).
- **Architecture**: `SingleStream3DCNN` and `MultiScale3DCNN` applying 3D spatio-temporal convolutions over time, height, and width simultaneously.
- **Observed Behavior & Failure Modes**:
  - High memory footprint: 3D convolutional feature maps across 31 high-resolution daily frames severely restricted batch sizes ($B=1$ or $B=2$).
  - Sensitivity to regularization: Because 3D kernels had to learn both visual spatial filters and temporal evolution from scratch on a small dataset (~15 stations), models oscillated between gross overfitting (under light regularization) and severe underfitting (under strong dropout and augmentations).

### Historical DINOv2 Pipeline (`archive/DINOv2_BBCH/` & Variants)

- **Backbone**: Frozen `facebook/dinov2-base`.
- **Tiling & Pooling**: Images were tiled at $224 \times 224$ (stride 112 or 224), but each crop was pooled into a **single 768-d global vector** (no dense patch grid).
- **Dual-Camera Exploration (`DINOv2_Gated`, `DINOv2_CrossAttention`)**:
  - Explored fusing 1X wide-angle canopy imagery with 10X micro close-up imagery via learned gating and cross-attention mechanisms.
  - **Empirical Finding**: Adding the 1X macro stream did not improve held-out station generalization over the 10X micro stream alone. Wide-angle views introduced noisy background variability (sky, horizon, soil patches) without adding discriminative morphological cues.
  - This conclusion established the 10X micro stream as the standard for all subsequent architectures.

---

## 5. Architectural Progression Summary

The following table summarizes the evolutionary progression across model families:

| Architecture Family | Image Representation | Spatial Pooling | Temporal Modeling | Metadata Integration | Key Limitation / Finding |
| --- | --- | --- | --- | --- | --- |
| **Historical 2D CNN** (`archive/CNN`) | EfficientNetV2 / ConvNeXt | Global Average Pooling | Causal Transformer (31d) | Concatenated days since planting | Severe station overfitting (train 95% vs val 65%). |
| **Historical 3D CNN** (`archive/CNN3D`) | Raw Image Tensor $[T, C, H, W]$ | 3D Spatial Convolutions | 3D Temporal Convolutions | Concatenated days since planting | Prohibitive VRAM; brittle regularization balance. |
| **DINOv2 Baseline** (`archive/DINOv2_BBCH`) | Frozen DINOv2-Base | Tile Attention (1 vector per tile) | Causal Transformer (31d) | Concatenated / Early Gated | Tile vectors lost intra-crop spatial arrangements. |
| **DINOv3 Linear** (`DINOv3_BBCH_Linear`) | Fine-tuned DINOv3 ViT-B/16 | Hierarchical Patch + Tile | None (Single day) | Direct Concatenation | Lacks temporal context; relies heavily on static day. |
| **DINOv3 Adapters** (`Adapter_BBCH`) | DINOv3 + PEFT (LoRA / Bottleneck / Prior) | Hierarchical Patch + Tile | None (Single day) | None (Pure visual) | Parameter-efficient adaptation without full backbone training. |
| **ExG Baseline** (`Baseline_BBCH`) | Illumination-Tolerant ExG Stats | Spatial Mean/Std/Percentiles | 21-day Causal Rolling Statistics | Tabular (GDD, Lat/Lon, Calendar) | Measures non-deep-learning ceiling (~tabular performance). |
| **DINOv3 Full Pipeline** (`DINOv3_BBCH`) | DINOv3 ViT-B/16 (Frozen or Fine-Tuned) | Hierarchical Patch (2x2) + Tile | Causal Transformer (21d) | Gated Residual MLPs (Calendar, GDD, Lat/Lon) | **Current State of the Art**: High macro-F1 (0.846) with 100% adjacent-stage accuracy. |
