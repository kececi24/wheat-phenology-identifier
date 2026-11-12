# Project Overview & System Architecture

This document outlines the core problem statement, biological foundation, developmental stage taxonomy, and architectural design choices of the **Wheat Phenology Identifier** system.

---

## Problem Statement

### In-Field Daily Wheat Phenology Prediction

Phenology—the study of periodic plant life cycle events—is essential for modern agronomic decision-making, including optimal timing for fertilization, irrigation, pesticide application, and harvest yield forecasting. Traditional phenology monitoring relies on manual field scouting, which is labor-intensive, subjective, and difficult to scale across broad geographical areas.

This project automates wheat phenological stage estimation using stationary, fixed-camera digital camera stations deployed in agricultural test fields across Turkey.

```text
Fixed In-Situ Camera  ──►  Daily 10X Micro Photos  ──►  Causal Window  ──►  Phenological Stage
(e.g., K1 / K2)            (High-res crop canopy)       (Past 20 + Day t)    (BBCH0 ... BBCH8)
```

### Strict Temporal Causality

In an operational agricultural deployment, decisions are made on a day-by-day basis. The model must operate strictly causally:
- When predicting the phenological stage for calendar day $t$, the system may condition only on visual observations and metadata from days $\le t$ (specifically, a historical context window of $W$ days: $[t - W + 1, \, \dots, \, t]$).
- **Anti-causal lookahead is forbidden**: future days $t+1, t+2, \dots$ are never accessible to the feature extractors or temporal encoders.
- Missing observation days (e.g., camera failures, adverse weather outages) are explicitly represented via causal temporal masks rather than interpolated using future frames.

### Domain Challenges in Field Imagery

Fixed agricultural cameras face substantial real-world visual variance:
1. **Extreme Illumination and Atmospheric Shifts**: Harsh direct sunlight, severe specular highlights, deep shadows, overcast diffusion, golden-hour color casts, dense fog, frost, and snow accumulation.
2. **Sensor and Environmental Perturbations**: Rain droplet distortions on optical dome enclosures, lens dust, wind-induced canopy motion blur, and minor mechanical vibration.
3. **Station Domain Shifts**: Significant geographic domain shifts across distinct field stations, including differences in soil hue (dark loam vs pale chalky clay), distinct wheat cultivar cultivars with varying awn morphology and canopy architecture, and varying row orientations.
4. **Subtle Morphological Boundaries**: Phenological transitions (such as tillering into stem elongation, or heading into anthesis) manifest gradually over several days rather than in sudden macroscopic jumps.

---

## The BBCH Scale and Developmental Progression

### The BBCH Phenological Framework

The model adopts the internationally standardized **BBCH scale** (Biologische Bundesanstalt, Bundessortenamt und CHemische Industrie) for cereal crops. In this repository, the continuous lifecycle is discretized into seven active operational stages derived from canonical field milestone annotations:

| Class Label | BBCH Numerical Code | Developmental Stage | Phenological Description |
| --- | --- | --- | --- |
| `BBCH0` | 00–09 | Germination / Emergence | Sowing to seedling emergence; bare soil transitioning to early coleoptile shoots breaking the crust. |
| `BBCH1` | 10–19 | Leaf Development | Emergence to tillering; development of first true leaves; seedling establishment. |
| `BBCH2` | 20–29 | Tillering | Main shoot develops side tillers; canopy begins closing across rows. |
| `BBCH3` | 30–39 | Stem Elongation | Jointing / node formation; pseudostem erect, rapid vertical stem extension, flag leaf emergence. |
| `BBCH5` | 50–59 | Inflorescence Emergence | Heading; emergence of the spike (inflorescence) from the flag leaf sheath, awn appearance. |
| `BBCH6_7` | 60–79 | Flowering & Milk/Dough | Anthesis (flowering / visible anthers) followed by grain milk and dough development; initial loss of chlorophyll. |
| `BBCH8` | 80–89 | Ripening / Maturity | Grain hardening, complete plant senescence; canopy turns from green/yellow to uniform golden straw. |

```text
[Sowing] ──► BBCH0 ──► BBCH1 ──► BBCH2 ──► BBCH3 ──► BBCH5 ──► BBCH6_7 ──► BBCH8 ──► [Harvest]
(Emergence)   (Leaves)   (Tillers)  (Stem/Joint) (Heading)  (Flowering/Grain) (Senescence)
```

### OffSeason Handling

The non-vegetative period surrounding the crop cycle is annotated as `OffSeason` (encompassing pre-sowing fallow and post-harvest stubble/tilled soil). 
- In the recommended pipeline (`--exclude-offseason`), `OffSeason` records are excluded from training loss calculation and prediction targets.
- However, when target dates fall near the beginning of the vegetative cycle, preceding `OffSeason` days are retained in the causal window as visual and temporal context.

### Monotonicity and Soft Target Transitions

1. **Biological Monotonicity**: Plant development is strictly forward-directed in time ($S_{t} \le S_{t+1}$). Wheat cannot regress from heading back into tillering. The pipeline respects this constraint both through ordinal CDF loss formulations during training and through optional post-hoc constrained Viterbi sequence decoding during inference.
2. **Soft Milestone Targets**: The exact biological transition between stages takes place continuously over several days. Assigning a discrete one-hot label on a single cutoff date introduces arbitrary label noise. To address this, transition boundary days are assigned soft probability targets (e.g., 60% probability to the current stage, 40% probability to the adjacent transitional stage), smoothing training gradients at stage boundaries.

---

## System Design and Key Architectural Choices

The overall architecture combines high-resolution spatial feature extraction with causal temporal sequence aggregation and multimodal metadata gating:

```text
Full-Resolution In-Field Photo (10X Micro)
                 │
                 ▼
    Overlapping Tiling (224x224, stride 112)
                 │
                 ▼
    Frozen / Fold-Adapted DINOv3 ViT-B/16 Backbone
                 │
                 ▼
    Dense Descriptors (2x2 spatial patch grid + CLS)
                 │
                 ▼
    Hierarchical Patch Attention Pooling
                 │
                 ▼
    Hierarchical Tile Attention Pooling
                 │
                 ▼
    Daily Visual Embedding (d = 768)
                 │
                 ▼
    Causal Temporal Transformer (21-day causal window)
                 │
                 ▼
    Temporal Sequence Embedding (d = 768)
                 │
                 ├── Gated Residual MLP (Days Since Sowing)
                 ├── Gated Residual MLP (Cumulative GDD)
                 └── Gated Residual MLP (Normalized Lat/Lon/Elevation)
                 │
                 ▼
    Ordinal / Soft Cross-Entropy Classification Head
                 │
                 ▼
    BBCH Stage Probabilities (BBCH0 ... BBCH8)
```

### 1. Close-Up 10X Micro Stream Focus

Field stations feature two fixed cameras:
- **1X Macro Camera**: Wide-angle lens viewing the broader field plot, sky, and horizon.
- **10X Micro Camera**: Narrow field-of-view close-up focused directly on the crop canopy, leaves, tillers, and emerging spikes.

Extensive empirical evaluations across earlier DINOv2 and CNN iterations demonstrated that the **10X close-up stream alone is superior for phenology**:
- Wide-angle 1X images introduce irrelevant background variation (distant tree lines, clouds, tractor paths, horizon illumination) that promotes spurious correlation.
- Critical phenological milestones (ligule development, tiller separation, awn emergence, anther extrusion) are sub-pixel or invisible in 1X shots, but sharply resolved in 10X close-ups.
- Deploying the single 10X stream simplifies field telemetry and eliminates cross-camera alignment and synchronization issues.

### 2. Overlapping Tiling Strategy

Standard Vision Transformers resize input images directly to 224x224, destroying fine high-frequency spatial structures. To preserve sub-millimeter details across 2-megapixel to 5-megapixel camera frames:
- Each image is partitioned into overlapping tiles of size $224 \times 224$ pixels.
- A stride of 112 pixels (50% overlap) is used across both horizontal and vertical axes.
- Overlapping ensures that fine structures (such as wheat spikes or leaf tips) intersecting tile boundaries are captured fully in adjacent crops.

### 3. Foundation Vision Backbone: DINOv3 ViT-B/16

The visual encoder uses Meta's foundational self-supervised **DINOv3 ViT-B/16** (`facebook/dinov3-vitb16-pretrain-lvd1689m`):
- Pretrained on billions of curated natural images using self-distillation and masked image modeling.
- Captures semantic and structural plant features (vein orientation, textures, chlorophyll saturation, geometry) without requiring millions of supervised crop images.
- **Register Token Exclusion**: DINOv3 introduces specialized register tokens to store global background artifacts. These register tokens are explicitly stripped before spatial pooling, preventing non-localized visual noise from leaking into downstream representations.
- **Dense 2x2 Patch Representation**: Rather than condensing each 224x224 tile into a single global CLS vector, the pipeline adaptively pools ViT patch tokens to a compact 2x2 spatial grid. Retaining the 4 pooled patch tokens plus the CLS token yields 5 dense descriptors per tile, preserving spatial layout within each crop.

### 4. Hierarchical Attention Pooling

To aggregate thousands of patch tokens across dozens of tiles into a single daily image vector:
1. **Patch Attention**: A learned multi-head attention module attends over the 5 descriptors within each tile, computing a single context-aware tile embedding.
2. **Tile Attention**: A second multi-head attention module attends across all tile embeddings from the entire image, weighted by 2D tile positional encodings. This allows the model to selectively focus on salient crop structures (e.g., an emerging spike in the upper corner) while down-weighting uninformative soil or shadow patches.

### 5. Causal Temporal Sequence Modeling

Individual daily snapshots are often ambiguous; observing canopy trajectory over time provides unambiguous phenological context:
- Input: A sequence of $W$ daily embeddings ($W = 21$ causal days provides the best empirical tradeoff; $W = 31$ remains a strong baseline).
- The temporal sequence is processed by a 2-layer, 8-head **temporal Transformer encoder** with pre-layer normalization and learnable positional encodings.
- A causal attention mask enforces that timestep $i$ attends only to timesteps $j \le i$.
- Aggregation is performed using a prepended learnable `CLS` token that summarizes temporal dynamics over the causal window.

### 6. Gated Residual Multimodal Metadata Fusion

Phenology is strongly driven by thermal time and calendar progression. However, directly concatenating scalar metadata into the visual sequence risks overwhelming visual features and learning spurious correlations. 

The system implements **gated residual fusion**:
- **Calendar Signal**: Normalized days since sowing ($(\text{date} - \text{sowing}) / 365.0$) passed through a dedicated MLP.
- **Weather / Thermal Time**: Cumulative Growing Degree Days ($\text{GDD} = \max(0, \, T_{\text{avg}} - T_{\text{base}})$ with $T_{\text{base}} = 0^\circ\text{C}$) computed over the full calendar sequence and normalized.
- **Geographic Location**: Normalized station coordinates (latitude, longitude, elevation) passed through a separate MLP.
- **Residual Sigmoid Gating**: Each metadata projection is added to the visual embedding via a learnable residual gate:
  $$\mathbf{x}_{\text{fused}} = \mathbf{x}_{\text{temporal}} + \sigma(\gamma) \cdot \operatorname{MLP}_{\text{meta}}(\mathbf{m})$$
- Gate parameters $\gamma$ are initialized to $\sigma(\gamma) = 0.1$, allowing visual evidence to dominate early in training while giving the model the capacity to gradually incorporate metadata where beneficial.

---

## Evaluation Discipline: Leave-One-Station-Out (LOSO)

To ensure that reported metrics reflect true real-world generalizability to unseen agricultural fields:
- **Strict Station-Level Grouping**: Data is partitioned strictly by physical station code. Multiple years or camera streams from the same physical farm never appear in both training and test sets.
- **Leave-One-Station-Out (LOSO)**: In an $N$-station dataset, $N$ independent folds are trained. For each fold, $N - 3$ stations form the training set, 2 stations form the validation set, and exactly 1 station serves as the completely held-out test station.
- **No Temporal or Random Leakage**: Random image-level or window-level splits are strictly prohibited, as neighboring days at the same station share identical backgrounds and weather conditions, producing artificially inflated metrics.
- **Validation-Selected Checkpoints**: All model selection and hyperparameter decisions use validation macro-F1; test stations are evaluated strictly once on the frozen selected checkpoint.

---

## References and Visual Assets

- **Pipeline Architecture Schematic**: [Raster PNG](figures/dinov3_bbch_network_pipeline.png) | [Vector PDF](figures/dinov3_bbch_network_pipeline.pdf)
- **Foundational Literature Reference**: [Automated Image Labeling for Precision Agriculture: Scaling Crop Phenology (PDF)](references/Final%20Automated%20Image%20Labeling%20for%20Precision%20Agriculture%20Scaling%20Crop%20Phenology.pdf)

