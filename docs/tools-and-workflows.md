# Practical Tooling, Verification, and Deployment Workflows

This document details practical operational utilities across the project lifecycle: image quality cleaning, classical magnification audits, station landmark verification scripts, distributed PyTorch FSDP2 execution, and single-image and temporal cached-window inference workflows.

---

## 1. Dataset Quality Cleaning and Repair (`dataset_cleaner/`)

The [dataset_cleaner](../dataset_cleaner/) suite detects damaged, truncated, or uninformative frames, repairs incomplete JPEG streams in place, and moves bad images to a quarantine directory.

### Image Quality Cleaning: `clean_dataset.py`

[clean_dataset.py](../dataset_cleaner/clean_dataset.py) scans an individual image, a station folder, or the entire `data` root.

```bash
# Check a single image (outputs GOOD or BAD)
python -m dataset_cleaner.clean_dataset path/to/image.jpeg

# Scan a station folder and print defect diagnostics
python -m dataset_cleaner.clean_dataset data/01.01

# In-place repair of truncated JPEGs and move confirmed bad images to quarantine
python -m dataset_cleaner.clean_dataset data --repair-jpeg --apply --workers 4
```

#### Quality Checks and Detection Thresholds

| Defect Type | Status Code | Detection Logic / Mathematical Condition |
| :--- | :--- | :--- |
| **Empty File** | `CORRUPT` | File byte size is 0 bytes or unreadable stream. |
| **Decoding Error** | `CORRUPT` | PIL throws `OSError`, `SyntaxError`, or `DecompressionBombError`. |
| **Truncated Stream** | `TRUNCATED` | Final two bytes of file stream are not `0xFFD9` (JPEG EOI marker). |
| **Dark Frame** | `DARK` | Grayscale pixel mean $\mu < 5.0$, or contrast $\sigma < 1.0$ with $\mu < 10.0$. |
| **White / Washed Out**| `WHITE` | Grayscale pixel mean $\mu > 250.0$, or contrast $\sigma < 1.0$ with $\mu > 245.0$. |
| **Low Contrast** | `LOW_CONTRAST` | Standard deviation $\sigma < 5.0$. |
| **Global Blur** | `BLURRED` | Mean side-patch Laplacian variance $\text{Var}(\nabla^2 I) < \text{threshold}$. |
| **Local Blur** | `LOCAL_BLUR` | 25th percentile side patch $< \max(10.0, 0.45 \times \text{threshold})$ or min patch $< 0.5 \times \text{threshold}$. |
| **Monochrome** | `MONOCHROME` | Mean RGB difference $< 3.0$ (when `--monochrome` enabled). |

> [!NOTE]
> **Spatial Side-Patch Blur Calculation**: The image is split into a $4 \times 4$ grid ($64 \times 64$ patches). Blur detection evaluates only columns 0 and 3 (the 8 side patches), explicitly **excluding columns 1 and 2**. This prevents the sharp vertical mounting pole or mast from masking blurred background crops.

#### CLI Arguments Reference

| Option | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--apply` | Flag | `False` | Move detected defective images into `<station>/bad_images/`. |
| `--repair-jpeg` | Flag | `False` | Verify and append missing `0xFFD9` marker in place. |
| `--mode` | `conservative`, `medium`, `aggressive` | `medium` | Blur sensitivity preset (Laplacian thresholds: 15.0, 45.0, 100.0). |
| `--blur-threshold` | Float | Preset value | Manually override the Laplacian blur threshold. |
| `--no-local-blur` | Flag | `False` | Disable local patch blur detection. |
| `--monochrome` | Flag | `False` | Quarantine black-and-white camera frames. |
| `--rigorous` | Flag | `False` | Enable extra fog, glare ($>30\%$ pixels $>250$), and overexposure checks. |
| `--workers` | Integer | $\le 4$ | Number of concurrent worker processes. |

#### In-Place JPEG Stream Repair (`--repair-jpeg`)
Many sensor cameras lose power before writing the JPEG End of Image (`EOI`) marker. When `--repair-jpeg` is provided:
1. It validates that the image begins with `0xFFD8` (SOI) and lacks `0xFFD9` (EOI).
2. It attempts full PIL decoding on a memory buffer containing the image bytes plus `0xFFD9`.
3. If decode and load succeed, it validates that the file has not been modified on disk, then appends `b"\xff\xd9"` to the file in place.

---

### Magnification Sorting Ensemble

For automated reclassification of misfiled images across `1X` and `10X`, the package includes a 5-fold vision ensemble ([train_image_sorter.py](../dataset_cleaner/train_image_sorter.py) and [image_prediction_sorter.py](../dataset_cleaner/image_prediction_sorter.py)).

1. **Train 5-Fold Ensemble**:
   ```bash
   python -m dataset_cleaner.train_image_sorter --config dataset_cleaner/config.yaml
   ```
   Trains an image classifier (configured in [config.yaml](../dataset_cleaner/config.yaml), e.g. `facebook/deit-tiny-patch16-224`) across 5 station/year-stratified folds on verified trusted folders.
2. **Audit Target Directories**:
   ```bash
   python -m dataset_cleaner.image_prediction_sorter --config dataset_cleaner/config.yaml audit --report magnification_audit.csv --min-votes 4
   ```
   Writes fold votes, ensemble confidence, and status (`MATCH`, `MOVE`, `FILENAME_MISMATCH`, `LOW_CONSENSUS`).
3. **Execute Approved Moves**:
   ```bash
   python -m dataset_cleaner.image_prediction_sorter --config dataset_cleaner/config.yaml apply --report magnification_audit.csv
   ```
   Safely transfers rows flagged as `MOVE` to the corresponding sibling magnification folder under the same station, year, and camera.

---

## 2. Classical 1X/10X Magnification Auditing

[scripts/audit_image_magnification.py](../scripts/audit_image_magnification.py) provides a strictly **read-only** audit using classical computer vision. It compares every image against nearby trusted temporal references from both streams.

```bash
# Run SIFT-based magnification audit for station 02.03 in harvest year 2015
python scripts/audit_image_magnification.py \
  --data-path data \
  --station 02.03 \
  --year 2015 \
  --output magnification_audit_02_03.csv \
  --workers 4

# Fast metadata-only check (no OpenCV required)
python scripts/audit_image_magnification.py \
  --data-path data \
  --output filename_audit.csv \
  --filename-only

# Filter CSV output to actionable discrepancies only
python scripts/audit_image_magnification.py \
  --data-path data \
  --output discrepancies.csv \
  --only-discrepancies
```

### Visual Registration Mechanics
- **Keypoint Detector**: SIFT (default) or AKAZE descriptors extracted on images resized to `--max-side 960`.
- **Temporal References**: Selects trusted reference images (where folder and filename tokens agree) taken within `--max-days 14.0` days from both `1X` and `10X`.
- **RANSAC Homography Estimation**: Evaluates feature matches using Lowe's ratio test ($<0.75$) and RANSAC homography.
- **Scale Estimation**: Computes the geometric scale factor $s = \sqrt{|\det(H)|}$:
  - Same stream: $s \approx 1.0$ with high inlier ratio.
  - Cross stream: $s \approx 0.1$ or $10.0$ with low inliers.

### Status Classifications

| Status | Definition | Recommended Action |
| :--- | :--- | :--- |
| `CONSISTENT` | Folder, filename token, and visual registration agree. | None (valid image). |
| `LIKELY_MISFILED` | Visual registration strongly indicates opposite stream folder. | Move image to opposite magnification folder. |
| `LIKELY_FILENAME_ERROR` | Visual evidence agrees with folder, but filename token is inverted. | Update filename or retain in folder. |
| `FILENAME_FOLDER_MISMATCH` | Folder and filename disagree, but visual registration was inconclusive. | Manual human inspection required. |
| `VISUAL_INCONCLUSIVE` | Insufficient inliers or reference frames to establish certainty. | Inspect visually if critical. |
| `FILENAME_FOLDER_CONSISTENT`| Folder and filename agree in `--filename-only` mode. | Baseline metadata consistency. |

---

## 3. Station and Camera Verification Scripts

Field cameras undergo seasonal maintenance, battery outages, and perspective drift. Several utility scripts in `scripts/` audit temporal trajectories and physical camera integrity.

### 1. Excess Green Landmark Verification: `verify_all_stations.py`
[scripts/verify_all_stations.py](../scripts/verify_all_stations.py) parses [labeling_bbch_iso_dates.csv](../labeling_bbch_iso_dates.csv), discovers 10X images in `[1-Sowing, 8 - Harvest]`, and calculates the **Excess Green Index (ExG)** in parallel across the lower half of each image (region of interest for wheat canopy):

$$\text{ExG} = 2g - r - b, \quad \text{where } r = \frac{R}{R+G+B}, \quad g = \frac{G}{R+G+B}, \quad b = \frac{B}{R+G+B}$$

It extracts biological landmarks from the resulting seasonal curve:
- **Detected Emergence**: First date where $\text{ExG} > \mu_{\text{baseline}} + 2\sigma_{\text{baseline}}$.
- **Peak Greenness**: Maximum ExG date (physiologically corresponds to heading / flowering).
- **Detected Harvest**: Post-peak drop date where ExG reaches senescence floor.
- **Plots**: Saves seasonal curve figures superimposing human milestone annotations over the ExG curve.

### 2. Station Detail Analysis: `analyze_details.py`
[scripts/analyze_details.py](../scripts/analyze_details.py) parses the summary CSV produced by station verification:
- Flags flat or uninformative seasons: `Peak ExG < 0.20` indicates bare soil, extreme drought, or camera malfunction.
- Compares labeled emergence, heading, and harvest against detected greenness inflection points to flag label shift anomalies.

### 3. Camera Shift Detection: `detect_shifts.py`
[scripts/detect_shifts.py](../scripts/detect_shifts.py) detects physical displacement or wind drift of the camera pole:
- Computes mean RGB color channels on downsampled frames across consecutive dates.
- Flags abrupt total color differences $\Delta R + \Delta G + \Delta B > 30.0$, isolating exact timestamps where cameras were rotated, serviced, or shifted.

### 4. Verification Summary: `summarize_all_verifications.py`
[scripts/summarize_all_verifications.py](../scripts/summarize_all_verifications.py) compiles station verification metrics into `all_stations_verification_summary.csv`, recording image counts, peak ExG values, curve quality, and landmark offset errors.

### 5. Station Batch Utilities
- `scripts/Rename-Camera.ps1`: Interactive PowerShell tool to preview and rename camera tokens (`K1_TO_K2` or `K2_TO_K1`).
- `scripts/Rename-Magnification.ps1`: Interactive tool to preview and rename zoom tokens (`10X_TO_1X` or `1X_TO_10X`).
- `scripts/check_folders.py`: Quick verification that 10X directories exist for each season in the label table.

---

## 4. Distributed FSDP2 Execution Setup

Large-scale model training is accelerated across multi-GPU nodes using PyTorch Fully Sharded Data Parallel 2 (FSDP2) in [DINOv3_BBCH/distributed_runtime.py](../DINOv3_BBCH/distributed_runtime.py).

```text
       Process Group (NCCL)
     ┌───────────────────────┐
     │ 1D Device Mesh (CUDA) │
     └───────────┬───────────┘
                 │
   ┌─────────────┴─────────────┐
   ▼                           ▼
GPU Rank 0                 GPU Rank 1
┌─────────────────────┐    ┌─────────────────────┐
│ Temporal Transformer│    │ Temporal Transformer│
│ Block 0 Shard       │    │ Block 0 Shard       │
│ Block 1 Shard       │    │ Block 1 Shard       │
│ Classifier Shard    │    │ Classifier Shard    │
└─────────────────────┘    └─────────────────────┘
```

### Architecture and Components

#### 1. Distributed Initialization
`initialize_distributed()` checks the `LOCAL_RANK` environment variable:
- Assigns the local GPU device via `torch.cuda.set_device(local_rank)`.
- Initializes the process group using the `nccl` backend.
- Returns `(device, rank, world_size)`. Gracefully falls back to single-device CPU/CUDA when `LOCAL_RANK` is absent.

#### 2. Distributed Training Sampler
`make_train_sampler(dataset, rank, world_size, seed)` builds a `DistributedSampler`:
- Ensures non-overlapping batches across ranks (`shuffle=True`, `drop_last=True`).
- Sets per-epoch seed for reproducibility.

#### 3. Sharding Temporal Transformer with FSDP2
`shard_temporal_model(model, world_size)` implements PyTorch 2.x FSDP2:
```python
mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("shard",))

# Disable fastpath MHA and nested tensors for sharded execution
torch.backends.mha.set_fastpath_enabled(False)
model.temporal.enable_nested_tensor = False
model.temporal.use_nested_tensor = False

# Fully shard individual transformer encoder layers
for block in model.temporal.layers:
    fully_shard(block, mesh=mesh, reshard_after_forward=True)

# Fully shard top-level model container
fully_shard(model, mesh=mesh, reshard_after_forward=True)
```

#### 4. Reading Sharded DTensor Gate Values
Learned metadata gates (calendar, weather, location) are sharded as `DTensor` instances in FSDP2. `read_gate_value(gate_logit)` unwraps the tensor across ranks via `gate_logit.detach().full_tensor()`, applies the sigmoid function, and returns a scalar float for logging and TensorBoard.

---

## 5. Inference Workflows

The repository provides two distinct inference pipelines: single-image diagnostic inference and realistic cached-window causal deployment.

### 1. Single-Image Diagnostic Inference: `infer_single_image.py`

[DINOv3_BBCH/infer_single_image.py](../DINOv3_BBCH/infer_single_image.py) runs inference on a single static frame with a full-image checkpoint.

```bash
python DINOv3_BBCH/infer_single_image.py \
  --checkpoint path/to/best_model.pt \
  --image-path data/02.02/2015/K1/10X/02_02-2015_03_04-10_00-K1-10X.jpeg \
  --planting-date 2014-12-02 \
  --station-code 02.02 \
  --device cuda
```

- **Mechanism**: Replicates the single target image across all $W$ days of the temporal window (`make_repeated_window`).
- **Use Case**: Diagnostic sanity check and smoke testing. It verifies checkpoint loading and feature fusion but **does not represent real deployment**, because static image repetition provides no real temporal motion or history.

---

### 2. Realistic Cached-Window Causal Deployment: `infer_cached_window.py`

[DINOv3_BBCH/infer_cached_window.py](../DINOv3_BBCH/infer_cached_window.py) executes causal operational inference matching the real-world agricultural monitoring setup.

```bash
python DINOv3_BBCH/infer_cached_window.py \
  --checkpoint results_dinov3_bbch_21_LOSO/temporal/run_fold_1/fold_1/best_model.pt \
  --image-path data/02.02/2015/K1/10X/02_02-2015_05_01-10_00-K1-10X.jpeg \
  --image-dir data/02.02/2015/K1/10X \
  --planting-date 2014-12-02 \
  --station-code 02.02 \
  --weather-cache results_dinov2_bbch_cache/meteostat_weather_cache.csv \
  --top-k 5 \
  --device cuda
```

#### Step-by-Step Execution Lifecycle

```text
Target Image & Date
       │
       ▼
1. Discover Historical Images  ───> Scan --image-dir for dates in [target_date - W + 1, target_date]
       │
       ▼
2. On-the-Fly Image Encoding    ───> Crop 224x224 tiles, extract DINOv3 2x2 dense features + CLS
       │
       ▼
3. Build Causal Window          ───> Assemble (W, num_tiles, 5, 768) tensor; mask missing dates
       │
       ▼
4. Fuse Auxiliary Metadata      ───> Compute days-since-sowing, cumulative GDD, and lat/lon/elev
       │
       ▼
5. Hierarchical Temporal Model  ───> Patch pooling -> Tile pooling -> Gated MLPs -> Transformer
       │
       ▼
Top-K Phenology Probabilities   ───> Print predicted BBCH stage and class likelihoods
```

#### Key Deployment Parameters

| Parameter | Purpose | Behavior |
| :--- | :--- | :--- |
| `--image-path` | Target day image | Resolves observation date and provides the prediction anchor. |
| `--image-dir` | Historical folder | Discovers preceding images in the causal temporal window ($W$ days). |
| `--repeat-missing` | Handling gaps | By default, missing intermediate dates are masked (`mask=False`). Supplying this repeats the target image. |
| `--planting-date` | Calendar anchor | Calculates days-since-sowing for the calendar gated MLP. |
| `--weather-cache` | Weather data | Reads Meteostat observations and computes normalized cumulative GDD. |
| `--station-code` | Coordinates | Looks up latitude, longitude, and elevation for the location gated MLP. |
| `--debug-window` | Diagnostics | Prints the resolved date sequence and image paths before encoding. |

---

### 3. Single-Day Linear Inference: `DINOv3_BBCH_Linear/infer_single_image.py`

For non-temporal ablation checkpoints trained with `DINOv3_BBCH_Linear`:

```bash
python DINOv3_BBCH_Linear/infer_single_image.py \
  --checkpoint results_dinov3_bbch_linear/fold_1/best_model.pt \
  --image-path data/02.02/2015/K1/10X/example.jpeg \
  --target-date 2015-03-04 \
  --planting-date 2014-10-20 \
  --station-code 02.02 \
  --weather-cache results_dinov3_bbch_linear/meteostat_weather_cache.csv \
  --device cuda
```

Linear checkpoints self-contain the full model definition, DINOv3 architecture config, preprocessing, tiling, metadata normalization, and class order.

---

### 4. Operational Summary of Inference Entry Points

The repository provides three dedicated command-line inference tools:

| Tool | Pipeline Type | Primary Input | Use Case |
| :--- | :--- | :--- | :--- |
| `DINOv3_BBCH/infer_cached_window.py` | Causal Temporal DINOv3 | Historical image folder + target image | Production-grade causal window deployment with masked sensor gaps |
| `DINOv3_BBCH/infer_single_image.py` | Repeated-Frame Temporal DINOv3 | Single image file | Diagnostic smoke test verifying weights, tiling, and metadata fusion |
| `DINOv3_BBCH_Linear/infer_single_image.py` | Single-Day Linear Probe | Single image file | Fast non-temporal ablation inference from standalone linear checkpoints |

> **Note on Graphical Interface**: A graphical desktop interface (`demo_ui`) was evaluated during early prototyping and has regression interface coverage in `tests/test_demo_ui.py`. Operational deployment is conducted using the dedicated command-line entry points above.


