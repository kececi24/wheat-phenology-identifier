# Dataset Hierarchy, Preprocessing, and Labeling

This document details the station dataset structure, camera configurations, close-up micro stream standards, BBCH milestone interval mappings, transition-day soft target formulation, and dataset verification tools for the wheat phenology identification pipeline.

---

## 1. Station Data Directory Hierarchy

The raw field image archive is organized strictly by agricultural monitoring station, harvest year, camera device, and magnification stream:

```text
data/
└── <station>/
    └── <year>/
        ├── K1/
        │   ├── 1X/
        │   │   └── <station>-<year>_<month>_<day>-<hour>_<minute>-K1-1X.jpeg
        │   └── 10X/
        │       └── <station>-<year>_<month>_<day>-<hour>_<minute>-K1-10X.jpeg
        └── K2/
            ├── 1X/
            │   └── ...
            └── 10X/
                └── ...
```

### Hierarchy Semantics
- **Station Folder (`<station>`)**: Formatted as a two-digit province and two-digit sub-station code separated by a dot (e.g., `01.01`, `02.02`, `02.03`, `06.01`, `27.02`).
- **Year Subfolder (`<year>`)**: The calendar year of the harvest or primary growing cycle. Because winter wheat is planted in autumn (e.g., October–December) and harvested the following summer (e.g., June–July), image files within a single season folder may span two calendar years.
- **Camera Subfolder (`K1`, `K2`)**: The mounted camera identifier on the field rig.
- **Magnification Subfolder (`1X`, `10X`)**:
  - `1X`: Wide-angle canopy overview (macro stream).
  - `10X`: Optical zoom close-up focusing on leaves, tillers, and spikes (micro stream).
- **Canonical Filename Pattern**:
  ```text
  <station_slug>-<YYYY>_<MM>_<DD>-<HH>_<MM>-<camera>-<zoom>.jpeg
  ```
  Example:
  ```text
  data/02.02/2015/K1/10X/02_02-2015_03_04-10_00-K1-10X.jpeg
  ```

### Quarantine Folders
Data cleaning tools automatically exclude and isolate corrupted or problematic imagery into dedicated quarantine directories within each station folder:
- `bad_images/`: Low-contrast, blurred, severely underexposed, or overexposed frames.
- `truncated_images/`: Incomplete JPEG files missing end-of-image markers.
- `corrupt_images/`: Zero-byte or undecodable image files.

Pipelines walking the filesystem strictly ignore directories matching `QUARANTINE_FOLDERS = {"bad_images", "truncated_images", "corrupt_images"}`.

---

## 2. Camera Configurations and Selection Logic

Field rigs are equipped with fixed cameras mounted on poles overlooking the trial plots.

| Camera | Role | Typical Deployment |
| :--- | :--- | :--- |
| **`K1`** | Primary fixed camera | Standard angle overlooking primary wheat plot. Default for almost all stations. |
| **`K2`** | Secondary / Alternate camera | Alternate viewing angle or backup camera. Used primarily where K1 was offline or repositioned (e.g., `01.05`, `02.06`). |
| **`K3`** | Legacy alias | Historical designation in station `02.06` where camera 2 files were indexed under K3. |

### Camera Resolution Modes
When parsing metadata in [multiscale_phenology.py](../DINOv3_BBCH/multiscale_phenology.py#L936-L990) or running [metadata_sanity_check.py](../DINOv3_BBCH/metadata_sanity_check.py#L5-L18):
1. **`--camera AUTO` (Recommended)**: Automatically checks the label table's `kamera` or `Camera` column (e.g., `1` for K1, `2` for K2). If not specified for a station-year, it selects `K1` if present, falling back gracefully to `K2`.
2. **`--camera K1`**: Restricts image discovery strictly to `K1` directories.
3. **`--camera K2`**: Restricts image discovery strictly to `K2` directories.
4. **`--camera ALL`**: Ingests all available camera perspectives without filtering.

---

## 3. The 10X Close-Up Micro Stream

While both `1X` (canopy overview) and `10X` (close-up) image streams exist in the repository, **`--stream micro` (10X only)** is the standard and leading pipeline architecture.

### Why 10X is Selected
1. **Morphological Salience**: Critical phenological transitions in wheat—such as coleoptile emergence from soil, early tiller initiation, micro-spikelet appearance during heading, and individual anther extrusion during flowering—are subtle structural features that occupy only a few pixels in a wide `1X` canopy shot.
2. **Elimination of Scale Ambiguity**: Tiling 10X images preserves high physical resolution per millimetre of plant tissue without forcing the vision backbone to reconcile two drastically different spatial zoom scales.
3. **Operational Robustness**: 10X images focus on the crop canopy rather than surrounding field soil, tractor wheel ruts, or distant horizon lines, reducing station-specific background overfitting.

---

## 4. BBCH Milestone Interval Definitions

Wheat developmental progression is annotated based on the internationally recognized **BBCH (Biologische Bundesanstalt, Bundessortenamt und CHemische Industrie)** scale.

The canonical ground truth defines eight milestone calendar dates, which delimit **seven ordered phenological intervals**, plus non-growing season periods:

| Milestone Index | Label Table Column | Biological Event | Demarcates Start Of |
| :---: | :--- | :--- | :--- |
| **1** | `1-Sowing` | Seed planted in soil | **`BBCH0`** (Germination / Emergence) |
| **2** | `2 - Emergence` | Seedlings emerge through soil surface | **`BBCH1`** (Leaf Development) |
| **3** | `3 - Tillering` | Lateral shoot / tiller formation | **`BBCH2`** (Tillering) |
| **4** | `4 - Stem Elongation` | Internode elongation / jointing | **`BBCH3`** (Stem Elongation) |
| **5** | `5 - Heading` | Inflorescence / spike emergence | **`BBCH5`** (Inflorescence Emergence) |
| **6** | `6 - Flowering` | Anthesis / floral opening | **`BBCH6_7`** (Flowering to Fruit Development) |
| **7** | `7 - Maturity` | Grain ripening and caryopsis hardening | **`BBCH8`** (Ripening / Maturity) |
| **8** | `8 - Harvest` | Crop mechanically harvested | **`OffSeason`** (Post-harvest) |

### Active Stage Classes

```text
BBCH0  ──>  BBCH1  ──>  BBCH2  ──>  BBCH3  ──>  BBCH5  ──>  BBCH6_7  ──>  BBCH8
Sowing     Emergence   Tillering    Jointing     Heading      Flowering     Maturity
```

- **`BBCH0`**: `[1-Sowing, 2 - Emergence)` — Germination to seedling emergence.
- **`BBCH1`**: `[2 - Emergence, 3 - Tillering)` — Vegetative leaf emergence and development.
- **`BBCH2`**: `[3 - Tillering, 4 - Stem Elongation)` — Tillering stage.
- **`BBCH3`**: `[4 - Stem Elongation, 5 - Heading)` — Stem elongation / shooting / boot stage.
- **`BBCH5`**: `[5 - Heading, 6 - Flowering)` — Inflorescence heading.
- **`BBCH6_7`**: `[6 - Flowering, 7 - Maturity)` — Anthesis through milk and dough development.
- **`BBCH8`**: `[7 - Maturity, 8 - Harvest)` — Ripening, desiccation, and harvest readiness.

> [!IMPORTANT]
> Class ordering is biological and strictly ordinal (`BBCH0` through `BBCH8`). Class indices must never be alphabetized; alphabetical sorting would place `BBCH0` next to `BBCH1`, but `BBCH6_7` and `BBCH8` ahead of `BBCH2`.

### Off-Season Handling
- Dates before `1-Sowing` or on/after `8 - Harvest` belong to the **`OffSeason`** class.
- In primary in-season models, training and inference use `--exclude-offseason`. OffSeason calendar dates in temporal windows provide causal context for the model but are never treated as active prediction targets.

---

## 5. ISO Label Files and Metadata Schema

The canonical ground truth files and raw observation tables are organized in the dedicated [`labels/`](../labels/README.md) directory:
- [`labels/labeling_bbch_iso_dates.csv`](../labels/labeling_bbch_iso_dates.csv) (canonical text reference)
- [`labels/labeling_bbch_iso_dates.xlsx`](../labels/labeling_bbch_iso_dates.xlsx) (Excel workbook format)
- [`labels/Fenolojik_Evreler_v3.xlsx`](../labels/Fenolojik_Evreler_v3.xlsx) (Historical raw Turkish observation table)

### Table Schema and Columns

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `ID` | Integer | Season identifier / fold grouping row | `1` |
| `Station Code` | Float/String | Two-part station code | `01.01` |
| `Year` | Integer | Primary harvest year | `2014` |
| `1-Sowing` | ISO Date | Sowing milestone date | `2014-12-05` |
| `2 - Emergence` | ISO Date | Emergence milestone date | `2014-12-16` |
| `3 - Tillering` | ISO Date | Tillering milestone date | `2015-01-08` |
| `4 - Stem Elongation` | ISO Date | Stem elongation milestone date | `2015-03-02` |
| `5 - Heading` | ISO Date | Heading milestone date | `2015-03-26` |
| `6 - Flowering` | ISO Date | Flowering milestone date | `2015-04-12` |
| `7 - Maturity` | ISO Date | Maturity milestone date | `2015-05-17` |
| `8 - Harvest` | ISO Date | Harvest milestone date | `2015-06-10` |
| `BBCH 0` .. `BBCH 8` | Integer | Audit durations in days | `11`, `23`, `53`... |
| `total` | Integer | Total days between sowing and harvest | `187` |
| `kamera` | Integer | Preferred camera index (`1` = K1, `2` = K2) | `1` |
| `bitki` | String | Crop type | `bugday` (wheat) |
| `yorumlar` | String | Expert observational notes | `fotoğraflar iyi değil`, `sıkıntı` |

> [!NOTE]
> The numeric duration columns (`BBCH 0`, `BBCH 1`, ..., `total`) serve strictly as sanity audit checks. Stage boundaries in the data loader are parsed directly from the ISO milestone date columns.

---

## 6. Transition Day Soft Targets

Biological stage transitions are continuous physical processes that unfold over several days rather than abrupt binary steps at 00:00 UTC. To prevent noisy label gradients at stage boundaries, the data pipeline implements **soft transition targets**.

### Formulation
For an image taken at calendar date $d$, boundary date $B_j$ between stages $S_{j-1}$ and $S_j$, and transition half-window $T = \text{transition\_days}$ (default $T = 2$):

$$\Delta = |d - B_j|$$

If $\Delta \le T$, the transition blending weight is:

$$\text{blend} = 0.4 \times \left(1.0 - \frac{\Delta}{\max(T, 1)}\right)$$

The soft target vector $\mathbf{y} \in \mathbb{R}^C$ is constructed as:

$$\mathbf{y}[k] = \begin{cases} 
1.0 - \text{blend}, & \text{for the interval containing date } d \\
\text{blend}, & \text{for the adjacent interval across boundary } B_j \\
0.0, & \text{for all other classes}
\end{cases}$$

```text
Class Weight
  1.0 ┤          Interval S_j-1               Interval S_j
      │         ┌───────────────┐           ┌───────────────
  0.6 ┼─────────┘               ╲           ┌───────────────
      │                          ╲ 60 / 40 ╱
  0.4 ┼───────────────────────────╲───────╱─────────────────
      │                            ╲     ╱
  0.0 ┼─────────────────────────────┴───┴───────────────────
                                      B_j (Boundary)
                    ◄─────── 2 days ───┼─── 2 days ──────►
```

### Key Properties
- **Exact Boundary Day ($\Delta = 0$)**: The interval containing the boundary date receives **0.60 (60%)** probability, and the adjacent interval receives **0.40 (40%)**.
- **No Argmax Tie**: The primary stage maintains a strict majority ($0.60 > 0.40$), eliminating non-deterministic tie-breaking during hard validation.
- **Linear Decay**: At $\Delta = 1$ day ($T = 2$), the blend is $0.20$ (80% primary, 20% adjacent). At $\Delta \ge 2$ days, $\text{blend} = 0.0$ (100% hard one-hot target).

---

## 7. Data Verification and Magnification Auditing

Human sorting errors and firmware misconfigurations occasionally cause `1X` images to be filed in `10X` folders or labeled with inverted filename tokens. To guarantee dataset fidelity without corrupting disk state, the repository provides classical feature-based audits.

### Magnification Audit Tool: `scripts/audit_image_magnification.py`
This read-only script verifies folder and filename agreements against classical vision evidence:
1. **SIFT / AKAZE Keypoint Extraction**: Identifies scale-invariant local feature descriptors in the target image.
2. **Temporal Neighbor Matching**: Matches features against trusted reference images taken within `--max-days 14.0` from both 1X and 10X streams of the same camera.
3. **RANSAC Homography & Scale Estimation**: Computes inlier count, inlier ratio, and affine scale factor:
   - Same-stream match: scale factor $\approx 1.000$ with high inliers.
   - Cross-stream match: scale factor $\approx 0.10$ or $10.0$ with low inliers.

```bash
python scripts/audit_image_magnification.py \
  --data-path data \
  --station 02.03 \
  --year 2015 \
  --output magnification_audit_02_03.csv \
  --workers 4
```

### Audit Status Classifications
- `CONSISTENT`: Folder, filename token, and visual registration agree on magnification.
- `LIKELY_MISFILED`: Strong visual and registration evidence demonstrates the image belongs in the opposite magnification folder.
- `LIKELY_FILENAME_ERROR`: Visual evidence confirms folder placement is correct, but the filename token is reversed.
- `FILENAME_FOLDER_MISMATCH`: Filename and folder disagree, but visual registration was inconclusive.
- `VISUAL_INCONCLUSIVE`: Insufficient feature matches or excessive weather difference against temporal references.
- `FILENAME_FOLDER_CONSISTENT`: Metadata-only mode confirmation (`--filename-only`).

> [!CAUTION]
> `scripts/audit_image_magnification.py` is strictly read-only. It writes a detailed CSV report with inlier counts and estimated scales and never modifies files on disk.

---

## 8. Metadata Sanity Checking Workflow

Before launching embedding precomputation or model training, run the metadata sanity verification script:

```bash
python DINOv3_BBCH/metadata_sanity_check.py \
  --label-path labels/labeling_bbch_iso_dates.csv \
  --data-path data \
  --camera AUTO \
  --stream micro
```

### Verification Checklist
1. **Class Counts**: Confirm non-zero support for all 7 in-season BBCH stages.
2. **Station-Year Coverage**: Ensure all expected station-years (e.g., 15 canonical stations) resolve both metadata and valid 10X image paths.
3. **Temporal Bounds**: Check the first and last resolved image dates against `1-Sowing` and `8 - Harvest`.
4. **Camera Fallback**: Confirm whether `--camera AUTO` correctly resolves designated camera preferences.
