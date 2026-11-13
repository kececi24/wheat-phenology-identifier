# Dataset Quality Cleaning and Repair

This directory contains automated image quality inspection, in-place JPEG end-marker repair, corrupted/blurred frame isolation, and 5-fold ensemble 1X/10X magnification classification utilities.

## Documentation

Full operational workflows, command-line arguments, defect detection math, and threshold tables are documented in the [`docs/`](../docs/README.md) hub:

- **Quality Cleaning and Repair Workflow**: [Tools and Workflows Guide - Section 1](../docs/tools-and-workflows.md#1-dataset-quality-cleaning-and-repair-dataset_cleaner)
  - Single-image, station-level, and root dataset inspection commands.
  - In-place stream repair appending missing `0xFFD9` EOI markers (`--repair-jpeg`).
  - Spatial side-patch Laplacian blur calculation (excluding central pole columns).
  - 5-fold vision ensemble magnification audit and safe folder relocation.
- **Dataset Structure & Integrity Rules**: [Dataset and Preprocessing Guide](../docs/dataset-and-preprocessing.md)
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `clean_dataset.py` | Standalone script for corrupted JPEG repair, exposure checks, and side-patch blur detection | [Tools Guide](../docs/tools-and-workflows.md#image-quality-cleaning-clean_datasetpy) |
| `train_image_sorter.py` | Trains 5-fold stratified vision ensemble on verified trusted folders for 1X vs 10X separation | [Tools Guide](../docs/tools-and-workflows.md#magnification-sorting-ensemble) |
| `image_prediction_sorter.py` | Audits candidate folders and safely moves verified misfiled images to sibling directories | [Tools Guide](../docs/tools-and-workflows.md#magnification-sorting-ensemble) |
| `config.yaml` | Configuration for trusted folders, ensemble architectures, and voting thresholds | [Tools Guide](../docs/tools-and-workflows.md#magnification-sorting-ensemble) |
| `image_viewer.py` | Localhost viewer for manual image review and immediate folder moves | Instructions below |

## Manual image review on a remote machine

Run the viewer on the machine holding the dataset, from the repository root in the project Python environment. The input can be a station folder or a dataset root containing station folders.

```bash
python -m dataset_cleaner.image_viewer /path/to/data --port 8765
```

The server binds to `127.0.0.1`. To access it from your computer, run this command on your computer and keep the SSH session open. Replace `user@remote-host` with your SSH destination.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

Open `http://127.0.0.1:8765` in your local browser. If local port 8765 is occupied, use `-L 8766:127.0.0.1:8765` and open local port 8766. VS Code Remote SSH users can also forward remote port 8765 through the Ports panel.

The page displays one image and its relative path, which includes the filename. Use Previous and Next or the left and right arrow keys to browse. Arrow shortcuts are inactive while editing the Folder field or while a request is running. The Folder field selects a folder within the configured root, such as `01.01/2015/K1/1X`, and Load folder refreshes the selection. Use `.` for the whole configured root. Quarantine directories are excluded.

The move buttons execute immediately on the machine running Python. Send to bad_images preserves the path inside the station, for example moving `01.01/2015/K1/1X/photo.jpeg` to `01.01/bad_images/2015/K1/1X/photo.jpeg`. Move to 1X and Move to 10X move between sibling magnification folders under the same camera. Every move preserves the filename, and occupied destinations or changed source files produce an error. A completed move advances to the next image in the review list. Reload the folder to include relocated magnification images again.

Previews are resized in memory for browser display. A missing JPEG end marker can be supplied in memory for preview decoding. Source image bytes are preserved during viewing. Files that cannot be previewed can still be quarantined using the button. Stop the server with Ctrl+C.

Direct moves suit interactive review because each decision takes effect immediately. The magnification model's CSV audit and apply commands remain available for reviewing a batch of proposed changes before execution.

## Automated image quality cleaning

Scan a station folder or a dataset root containing station folders. Folder scans print the detected bad images with their reasons and a summary.

```powershell
python -m dataset_cleaner.clean_dataset data/01.01
python -m dataset_cleaner.clean_dataset data
```

Add `--apply` to move detected bad images into each station's `bad_images` folder.

```powershell
python -m dataset_cleaner.clean_dataset data/01.01 --apply
python -m dataset_cleaner.clean_dataset data --apply
```

Moves preserve filenames and the path within the station. For example, `data/01.01/2015/K1/1X/photo.jpeg` moves to `data/01.01/bad_images/2015/K1/1X/photo.jpeg`. Existing destinations and files changed during inspection are skipped. Scans exclude `bad_images`, `truncated_images`, and `corrupt_images` directories. Station folders are identified by year subfolders or images directly inside them.

Use `--repair-jpeg` to repair missing JPEG end markers in place before quality checks. The command validates a candidate with the appended marker using strict Pillow decoding, then appends the two marker bytes to the original file. Repair works for single images and folders and can be used independently of `--apply`. Folder output reports repaired files and their count. Single-image output remains `GOOD` or `BAD` after repair. The quality checks still run after repair, so an image with another problem can be moved by `--apply`. An end marker repair cannot reconstruct missing image content.

```powershell
python -m dataset_cleaner.clean_dataset path/to/image.jpeg --repair-jpeg
python -m dataset_cleaner.clean_dataset data --repair-jpeg
python -m dataset_cleaner.clean_dataset data --repair-jpeg --apply
```

| Option | Behavior |
| --- | --- |
| `--apply` | Move bad images found during a folder scan. |
| `--repair-jpeg` | Repair missing JPEG end markers in place before checking image quality. |
| `--mode conservative\|medium\|aggressive` | Select blur sensitivity. The default is `medium`. |
| `--blur-threshold FLOAT` | Override the preset grid blur threshold. |
| `--no-local-blur` | Disable local blur detection, which is enabled by default. |
| `--monochrome` | Also flag grayscale or monochrome camera images. |
| `--rigorous` | Enable stricter glare, fog, exposure, contrast, and blur checks. |
| `--workers N` | Set parallel worker processes. The default is up to four. |

```powershell
python -m dataset_cleaner.clean_dataset data --mode conservative --workers 4
python -m dataset_cleaner.clean_dataset data/01.01 --rigorous --monochrome
```

## Magnification sorter

```powershell
python -m dataset_cleaner.train_image_sorter --config dataset_cleaner/config.yaml
python -m dataset_cleaner.image_prediction_sorter --config dataset_cleaner/config.yaml audit --report magnification_audit.csv
python -m dataset_cleaner.image_prediction_sorter --config dataset_cleaner/config.yaml apply --report magnification_audit.csv
```

The trainer learns 1X and 10X from the trusted folders listed in the YAML. It splits by station and year and saves the lowest-validation-loss checkpoint from each of five folds. Use the full output directory from one completed run.

The audit reads images in `station/year/camera/1X` and `station/year/camera/10X`. Its CSV contains the current folder label, the filename label, each fold's prediction and confidence, the vote counts, and the proposed destination. The command prints counts by status and requires a new report filename. The default is a 3-of-5 majority. Use `audit --min-votes 4` or `audit --min-votes 5` for stricter agreement. Filename labels are recorded as separate evidence. Votes on trusted images can reflect exposure during training.

Review rows marked `MOVE` before running `apply`. Remove any of those rows you want to withhold. Applying uses the original filename in the sibling magnification folder under the same station, year, and camera. Files in trusted folders, moves into trusted folders, existing destinations, changed sources, and unreadable images are excluded or skipped. A `FILENAME_MISMATCH` calls for manual review of the name. Apply prints the number of moves and skips.

After moving images, synchronize their final filename magnification token with their current `1X` or `10X` folder using Bash (Linux, WSL, or Git Bash). The argument can be a dataset root, station folder, or magnification folder. The script renames mismatches recursively and skips quarantine folders.

```bash
bash dataset_cleaner/rename_magnification.sh data/01.01
```

For example, `10X/photo-1X.jpeg` becomes `10X/photo-10X.jpeg`. An occupied destination produces `photo-10X(2).jpeg`, then `(3)` and higher as needed. The script processes every matching filename in the supplied folder tree.
