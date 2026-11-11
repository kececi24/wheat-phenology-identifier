# Dataset cleaner

Run these commands from the repository root in the project Python environment.

## Image quality cleaning

`clean_dataset.py` checks empty or corrupt files, truncated JPEGs, exposure, contrast, and global and local blur. It requires Pillow and NumPy.

Check a single image. The command prints only `GOOD` or `BAD`, including when `--apply` is supplied.

```powershell
python -m dataset_cleaner.clean_dataset path/to/image.jpeg
```

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

Moves preserve filenames and the path within the station. For example, `data/01.01/2015/K1/1X/photo.jpeg` moves to `data/01.01/bad_images/2015/K1/1X/photo.jpeg`. Existing destinations and files changed during inspection are skipped. Scans exclude `bad_images`, `truncated_images`, and `corrupt_images` directories. Inspection preserves image bytes, and station folders are identified by year subfolders or images directly inside them.

| Option | Behavior |
| --- | --- |
| `--apply` | Move bad images found during a folder scan. |
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
