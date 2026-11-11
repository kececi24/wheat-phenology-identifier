# Magnification sorter

Run these commands from the repository root in the project Python environment.

```powershell
python -m dataset_cleaner.train_image_sorter --config dataset_cleaner/config.yaml
python -m dataset_cleaner.image_prediction_sorter --config dataset_cleaner/config.yaml audit --report magnification_audit.csv
python -m dataset_cleaner.image_prediction_sorter --config dataset_cleaner/config.yaml apply --report magnification_audit.csv
```

The trainer learns 1X and 10X from the trusted folders listed in the YAML. It splits by station and year and saves the lowest-validation-loss checkpoint from each of five folds. Use the full output directory from one completed run.

The audit reads images in `station/year/camera/1X` and `station/year/camera/10X`. Its CSV contains the current folder label, the filename label, each fold's prediction and confidence, the vote counts, and the proposed destination. The command prints counts by status and requires a new report filename. The default is a 3-of-5 majority. Use `audit --min-votes 4` or `audit --min-votes 5` for stricter agreement. Filename labels are recorded as separate evidence. Votes on trusted images can reflect exposure during training.

Review rows marked `MOVE` before running `apply`. Remove any of those rows you want to withhold. Applying uses the original filename in the sibling magnification folder under the same station, year, and camera. Files in trusted folders, moves into trusted folders, existing destinations, changed sources, and unreadable images are excluded or skipped. A `FILENAME_MISMATCH` calls for manual review of the name. Apply prints the number of moves and skips.
