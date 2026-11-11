# Unified quality cleaner

Status: complete, 2026-11-11

Provide scripts/clean_dataset.py for single images, station folders, and dataset roots. Reuse the existing exposure, contrast, monochrome, corruption, and grid blur checks. Default to inspection. Single-image mode prints GOOD or BAD. Folder apply moves bad images into station/bad_images while retaining station-relative paths and filenames. Skip quarantine directories and preserve occupied destinations. Consolidate duplicated general entry points into compatibility wrappers. Keep the model-based magnification labeling workflow separate from physical quality checks.

Preflight passed with project Python 3.11, Pillow 12.2, and NumPy 2.4.6. No external services or weights are required. Discovery found no external callers of the duplicated cleaner functions. Validate read-only byte preservation, all input scopes, station-local moves, repeated apply, collisions, and multiprocessing using scratch fixtures.

Implemented clean_dataset.py as the canonical command. Four earlier detector scripts now forward to it, with the former rigorous scanner enabling --rigorous. Station detection uses year directories or direct images, otherwise input children are treated as stations. Scans skip quarantine directories and redirected paths. Apply preserves relative paths and basenames and skips collisions or changed files. Single-file mode always prints only GOOD or BAD.

Validation passed using generated scratch fixtures through the actual CLI for image, station, dataset, multiprocessing, all compatibility entry points, monochrome, blur, rigorous glare, corrupt/truncated files, unchanged scan bytes, station-local moves, occupied destinations, repeated scans, and changed-source protection. Compilation and git diff --check passed. No production dataset cleaning was applied.

Follow-up consolidation removed the four compatibility entry points and six historical cleanup, repair, and restore helpers at the user's request. scripts/clean_dataset.py is the single image-quality cleaning command. Repository reference checks found no callers requiring migration. The earlier in-place JPEG repair and historical CSV restore helpers were retired with this workflow.

The user subsequently moved the command to dataset_cleaner/clean_dataset.py. Added opt-in --repair-jpeg there to append a missing JPEG end marker after strict in-memory candidate verification and decoding. Quality checks run after repair, and the move fingerprint is refreshed. Folder scans report repair counts, while single-image output remains GOOD/BAD. README and command help document the new location and repair flag. Scratch CLI validation passed for all three input scopes, unchanged valid files, undecodable repair candidates, repeat repair, multiprocessing, and quarantining repaired files with other defects.
