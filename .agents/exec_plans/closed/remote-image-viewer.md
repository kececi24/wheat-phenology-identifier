# Remote image viewer

Status: complete, 2026-11-13

Build a minimal localhost HTTP image viewer under dataset_cleaner. The Python process runs beside the remote dataset. Select a subfolder, navigate images, move to station/bad_images, or move between sibling 1X/10X folders while retaining the filename. Use direct moves for interactive review. CSV remains the existing model audit workflow.

Preflight passed with Python 3.11, Pillow 12.2, and existing cleaner helpers. Use standard-library HTTP serving and existing Pillow/NumPy runtime. Bind only to 127.0.0.1 and document SSH port forwarding. Reuse the cleaner's collision and changed-source checks. Restrict browsing to the configured station or dataset root and require same-origin authenticated mutation requests. Validate navigation data, previews, moves, collisions, stale requests, and path bounds with HTTP scratch fixtures.

Implemented image_viewer.py and image_viewer.html. Direct moves use shared clean_dataset.move_image and move_bad_image helpers, with POSIX rollback if unlinking the source fails. Viewer IDs invalidate completed moves, so retries cannot move a different image. Previews decode missing EOI markers in memory. Folder selection stays inside the configured root and excludes quarantine. README documents SSH forwarding, immediate moves, selection, collisions, and preview behavior.

Validation passed through real localhost HTTP requests using scratch images, including both root scopes, preview bytes, request origin/token checks, path bounds, collisions, stale IDs and changed sources, both magnification moves, quarantine, and unreadable previews. The sandbox blocked local HTTP client connections, so HTTP and browser-server tests used approved escalation. Browser controls were exercised on generated scratch images through next/previous, subfolder selection, quarantine, 1X/10X moves, and empty list state. JavaScript syntax and Python compilation passed. No production dataset files were moved.
