# Cleaner and review split

Separate the automated cleaner and model tools from the manual viewer, with a compatibility entry point for existing deployments. Share dataset traversal and collision-safe file moves through a standard-library module. Verify the new viewer command, its dependency boundary, and scratch moves through the reviewer and CSV apply paths.

Preflight passed with the project Python environment, Pillow, NumPy, and PyYAML. Existing viewer behavior is described in the closed remote image viewer plan.

The viewer and its HTML now live in dataset_review. The former module delegates to the new CLI. Shared traversal and move helpers live in dataset_files.py and serve automated quality cleaning, CSV application, and the viewer. Accounts and approvals are a separate follow-up.

Scratch validation exercised viewer preview, camera move and undo, CSV protection and apply, parallel cleaner apply, both viewer launch paths, and the HTTP handler's relocated HTML. The viewer imported successfully with automated cleaner and model dependencies blocked.
