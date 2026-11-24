# Review accounts and approvals

Build on the split dataset_review app. Use SQLite for account and session storage and to serialize writes. Store proposals and outcomes in per-user action files, and log successful preview requests separately. Reviewers propose and cancel, and admins approve, reject, and reverse moves. Keep the localhost and SSH workflow.

Preflight passed with the project Pillow, SQLite, Python scrypt, and Node. Validate account lifecycle, access boundaries, queue updates, preview logging, collisions and interrupted moves, and the browser script.

Implemented operator CLI account management and authenticated viewer sessions. The admin queue supports preview, batch selection, approval, rejection, and latest-move undo. Reviewers can view, submit, and cancel their own proposals. One atomically written JSON action file and one JSONL preview log are stored per user outside the dataset. Interrupted writes and moves are reconciled using recorded file identity and path state.

Verified HTTP handlers using two reviewer accounts and an admin with scratch images, and exercised the browser-side JavaScript with a simulated DOM and API. Checks covered view-only logging, authentication, request origin and token validation, conflicting proposals, undo, interrupted moves, and session revocation.
