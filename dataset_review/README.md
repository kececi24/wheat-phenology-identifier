# Dataset review app

Run these commands on the machine holding the dataset, from the repository root, with Python and Pillow installed. The input is a station folder or dataset root. Create the admin and individual reviewer accounts before starting the server. Passwords are entered at a prompt.

```bash
python -m dataset_review.image_viewer /path/to/data --create-admin admin
python -m dataset_review.image_viewer /path/to/data --create-reviewer reviewer1
python -m dataset_review.image_viewer /path/to/data --port 8765
```

The default state directory is a hidden sibling named after the dataset, such as `/path/to/.data-review`. Use the same `--state-dir /private/review-state` for account management and server startup if you specify one. The state directory belongs to one dataset root. State stores account and session records in `accounts.sqlite3`. The operator can use `--reset-password USERNAME` or `--disable-user USERNAME`. Both revoke existing sessions. Use one server per dataset and keep its state directory accessible only to the server operator.

Existing launch commands using `python -m dataset_cleaner.image_viewer` also work.

The server binds to `127.0.0.1`. To access it from your computer, run this command on your computer and keep the SSH session open. Replace `user@remote-host` with your SSH destination.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

Open `http://127.0.0.1:8765` in your local browser. If local port 8765 is occupied, use `-L 8766:127.0.0.1:8765` and open local port 8766. VS Code Remote SSH users can also forward remote port 8765 through the Ports panel.

Choose 1, 4, or 8 images per page. Select an image with its radio button or by clicking its tile, then use the move buttons on that image. Previous and Next and the left and right arrow shortcuts browse one page at a time. Inside form controls, arrow keys use the browser's normal behavior. The Folder field selects a folder within the configured root, such as `01.01/2015/K1/1X`. Use `.` for the whole configured root. Quarantine directories are excluded.

Reviewers propose moves to `bad_images`, `1X`, `10X`, `K1`, or `K2`. Each pending tile shows the proposed destination. Reviewers can cancel a proposal from the gallery or **My pending moves**. **Undo last proposal** cancels their newest pending proposal. Outcomes are visible in **My activity**.

Admins use **Approval queue** to preview proposals and apply or reject selected rows. Admin gallery moves take effect immediately. All moves preserve filenames and either change the requested folder or retain the station-relative path inside `bad_images`. Apply checks the source identity and destination before moving. **Undo last move** restores the most recent applied move when the original path is free and the moved file still matches.

The server stores each user's proposals and outcomes in `actions/<user-id>.json`. A separate `views/<user-id>.jsonl` records the username, UTC timestamp, and relative path for each successful reviewer preview request. The log includes previews with no proposal and repeat requests. In grid mode, the browser may request nearby previews before they scroll into view.

For approval to govern dataset changes, give write access to the server operator and keep reviewer SSH access limited to forwarding or otherwise read-only for the dataset. Each reviewer signs in with their own viewer account.

Previews are resized in memory for browser display. A missing JPEG end marker can be supplied in memory for preview decoding. Source image bytes are preserved during viewing. Users can still propose quarantine for files that cannot be previewed. Stop the server with Ctrl+C.

The automated magnification model retains its CSV audit and apply workflow.
