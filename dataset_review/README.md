# Dataset review app

Run the viewer on the machine holding the dataset, from the repository root with Python and Pillow installed. The input can be a station folder or a dataset root containing station folders.

```bash
python -m dataset_review.image_viewer /path/to/data --port 8765
```

Existing launch commands using `python -m dataset_cleaner.image_viewer` also work.

The server binds to `127.0.0.1`. To access it from your computer, run this command on your computer and keep the SSH session open. Replace `user@remote-host` with your SSH destination.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

Open `http://127.0.0.1:8765` in your local browser. If local port 8765 is occupied, use `-L 8766:127.0.0.1:8765` and open local port 8766. VS Code Remote SSH users can also forward remote port 8765 through the Ports panel.

Choose 1, 4, or 8 images per page. Select an image with its radio button or by clicking its tile, then use the move buttons on that image. Previous and Next and the left and right arrow shortcuts browse one page at a time. Inside form controls, arrow keys use the browser's normal behavior. The Folder field selects a folder within the configured root, such as `01.01/2015/K1/1X`. Use `.` for the whole configured root. Quarantine directories are excluded.

The move buttons execute immediately on the machine running Python. Send to bad_images preserves the path inside the station, for example moving `01.01/2015/K1/1X/photo.jpeg` to `01.01/bad_images/2015/K1/1X/photo.jpeg`. The 1X and 10X buttons change magnification under the same camera. The K1 and K2 buttons change cameras within the same station and year while preserving magnification. Every move preserves the filename. The viewer reports occupied destinations and changed source files.

Undo last move restores the original path for the most recent successful move, including quarantine. Undo is available in the tab that made the move until the page is refreshed, the server stops, or a newer move succeeds. An occupied original path or a changed moved file prevents restoration.

Previews are resized in memory for browser display. A missing JPEG end marker can be supplied in memory for preview decoding. Source image bytes are preserved during viewing. Files that cannot be previewed can still be quarantined using the button. Stop the server with Ctrl+C.

Direct moves suit interactive review because each decision takes effect immediately. The magnification model's CSV audit and apply commands remain available for reviewing a batch of proposed changes before execution.
