"""Review a remote dataset through a localhost browser and move images in place."""

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import secrets
from threading import RLock
from urllib.parse import parse_qs, urlsplit
import warnings

from PIL import Image, ImageOps

from .clean_dataset import (
    QUARANTINE_FOLDERS,
    images_in_station,
    move_bad_image,
    move_image,
    station_folders,
)


@dataclass
class ReviewImage:
    path: Path
    station: Path
    fingerprint: tuple


class Viewer:
    def __init__(self, root):
        self.root = root.resolve()
        self.token = secrets.token_urlsafe(32)
        self.images = {}
        self.paths = {}
        self.lock = RLock()

    def selected_folder(self, name):
        folder = (self.root / name).absolute()
        if folder.resolve() != folder:
            raise ValueError("Choose a folder without redirected paths")
        folder.relative_to(self.root)
        if not folder.is_dir():
            raise ValueError("Folder does not exist")
        if any(part.lower() in QUARANTINE_FOLDERS for part in folder.relative_to(self.root).parts):
            raise ValueError("Quarantine folders are excluded from review")
        return folder

    def scan(self, name):
        folder = self.selected_folder(name)
        result = []
        with self.lock:
            for station in station_folders(self.root):
                if not (station.is_relative_to(folder) or folder.is_relative_to(station)):
                    continue
                selected = folder if folder.is_relative_to(station) else station
                for path in images_in_station(selected):
                    stat = path.stat()
                    fingerprint = stat.st_size, stat.st_mtime_ns
                    identifier = self.paths.get(path)
                    if identifier and self.images[identifier].fingerprint != fingerprint:
                        del self.images[identifier]
                        identifier = None
                    if identifier is None:
                        identifier = secrets.token_urlsafe(18)
                        self.paths[path] = identifier
                        self.images[identifier] = ReviewImage(path, station, fingerprint)
                    result.append({
                        "id": identifier, "name": path.name,
                        "path": path.relative_to(self.root).as_posix(),
                        "magnification": path.parent.name if path.parent.name in {"1X", "10X"} else "",
                    })
        return result

    def current(self, identifier):
        image = self.images.get(identifier)
        if image is None:
            raise ValueError("Image was moved or is no longer available. Reload the folder.")
        if image.path.resolve() != image.path:
            raise ValueError("Image path is redirected")
        stat = image.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != image.fingerprint:
            raise ValueError("Image changed since the folder was loaded. Reload the folder.")
        return image

    def preview(self, identifier):
        with self.lock:
            image = self.current(identifier)
            data = image.path.read_bytes()
            # Missing EOI markers can be supplied in memory for viewing.
            if data.startswith(b"\xff\xd8") and not data.endswith(b"\xff\xd9"):
                data += b"\xff\xd9"
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    frame = ImageOps.exif_transpose(source).convert("RGB")
                    frame.thumbnail((1600, 1000))
                    output = BytesIO()
                    frame.save(output, format="JPEG", quality=90)
            return output.getvalue()

    def move(self, identifier, action):
        with self.lock:
            image = self.current(identifier)
            if action == "bad":
                destination = move_bad_image(image.path, image.station, image.fingerprint)
            elif action in {"1X", "10X"}:
                if image.path.parent.name not in {"1X", "10X"}:
                    raise ValueError("Magnification moves require an image in a 1X or 10X folder")
                if image.path.parent.name == action:
                    raise ValueError(f"Image is already in {action}")
                destination = image.path.parent.with_name(action) / image.path.name
                destination.relative_to(self.root)
                move_image(image.path, destination, image.fingerprint)
            else:
                raise ValueError("Unknown move action")
            del self.images[identifier]
            del self.paths[image.path]
            return destination.relative_to(self.root).as_posix()


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal output focused on startup and completed moves.
        pass

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'")
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, data):
        self.send_body(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def local_request(self):
        try:
            hostname = urlsplit("http://" + self.headers.get("Host", "")).hostname
        except ValueError:
            return False
        return hostname in {"127.0.0.1", "localhost"}

    def do_GET(self):
        if not self.local_request():
            self.send_json(403, {"error": "Use localhost or 127.0.0.1"})
            return
        route = urlsplit(self.path)
        query = parse_qs(route.query)
        try:
            if route.path == "/":
                html = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
                html = html.replace("__VIEWER_TOKEN__", self.server.viewer.token)
                self.send_body(200, html.encode(), "text/html; charset=utf-8")
            elif route.path == "/api/images":
                images = self.server.viewer.scan(query.get("folder", ["."])[0])
                self.send_json(200, {"images": images, "root": str(self.server.viewer.root)})
            elif route.path == "/image":
                preview = self.server.viewer.preview(query.get("id", [""])[0])
                self.send_body(200, preview, "image/jpeg")
            else:
                self.send_json(404, {"error": "Unknown endpoint"})
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as error:
            self.send_json(409, {"error": str(error)})

    def do_POST(self):
        origin = self.headers.get("Origin")
        expected_origin = "http://" + self.headers.get("Host", "")
        if (
            not self.local_request()
            or self.headers.get("X-Viewer-Token") != self.server.viewer.token
            or (origin is not None and origin != expected_origin)
        ):
            self.send_json(403, {"error": "Invalid viewer request"})
            return
        if self.path != "/api/move":
            self.send_json(404, {"error": "Unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            identifier = request.get("id")
            action = request.get("action")
            if not isinstance(identifier, str) or not isinstance(action, str):
                raise ValueError("Expected image id and move action")
            destination = self.server.viewer.move(identifier, action)
            print(f"Moved image to {destination}", flush=True)
            self.send_json(200, {"destination": destination})
        except (OSError, ValueError) as error:
            self.send_json(409, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Station folder or dataset root on this machine")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = args.folder.resolve()
    if not root.is_dir() or any(part.lower() in QUARANTINE_FOLDERS for part in root.parts):
        parser.error("Choose a station folder or dataset root outside quarantine")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    except OSError as error:
        parser.exit(1, f"{error}\n")
    server.viewer = Viewer(root)
    print(f"Dataset {root}")
    print(f"Viewer http://127.0.0.1:{args.port}")
    print("Buttons move files immediately. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
