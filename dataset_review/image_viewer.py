"""Review a remote dataset through a localhost browser and move images in place."""

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from getpass import getpass
from http.cookies import CookieError, SimpleCookie
import hmac
import json
import sqlite3
from pathlib import Path
import secrets
from threading import RLock
from urllib.parse import parse_qs, urlsplit
import warnings

from PIL import Image, ImageOps

from dataset_files import (
    QUARANTINE_FOLDERS,
    images_in_station,
    station_folders,
)


from .actions import ActionQueue
from .review_store import ReviewStore, SESSION_SECONDS


@dataclass
class ReviewImage:
    path: Path
    station: Path
    fingerprint: tuple


def camera_folder(path):
    if path.parent.name in {"1X", "10X"} and path.parent.parent.name in {"K1", "K2"}:
        return path.parent.parent
    return None


class Viewer:
    def __init__(self, root):
        self.root = root.resolve()
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
                    camera = camera_folder(path)
                    result.append({
                        "id": identifier, "name": path.name,
                        "path": path.relative_to(self.root).as_posix(),
                        "magnification": path.parent.name if path.parent.name in {"1X", "10X"} else "",
                        "camera": camera.name if camera and station in camera.parents else "",
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
        return render_preview(data), image.path.relative_to(self.root).as_posix()

    def proposal(self, identifier, action):
        with self.lock:
            image = self.current(identifier)
            path = image.path
            if action == "bad":
                destination = image.station / "bad_images" / path.relative_to(image.station)
            elif action in {"1X", "10X"} and path.parent.name in {"1X", "10X"}:
                destination = path.parent.with_name(action) / path.name
            elif action in {"K1", "K2"}:
                camera = camera_folder(path)
                if camera is None or image.station not in camera.parents:
                    raise ValueError("Image must be inside a K1 or K2 camera folder")
                destination = camera.with_name(action) / path.relative_to(camera)
            else:
                raise ValueError("Invalid move for this image")
            if destination == path:
                raise ValueError("Image is already in the requested folder")
            return path, destination, image.fingerprint


def render_preview(data):
    # Supply a missing JPEG EOI marker in memory.
    if data.startswith(bytes.fromhex("ffd8")) and not data.endswith(bytes.fromhex("ffd9")):
        data += bytes.fromhex("ffd9")
    with Image.open(BytesIO(data)) as source:
        frame = ImageOps.exif_transpose(source).convert("RGB")
        frame.thumbnail((1600, 1000))
        output = BytesIO()
        frame.save(output, format="JPEG", quality=90)
    return output.getvalue()


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal output focused on startup and completed moves.
        pass

    def send_body(self, status, body, content_type, cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if cookie:
            self.send_header("Set-Cookie", cookie)
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

    def send_json(self, status, data, cookie=None):
        self.send_body(status, json.dumps(data).encode(), "application/json; charset=utf-8", cookie)

    def local_request(self):
        try:
            hostname = urlsplit("http://" + self.headers.get("Host", "")).hostname
        except ValueError:
            return False
        return hostname in {"127.0.0.1", "localhost"}

    def session_token(self):
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return ""
        return cookie["review_session"].value if "review_session" in cookie else ""

    def do_GET(self):
        if not self.local_request():
            self.send_json(403, {"error": "Use localhost or 127.0.0.1"})
            return
        route = urlsplit(self.path)
        query = parse_qs(route.query)
        store = self.server.store
        queue = self.server.queue
        try:
            if route.path == "/":
                html = Path(__file__).with_suffix(".html").read_bytes()
                self.send_body(200, html, "text/html; charset=utf-8")
                return
            preview_data = None
            with store.transaction() as db:
                user = store.session(db, self.session_token())
                if user is None:
                    self.send_json(401, {"error": "Sign in to continue"})
                    return
                if route.path == "/api/session":
                    self.send_json(200, {key: user[key] for key in ("username", "role", "csrf")})
                    return
                if route.path == "/api/actions":
                    self.send_json(200, {"actions": queue.list(user)})
                    return
                if route.path == "/image" and "proposal" in query and user["role"] == "admin":
                    _, _, action = queue.find(user, query["proposal"][0])
                    if action["status"] != "pending" or not queue.matches(action["source"], action["fingerprint"]):
                        raise ValueError("Proposal preview is no longer available")
                    preview_data = queue.path(action["source"]).read_bytes()
            viewer = self.server.viewer
            if route.path == "/api/images":
                images = viewer.scan(query.get("folder", ["."])[0])
                self.send_json(200, {"images": images, "root": str(viewer.root)})
            elif route.path == "/image":
                if preview_data is not None:
                    preview = render_preview(preview_data)
                else:
                    preview, relative = viewer.preview(query.get("id", [""])[0])
                    if user["role"] == "reviewer":
                        with store.transaction():
                            store.log_view(user, relative)
                self.send_body(200, preview, "image/jpeg")
            else:
                self.send_json(404, {"error": "Unknown endpoint"})
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                Image.DecompressionBombWarning, sqlite3.Error) as error:
            self.send_json(409, {"error": str(error)})

    def do_POST(self):
        expected_origin = "http://" + self.headers.get("Host", "")
        if (
            not self.local_request() or self.headers.get("Origin") != expected_origin
            or self.headers.get_content_type() != "application/json"
        ):
            self.send_json(403, {"error": "Invalid viewer request"})
            return
        store = self.server.store
        queue = self.server.queue
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/login":
                username, password = payload.get("username"), payload.get("password")
                if not isinstance(username, str) or not isinstance(password, str):
                    raise ValueError("Expected username and password")
                token = store.login(username, password)
                if token is None:
                    self.send_json(401, {"error": "Sign-in failed. Check credentials or try again shortly."})
                    return
                self.send_json(200, {}, f"review_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}")
                return
            token = self.session_token()
            with store.transaction() as db:
                user = store.session(db, token)
                if user is None:
                    self.send_json(401, {"error": "Sign in to continue"})
                    return
                csrf = self.headers.get("X-Viewer-Token", "")
                if not hmac.compare_digest(csrf.encode(), user["csrf"].encode()):
                    self.send_json(403, {"error": "Invalid session token"})
                    return
                if self.path == "/api/logout":
                    store.logout(db, token)
                    result = {}
                elif self.path in {"/api/move", "/api/decision", "/api/undo"}:
                    identifier = payload.get("id")
                    if not isinstance(identifier, str):
                        raise ValueError("Expected action or image ID")
                    queue.recover()
                    if self.path == "/api/move":
                        action = payload.get("action")
                        if not isinstance(action, str):
                            raise ValueError("Expected move action")
                        with self.server.viewer.lock:
                            proposal = self.server.viewer.proposal(identifier, action)
                            result = queue.propose(user, *proposal)
                    elif self.path == "/api/undo":
                        result = queue.undo(user, identifier)
                    else:
                        decision = payload.get("decision")
                        if not isinstance(decision, str):
                            raise ValueError("Expected a proposal decision")
                        if decision in ("apply", "reject") and user["role"] != "admin":
                            self.send_json(403, {"error": "Only admins can apply or reject proposals"})
                            return
                        result = queue.decide(user, identifier, decision)
                else:
                    self.send_json(404, {"error": "Unknown endpoint"})
                    return
            cookie = "review_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0" if self.path == "/api/logout" else None
            self.send_json(200, result, cookie)
        except (OSError, ValueError, sqlite3.Error) as error:
            self.send_json(409, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Station folder or dataset root on this machine")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path, help="Private directory for accounts, actions and views")
    accounts = parser.add_mutually_exclusive_group()
    accounts.add_argument("--create-admin", metavar="USERNAME")
    accounts.add_argument("--create-reviewer", metavar="USERNAME")
    accounts.add_argument("--disable-user", metavar="USERNAME")
    accounts.add_argument("--reset-password", metavar="USERNAME")
    args = parser.parse_args()
    root = args.folder.resolve()
    if not root.is_dir() or any(part.lower() in QUARANTINE_FOLDERS for part in root.parts):
        parser.error("Choose a station folder or dataset root outside quarantine")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    state = (args.state_dir or root.with_name(f".{root.name}-review")).resolve()
    if state.is_relative_to(root) or root.is_relative_to(state):
        parser.error("Choose a private state directory separate from the dataset")
    try:
        store = ReviewStore(state, root)
        if args.create_admin or args.create_reviewer or args.reset_password:
            password = getpass("Password: ")
            if password != getpass("Confirm password: "):
                raise ValueError("Passwords do not match")
            if args.reset_password:
                store.update_user(args.reset_password, "reset", password)
            else:
                store.create_user(args.create_admin or args.create_reviewer,
                                  "admin" if args.create_admin else "reviewer", password)
            return
        if args.disable_user:
            store.update_user(args.disable_user, "disable")
            return
        queue = ActionQueue(store, root)
        with store.transaction() as db:
            if not db.execute("SELECT 1 FROM users WHERE role = 'admin' AND enabled = 1").fetchone():
                raise ValueError("Create an admin with --create-admin USERNAME first")
            queue.recover()
        warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    server.viewer = Viewer(root)
    server.store = store
    server.queue = queue
    print(f"Dataset {root}")
    print(f"Viewer http://127.0.0.1:{args.port}")
    print(f"Review state {state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
