"""Accounts, sessions, and on-disk review records."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from tempfile import NamedTemporaryFile
import time

SESSION_SECONDS = 12 * 60 * 60


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=1,
                          maxmem=64 * 1024 * 1024, dklen=32)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class ReviewStore:
    def __init__(self, directory, root):
        self.directory = directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.actions_dir = directory / "actions"
        self.views_dir = directory / "views"
        for path in (self.actions_dir, self.views_dir):
            path.mkdir(mode=0o700, exist_ok=True)
        self.database = directory / "accounts.sqlite3"
        self.database.touch(mode=0o600, exist_ok=True)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (root TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, role TEXT NOT NULL,
                salt BLOB NOT NULL, password BLOB NOT NULL, enabled INTEGER NOT NULL DEFAULT 1)""")
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, csrf TEXT NOT NULL,
                expires INTEGER NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS login_limit (
                id INTEGER PRIMARY KEY CHECK(id = 1), attempts INTEGER NOT NULL, reset_at INTEGER NOT NULL)""")
            setting = db.execute("SELECT root FROM settings").fetchone()
            if setting is None:
                db.execute("INSERT INTO settings VALUES (?)", (str(root),))
            elif setting["root"] != str(root):
                raise ValueError("This state directory belongs to a different dataset root")

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(self.database, timeout=30)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            yield db

    def create_user(self, username, role, password):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", username):
            raise ValueError("Use 1–64 letters, digits, dots, underscores or hyphens for the username")
        salt, digest = self.password_fields(password)
        with self.transaction() as db:
            db.execute("INSERT INTO users(username, role, salt, password) VALUES (?, ?, ?, ?)",
                       (username, role, salt, digest))

    @staticmethod
    def password_fields(password):
        if len(password) < 12 or len(password.encode()) > 1024:
            raise ValueError("Use a password of at least 12 characters and at most 1024 bytes")
        salt = secrets.token_bytes(16)
        return salt, password_hash(password, salt)

    def update_user(self, username, operation, password=None):
        fields = self.password_fields(password) if operation == "reset" else None
        with self.transaction() as db:
            user = db.execute("SELECT id, role FROM users WHERE username = ?", (username,)).fetchone()
            if user is None:
                raise ValueError("Unknown account")
            if operation == "disable":
                if user["role"] == "admin" and db.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND enabled = 1 AND id != ?",
                    (user["id"],),
                ).fetchone()[0] == 0:
                    raise ValueError("Keep at least one enabled admin")
                db.execute("UPDATE users SET enabled = 0 WHERE id = ?", (user["id"],))
            else:
                db.execute("UPDATE users SET salt = ?, password = ? WHERE id = ?", (*fields, user["id"]))
            db.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))

    def login(self, username, password):
        if len(password.encode()) > 1024:
            return None
        now = int(time.time())
        with self.transaction() as db:
            limit = db.execute("SELECT attempts, reset_at FROM login_limit").fetchone()
            attempts = limit["attempts"] if limit and limit["reset_at"] > now else 0
            if attempts >= 30:
                return None
            reset_at = limit["reset_at"] if attempts else now + 60
            db.execute("INSERT OR REPLACE INTO login_limit VALUES (1, ?, ?)", (attempts + 1, reset_at))
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            candidate = password_hash(password, user["salt"] if user else bytes(16))
            if user is None or not user["enabled"] or not hmac.compare_digest(candidate, user["password"]):
                return None
            db.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                       (token_hash(token), user["id"], secrets.token_urlsafe(32), now + SESSION_SECONDS))
            return token

    def session(self, db, token):
        return db.execute("""SELECT users.id, username, role, csrf FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE token = ? AND expires > ? AND enabled = 1""",
            (token_hash(token), int(time.time()))).fetchone()

    def logout(self, db, token):
        db.execute("DELETE FROM sessions WHERE token = ?", (token_hash(token),))

    def user_actions(self, user_id):
        path = self.actions_dir / f"{user_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def action_lists(self):
        for path in sorted(self.actions_dir.glob("*.json")):
            yield int(path.stem), self.user_actions(int(path.stem))

    def save_actions(self, user_id, actions):
        destination = self.actions_dir / f"{user_id}.json"
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.actions_dir, delete=False) as fp:
            temporary = fp.name
            try:
                json.dump(actions, fp, indent=2, ensure_ascii=False)
                fp.write("\n")
                fp.flush()
                os.fsync(fp.fileno())
                fp.close()
                os.replace(temporary, destination)
            finally:
                fp.close()
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def log_view(self, user, relative_path):
        record = json.dumps({"at": timestamp(), "user": user["username"], "path": relative_path},
                            ensure_ascii=False)
        path = self.views_dir / f"{user['id']}.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(record + "\n")
