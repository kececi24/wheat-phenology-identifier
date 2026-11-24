"""Approval workflow for a single dataset."""

import os
import secrets

from dataset_files import move_image
from .review_store import timestamp


def fingerprint(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


class ActionQueue:
    def __init__(self, store, root):
        self.store = store
        self.root = root

    def path(self, relative):
        path = self.root / relative
        if path.resolve() != path or not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Action path is outside the dataset or redirected")
        return path

    def matches(self, relative, expected):
        try:
            path = self.path(relative)
            return path.is_file() and fingerprint(path) == expected
        except (OSError, ValueError):
            return False

    def list(self, user):
        if user["role"] == "admin":
            records = [action for _, actions in self.store.action_lists() for action in actions]
        else:
            records = self.store.user_actions(user["id"])
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def find(self, user, action_id):
        lists = (self.store.action_lists() if user["role"] == "admin"
                 else [(user["id"], self.store.user_actions(user["id"]))])
        for owner, actions in lists:
            for action in actions:
                if action["id"] == action_id:
                    return owner, actions, action
        raise ValueError("Action is unavailable")

    def propose(self, user, source, destination, expected):
        source = self.path(source.relative_to(self.root).as_posix())
        destination = self.path(destination.relative_to(self.root).as_posix())
        source_id = fingerprint(source)
        if source_id[2:] != list(expected):
            raise ValueError("Image changed since the folder was loaded")
        relative = source.relative_to(self.root).as_posix()
        actions = self.store.user_actions(user["id"])
        if any(item["source"] == relative and item["status"] == "pending" for item in actions):
            raise ValueError("You already have a pending proposal for this image")
        action = {
            "id": secrets.token_urlsafe(18), "reviewer": user["username"],
            "source": relative, "destination": destination.relative_to(self.root).as_posix(),
            "fingerprint": source_id, "created_at": timestamp(), "status": "pending",
        }
        actions.append(action)
        self.store.save_actions(user["id"], actions)
        if user["role"] == "admin":
            return self.decide(user, action["id"], "apply")
        return action

    def decide(self, user, action_id, decision):
        owner, actions, action = self.find(user, action_id)
        if action["status"] != "pending":
            raise ValueError("This proposal is no longer pending")
        if decision == "cancel":
            if owner != user["id"]:
                raise ValueError("Only the proposer can cancel this action")
            action["status"] = "cancelled"
        elif user["role"] == "admin" and decision in {"apply", "reject"}:
            if decision == "apply":
                return self.transfer(user, owner, actions, action, reverse=False)
            action["status"] = "rejected"
        else:
            raise ValueError("This action requires an admin")
        action.update(decided_by=user["username"], updated_at=timestamp())
        self.store.save_actions(owner, actions)
        return action

    def undo(self, user, action_id):
        if user["role"] != "admin":
            return self.decide(user, action_id, "cancel")
        applied = [item for item in self.list(user) if item["status"] == "applied"]
        latest = max(applied, key=lambda item: item["applied_at"], default=None)
        if latest is None or latest["id"] != action_id:
            raise ValueError("Only the latest applied move can be undone")
        owner, actions, action = self.find(user, action_id)
        return self.transfer(user, owner, actions, action, reverse=True)

    def transfer(self, user, owner, actions, action, reverse):
        start, end = ("destination", "source") if reverse else ("source", "destination")
        original, destination = action[start], action[end]
        try:
            if not self.matches(original, action["fingerprint"]):
                raise ValueError("The source image changed or is missing")
            if os.path.lexists(self.path(destination)):
                raise FileExistsError("The destination already exists")
        except (OSError, ValueError) as error:
            action.update(status="applied" if reverse else "failed", error=str(error),
                          decided_by=user["username"], updated_at=timestamp())
            self.store.save_actions(owner, actions)
            return action
        action.pop("error", None)
        action.update(status="reverting" if reverse else "applying",
                      decided_by=user["username"], updated_at=timestamp())
        self.store.save_actions(owner, actions)
        try:
            move_image(self.path(original), self.path(destination), tuple(action["fingerprint"][2:]))
        except (OSError, ValueError) as error:
            self.reconcile(action, str(error), interrupted=False)
        else:
            self.complete(action, reverse)
        self.store.save_actions(owner, actions)
        return action

    def complete(self, action, reverse):
        action.update(status="reverted" if reverse else "applied", updated_at=timestamp())
        if reverse:
            action["reverted_by"] = action.get("decided_by")
        else:
            action["applied_at"] = action["updated_at"]
            action["applied_by"] = action.get("decided_by")

    def reconcile(self, action, error, interrupted):
        reverse = action["status"] == "reverting"
        start, end = ("destination", "source") if reverse else ("source", "destination")
        source_matches = self.matches(action[start], action["fingerprint"])
        destination_matches = self.matches(action[end], action["fingerprint"])
        try:
            source_absent = not self.path(action[start]).exists()
            destination_absent = not self.path(action[end]).exists()
        except (OSError, ValueError):
            source_absent = destination_absent = False
        if source_absent and destination_matches:
            self.complete(action, reverse)
            return
        if source_matches and not destination_matches:
            status = "applied" if reverse else ("pending" if interrupted and destination_absent else "failed")
        else:
            status = "failed"
            if source_matches and destination_matches:
                error += ". Both paths contain the image. Check both paths before continuing."
        action.update(status=status, error=error, updated_at=timestamp())

    def recover(self):
        for owner, actions in self.store.action_lists():
            interrupted = [item for item in actions if item["status"] in {"applying", "reverting"}]
            for action in interrupted:
                self.reconcile(action, "Interrupted move recovered from disk state", interrupted=True)
            if interrupted:
                self.store.save_actions(owner, actions)
