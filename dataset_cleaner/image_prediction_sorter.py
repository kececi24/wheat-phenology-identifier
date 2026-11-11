"""Audit magnification labels with five trained folds, then apply a reviewed CSV."""

import argparse
import csv
import os
from collections import Counter
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from scripts.audit_image_magnification import (
    IMAGE_EXTENSIONS,
    QUARANTINE_DIRECTORIES,
    STREAMS,
    parse_filename_label,
)

FOLD_COUNT = 5
FOLD_FIELDS = [
    f"fold_{fold}_{field}"
    for fold in range(1, FOLD_COUNT + 1)
    for field in ("label", "confidence")
]
REPORT_FIELDS = [
    "data_root", "source", "folder_label", "filename_label", "ensemble_label",
    "votes_1x", "votes_10x", *FOLD_FIELDS, "status", "destination", "size",
    "mtime_ns", "detail",
]


def read_config(path):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def trusted_directories(root, config):
    return {(root / folder).resolve() for folder in config["data"]["trusted_folders"]}


def stream_label(path):
    if len(path.parts) == 5 and path.parts[3].upper() in STREAMS:
        if not QUARANTINE_DIRECTORIES.intersection(part.lower() for part in path.parts[:3]):
            return path.parts[3].upper()
    return ""


def dataset_images(root):
    for directory, folders, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        depth = len(parent.relative_to(root).parts)
        if depth < 4:
            folders[:] = sorted(
                name for name in folders
                if name.lower() not in QUARANTINE_DIRECTORIES
                and (parent / name).resolve() == parent / name
                and (depth < 3 or name.upper() in STREAMS)
            )
        else:
            folders.clear()
            for name in sorted(files):
                path = parent / name
                if path.suffix.lower() in IMAGE_EXTENSIONS and not path.is_symlink():
                    yield path


def summarize_vote(folder_label, filename_label, predictions, min_votes):
    counts = Counter(predictions)
    winner, votes = counts.most_common(1)[0]
    if votes < min_votes:
        return "", counts, "LOW_CONSENSUS"
    if winner != folder_label:
        return winner, counts, "MOVE"
    if filename_label and filename_label != winner:
        return winner, counts, "FILENAME_MISMATCH"
    return winner, counts, "MATCH"


def audit_rows(root, model_dir, trusted, batch_size, min_votes, device_name):
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    folds = []
    resize_image = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    normalizers = {}
    for fold in range(1, FOLD_COUNT + 1):
        folder = model_dir / f"fold_{fold}"
        processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(folder, local_files_only=True)
        if sorted(model.config.id2label.values()) != sorted(STREAMS):
            raise ValueError(f"Unexpected model labels in {folder}: {model.config.id2label}")
        normalization = tuple(processor.image_mean), tuple(processor.image_std)
        normalizers[normalization] = transforms.Normalize(*normalization)
        folds.append((model.to(device).eval(), normalization))

    paths = iter(dataset_images(root))
    while batch := list(islice(paths, batch_size)):
        rows = []
        images = []
        for path in batch:
            relative = path.relative_to(root)
            row = {
                "data_root": str(root),
                "source": relative.as_posix(),
                "folder_label": stream_label(relative),
                "filename_label": parse_filename_label(path.name),
            }
            try:
                stat = path.stat()
                with Image.open(path) as source:
                    image = resize_image(source.convert("RGB"))
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                row.update(status="READ_ERROR", detail=str(error))
                yield row
                continue
            row.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            rows.append(row)
            images.append(image)
        if not images:
            continue

        with torch.inference_mode():
            batch_tensor = torch.stack(images).to(device)
            inputs = {key: normalize(batch_tensor) for key, normalize in normalizers.items()}
            for fold, (model, normalization) in enumerate(folds, start=1):
                probabilities = model(pixel_values=inputs[normalization]).logits.float().softmax(dim=1)
                confidences, predictions = probabilities.max(dim=1)
                for row, prediction, confidence in zip(rows, predictions.tolist(), confidences.tolist()):
                    row[f"fold_{fold}_label"] = model.config.id2label[prediction]
                    row[f"fold_{fold}_confidence"] = confidence

        for row in rows:
            labels = [row[f"fold_{fold}_label"] for fold in range(1, FOLD_COUNT + 1)]
            winner, counts, status = summarize_vote(
                row["folder_label"], row["filename_label"], labels, min_votes
            )
            source = Path(row["source"])
            destination = source.parent.with_name(winner) / source.name if status == "MOVE" else None
            if root / source.parent in trusted or (destination and root / destination.parent in trusted):
                status = "TRUSTED"
            elif destination and os.path.lexists(root / destination):
                status = "DESTINATION_EXISTS"
            row.update(ensemble_label=winner, votes_1x=counts["1X"], votes_10x=counts["10X"],
                       status=status, destination=destination.as_posix() if destination else "")
            yield row


def write_audit(report, rows):
    if os.path.lexists(report):
        raise FileExistsError(f"Report already exists: {report}")
    counts = Counter()
    temporary = NamedTemporaryFile(
        mode="w", dir=report.parent, suffix=".tmp", newline="", encoding="utf-8", delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                writer.writerow(row)
                counts[row["status"]] += 1
                if index % 1000 == 0:
                    print(f"Audited {index} images", flush=True)
        move_without_overwrite(temporary_path, report)
    finally:
        temporary_path.unlink(missing_ok=True)
    return counts


def move_without_overwrite(source, destination):
    if os.name == "nt":
        source.rename(destination)  # Windows rename fails if the destination exists.
    else:
        os.link(source, destination)  # Exclusive destination creation on POSIX.
        source.unlink()


def apply_move(root, trusted, row):
    relative = Path(row["source"])
    label = stream_label(relative)
    winner = row["ensemble_label"]
    if (
        Path(row["data_root"]) != root or relative.anchor or ".." in relative.parts
        or not label or winner not in STREAMS or winner == label
        or relative.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise ValueError("Invalid source or ensemble label for this dataset")
    source = root / relative
    destination = source.parent.with_name(winner) / source.name
    if Path(row["destination"]) != destination.relative_to(root):
        raise ValueError("Destination must be the sibling magnification folder with the original filename")
    for path in (source, destination):
        if path.resolve() != path or path.parent in trusted:
            raise ValueError(f"Path is redirected or in a trusted folder: {path}")
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination exists: {destination}")
    stat = source.stat()
    if not source.is_file() or (stat.st_size, stat.st_mtime_ns) != (int(row["size"]), int(row["mtime_ns"])):
        raise ValueError("Source changed since the audit")
    destination.parent.mkdir(exist_ok=True)
    move_without_overwrite(source, destination)


def apply_report(root, trusted, report):
    counts = Counter()
    with report.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        if not set(REPORT_FIELDS).issubset(rows.fieldnames or []):
            raise ValueError("Expected a magnification ensemble audit CSV")
        for row in rows:
            if row["status"] != "MOVE":
                continue
            try:
                apply_move(root, trusted, row)
            except (OSError, TypeError, ValueError) as error:
                counts["SKIPPED"] += 1
                print(f"Skipped {row['source']}: {error}")
            else:
                counts["MOVED"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    audit = modes.add_parser("audit", help="Write per-image predictions and proposed moves")
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--min-votes", type=int, choices=range(3, FOLD_COUNT + 1), default=3)
    audit.add_argument("--batch-size", type=int, default=16)
    audit.add_argument("--device", default="auto")
    apply = modes.add_parser("apply", help="Move only rows marked MOVE in a reviewed report")
    apply.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        config_path = args.config.resolve()
        config = read_config(config_path)
        root = (config_path.parent / config["data"]["root"]).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Dataset root must be a directory: {root}")
        model_dir = (config_path.parent / config["output"]["model_dir"]).resolve()
        trusted = trusted_directories(root, config)
        completed_config = model_dir / "loaded_config.yaml"
        if args.mode == "audit" and not completed_config.is_file():
            raise ValueError(f"No completed five-fold training run in {model_dir}")
        if args.mode == "audit" or completed_config.exists():
            trained = read_config(completed_config)
            trusted.update(trusted_directories(root, trained))
        if args.mode == "audit":
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            if trained["data"]["folds"] != FOLD_COUNT:
                raise ValueError(f"Expected a completed {FOLD_COUNT}-fold training run in {model_dir}")
            rows = audit_rows(root, model_dir, trusted, args.batch_size, args.min_votes, args.device)
            counts = write_audit(args.report, rows)
            print(f"Audit saved to {args.report}")
        else:
            counts = apply_report(root, trusted, args.report)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")

    print(f"Total {args.mode} rows: {counts.total()}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
