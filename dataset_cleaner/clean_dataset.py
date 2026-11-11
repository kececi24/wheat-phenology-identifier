#!/usr/bin/env python3
"""Inspect an image, station folder, or dataset root for image quality issues.

Examples:
    python scripts/clean_dataset.py image.jpeg
    python scripts/clean_dataset.py data/01.01
    python scripts/clean_dataset.py data --apply
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
import math
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
QUARANTINE_FOLDERS = {"bad_images", "truncated_images", "corrupt_images"}
BLUR_PRESETS = {"conservative": 15.0, "medium": 45.0, "aggressive": 100.0}


def laplacian_variance(array):
    laplacian = (
        array[1:-1, 2:] + array[1:-1, :-2]
        + array[2:, 1:-1] + array[:-2, 1:-1] - 4.0 * array[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def rigorous_issue(image):
    """Apply the additional checks from the earlier station-specific scanner."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    means = rgb.mean(axis=(0, 1))
    color_difference = np.mean(
        (np.abs(rgb[:, :, 0] - rgb[:, :, 1])
         + np.abs(rgb[:, :, 1] - rgb[:, :, 2])
         + np.abs(rgb[:, :, 2] - rgb[:, :, 0])) / 3.0
    )
    if color_difference < 3.0 and (means[0] < 40 or means[0] > 200):
        return "NIGHT_OR_FOG", "Grayscale night or fog profile"
    if np.all(means > 220):
        return "OVEREXPOSED", "All channel means exceed 220"
    bright_percent = float(np.mean(rgb > 250) * 100)
    if bright_percent > 30:
        return "GLARE_SUSPECT", f"Saturated channel values {bright_percent:.1f}%"
    if rgb.std(axis=(0, 1)).mean() < 15:
        return "LOW_CONTRAST", "Mean channel standard deviation below 15"
    gray = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.BILINEAR),
                      dtype=np.float32)
    if laplacian_variance(gray) < 80:
        return "BORDERLINE_BLUR", "Whole-image Laplacian variance below 80"
    if np.all(means < 25):
        return "UNDEREXPOSED", "All channel means below 25"
    return "OK", ""


def analyze_image(path, blur_threshold=45.0, monochrome=False, local_blur=True, rigorous=False):
    """Return a status and explanation, using the original grid blur thresholds."""
    try:
        if path.stat().st_size == 0:
            return "CORRUPT", "Empty file"
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.format == "JPEG":
                    with path.open("rb") as file:
                        file.seek(-2, os.SEEK_END)
                        if file.read() != b"\xff\xd9":
                            return "TRUNCATED", "JPEG end marker missing"
                gray = np.asarray(
                    image.convert("L").resize((256, 256), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                if monochrome:
                    rgb = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.NEAREST),
                                     dtype=np.float32)
                    difference = np.mean(
                        np.abs(rgb[:, :, 0] - rgb[:, :, 1])
                        + np.abs(rgb[:, :, 1] - rgb[:, :, 2])
                    )
                    if difference < 3:
                        return "MONOCHROME", "Grayscale or monochrome camera image"
                if rigorous:
                    status, detail = rigorous_issue(image)
                    if status != "OK":
                        return status, detail
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        return "CORRUPT", str(error)

    mean = float(gray.mean())
    contrast = float(gray.std())
    if contrast < 1:
        if mean < 10:
            return "DARK", f"Black image, mean {mean:.2f}"
        if mean > 245:
            return "WHITE", f"White image, mean {mean:.2f}"
        return "LOW_CONTRAST", f"Uniform image, mean {mean:.2f}"
    if contrast < 5:
        return "LOW_CONTRAST", f"Standard deviation {contrast:.2f}"
    if mean < 5:
        return "DARK", f"Mean brightness {mean:.2f}"
    if mean > 250:
        return "WHITE", f"Mean brightness {mean:.2f}"

    # Exclude the center columns so the camera pole cannot mask blurred scenery.
    patches = sorted(
        laplacian_variance(gray[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64])
        for row in range(4) for col in (0, 3)
    )
    average = float(np.mean(patches))
    if average < blur_threshold:
        return "BLURRED", f"Side-patch Laplacian variance {average:.1f} < {blur_threshold:.1f}"
    if local_blur:
        threshold = max(10.0, blur_threshold * 0.45)
        if patches[len(patches) // 4] < threshold or patches[0] < threshold * 0.5:
            return "LOCAL_BLUR", f"Blurred side patches, minimum variance {patches[0]:.1f}"
    return "OK", ""


def raise_walk_error(error):
    raise error


def images_in_station(station):
    for directory, folders, filenames in os.walk(station, onerror=raise_walk_error):
        parent = Path(directory)
        folders[:] = sorted(
            name for name in folders
            if name.lower() not in QUARANTINE_FOLDERS
            and (parent / name).resolve() == parent / name
        )
        for name in sorted(filenames):
            path = parent / name
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.is_symlink():
                yield path


def station_folders(folder):
    children = sorted(folder.iterdir())
    # Project stations contain year folders. Direct images also identify a station.
    if any(
        (child.is_dir() and len(child.name) == 4 and child.name.isdigit())
        or (child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS)
        for child in children
    ):
        return [folder]
    return [
        child for child in children
        if child.is_dir() and child.name.lower() not in QUARANTINE_FOLDERS
        and child.resolve() == child
    ]


def move_bad_image(source, station, fingerprint):
    destination = station / "bad_images" / source.relative_to(station)
    if source.resolve() != source or destination.resolve() != destination:
        raise ValueError("Source or destination is redirected")
    stat = source.stat()
    if not source.is_file() or (stat.st_size, stat.st_mtime_ns) != fingerprint:
        raise ValueError("Source changed during inspection")
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination exists {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        source.rename(destination)
    else:
        os.link(source, destination)
        source.unlink()
    return destination


def inspect_task(task, **options):
    path, station = task
    try:
        stat = path.stat()
        fingerprint = stat.st_size, stat.st_mtime_ns
    except OSError as error:
        return path, station, None, "ERROR", str(error)
    status, detail = analyze_image(path, **options)
    return path, station, fingerprint, status, detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, help="Image, station folder, or dataset root")
    parser.add_argument("--data_dir", "--folder", "-f", dest="legacy_input", type=Path,
                        help="Alternative spelling for the input path")
    parser.add_argument("--apply", action="store_true", help="Move bad images into station/bad_images")
    parser.add_argument("--mode", "-m", choices=BLUR_PRESETS, default="medium")
    parser.add_argument("--blur-threshold", "-b", type=float, help="Override grid blur threshold")
    parser.add_argument("--monochrome", action="store_true", help="Flag grayscale camera images")
    parser.add_argument("--rigorous", action="store_true", help="Also check glare, fog, and stricter exposure")
    parser.add_argument("--local-blur", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", "-w", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--recursive", "-r", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-repair", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.input and args.legacy_input:
        parser.error("Supply one input path")
    target = args.input or args.legacy_input
    if target is None:
        parser.error("An input path is required")
    target = target.resolve()
    threshold = args.blur_threshold if args.blur_threshold is not None else BLUR_PRESETS[args.mode]
    if not math.isfinite(threshold) or threshold < 0 or args.workers < 1:
        parser.error("Blur threshold must be finite and nonnegative, and workers must be positive")
    options = dict(blur_threshold=threshold, monochrome=args.monochrome,
                   local_blur=args.local_blur, rigorous=args.rigorous)
    try:
        if target.is_file():
            if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
                parser.error("Unsupported image extension")
            status, _ = analyze_image(target, **options)
            print("GOOD" if status == "OK" else "BAD")
            return 0
        if not target.is_dir():
            parser.error(f"Input does not exist {target}")
        if any(part.lower() in QUARANTINE_FOLDERS for part in target.parts):
            parser.error("Choose a station or dataset outside a quarantine folder")
        tasks = [
            (path, station) for station in station_folders(target)
            for path in images_in_station(station)
        ]
        inspect = partial(inspect_task, **options)
        pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 and tasks else nullcontext()
        counts = Counter()
        with pool as executor:
            results = executor.map(inspect, tasks, chunksize=32) if executor else map(inspect, tasks)
            for path, station, fingerprint, status, detail in results:
                counts["scanned"] += 1
                if status == "OK":
                    continue
                counts["bad"] += 1
                print(f"BAD {path} [{status}] {detail}", flush=True)
                if args.apply:
                    try:
                        destination = move_bad_image(path, station, fingerprint)
                    except (OSError, ValueError) as error:
                        counts["skipped"] += 1
                        print(f"SKIPPED {path} {error}", file=sys.stderr)
                    else:
                        counts["moved"] += 1
                        print(f"MOVED {destination}", flush=True)
        print(f"Scanned {counts['scanned']} | Good {counts['scanned'] - counts['bad']} | "
              f"Bad {counts['bad']} | Moved {counts['moved']} | Skipped {counts['skipped']}")
        return int(counts["skipped"] > 0)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
