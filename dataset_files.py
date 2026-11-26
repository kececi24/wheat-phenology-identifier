"""Filesystem operations shared by automated cleaners and manual review."""

import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
QUARANTINE_FOLDERS = {"bad_images", "truncated_images", "corrupt_images"}


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
    move_image(source, destination, fingerprint)
    return destination


def move_image(source, destination, fingerprint):
    if source.resolve() != source or destination.resolve() != destination:
        raise ValueError("Source or destination is redirected")
    stat = source.stat()
    if not source.is_file() or (stat.st_size, stat.st_mtime_ns) != fingerprint:
        raise ValueError("Source changed during inspection")
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination exists {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    move_without_overwrite(source, destination)
    return destination


def move_without_overwrite(source, destination):
    """Move a file using exclusive destination creation."""
    if os.name == "nt":
        source.rename(destination)
    else:
        stat = source.stat()
        os.link(source, destination)
        try:
            source.unlink()
        except OSError:
            if destination.exists() and os.path.samestat(stat, destination.stat()):
                destination.unlink()
            raise
