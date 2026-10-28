"""Reuse DINOv3 BBCH labeling/folds; pool frozen caches without learned weights."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DINOv3_BBCH.multiscale_phenology import (
    BASE_CLASSES,
    build_multiscale_daily_dataframe,
    generate_loso_train_val_test_folds,
)


CLASSES = [name for name in BASE_CLASSES if name != "OffSeason"]
POOLING = "mean_descriptors_then_mean_tiles_v1"


def image_key(path):
    """Portable data/<station>/<year>/<camera>/<scale>/<filename> identity."""
    parts = str(path).replace("\\", "/").split("/")
    if len(parts) >= 5 and re.fullmatch(r"\d{2}\.\d{2}", parts[-5]):
        return "/".join(parts[-5:])
    return str(Path(path).resolve()).replace("\\", "/")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_rows(args):
    frame = build_multiscale_daily_dataframe(
        args.label_path, args.data_path, include_preplant_days=0,
        include_postharvest_days=0, transition_days=args.transition_days,
        date_tolerance_days=5, classes=BASE_CLASSES,
        preferred_camera=args.camera, use_status_csv=not args.ignore_status_csv,
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No BBCH metadata discovered")
    frame["station_code"] = frame["station_code"].astype(str)
    # Preserve the full discovered station universe when constructing LOSO.
    folds = generate_loso_train_val_test_folds(
        frame, group_col="station_code", n_val=args.validation_groups,
        random_state=args.fold_seed,
    )
    stations = sorted(frame["station_code"].unique())
    if args.expected_stations is not None and len(stations) != args.expected_stations:
        raise ValueError(f"Expected {args.expected_stations} stations, found {len(stations)}: {stations}")
    assignments = []
    for fold_id, split in enumerate(folds, 1):
        for role, indices in zip(("train", "validation", "test"), split):
            for station in sorted(frame.iloc[indices]["station_code"].unique()):
                assignments.append(dict(fold=fold_id, role=role, station_code=station))
    path_column = f"{args.stream}_path"
    rows = frame.loc[frame["label"].isin(CLASSES) & frame[path_column].notna()].copy()
    rows["image_path"] = rows[path_column].astype(str)
    rows["image_key"] = rows["image_path"].map(image_key)
    rows["class_idx"] = rows["label"].map(CLASSES.index)
    active_indices = [BASE_CLASSES.index(name) for name in CLASSES]

    def active_target(value):
        full = ast.literal_eval(value) if isinstance(value, str) else value
        target = np.asarray(full, dtype=np.float32)[active_indices]
        if not np.isfinite(target).all() or (target < 0).any() or target.sum() <= 0:
            raise ValueError("Invalid BBCH target probability vector")
        return (target / target.sum()).tolist()

    rows["active_target"] = rows["target"].map(active_target)
    if rows["image_key"].duplicated().any():
        raise ValueError("Duplicate image identity in generated metadata; audit labels/cameras")
    rows = rows.sort_values(["station_code", "station_year", "date"]).reset_index(drop=True)
    missing_stations = set(stations) - set(rows["station_code"])
    if missing_stations:
        raise ValueError(f"Stations have no in-season {args.stream} images: {sorted(missing_stations)}")
    return rows, pd.DataFrame(assignments)


def pool_feature(value):
    if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2, 3):
        raise ValueError("Expected cache tensor [D], [tiles,D], or [tiles,descriptors,D]")
    if min(value.shape) < 1 or not torch.isfinite(value).all():
        raise ValueError("Cache features must be nonempty and finite")
    value = value.detach().float().cpu()
    while value.ndim > 1:
        value = value.mean(0)
    return value


def load_features(cache_path, rows, stream):
    """Only a frozen, public pretrained DINOv3 cache is valid across LOSO folds.

    Local supervised backbones are intentionally rejected: their update stations
    cannot be inferred from a global cache. No silent held-out row dropping.
    """
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    backbone = str(cache.get("image_backbone", ""))
    if (cache.get("pretrained") is not True or not backbone.startswith("facebook/dinov3-")
            or cache.get("cache_format") not in {"dinov3_dense_v1", "dinov3_global_v1"}):
        raise ValueError("Use an original frozen pretrained DINOv3 cache from "
                         "DINOv3_BBCH/precompute_multiscale_embeddings.py; "
                         "local/fine-tuned/unverified backbone caches are not fold-safe")
    if any(cache.get(key) for key in ("finetuned", "fine_tuned", "supervised_finetuning")):
        raise ValueError("Supervised backbone cache cannot be shared across LOSO folds")
    table = {}
    for path, value in cache.get(stream, {}).items():
        key = image_key(path)
        if key in table:
            raise ValueError(f"Duplicate portable cache identity: {key}")
        table[key] = value
    missing = [key for key in rows["image_key"] if key not in table]
    if missing:
        raise ValueError(f"Cache is missing {len(missing)}/{len(rows)} target images; "
                         f"validation/test are never silently filtered. Examples: {missing[:3]}")
    features = torch.stack([pool_feature(table[key]) for key in rows["image_key"]])
    if features.shape[1] != int(cache["feature_dim"]):
        raise ValueError("Cache tensor dimension disagrees with feature_dim")
    provenance = {key: cache.get(key) for key in (
        "cache_format", "image_backbone", "pretrained", "feature_dim", "preprocess", "dense_features")}
    provenance["tiling"] = {key: value for key, value in cache.get("tiling", {}).items()
                            if key in {"tile_streams", "tile_size", "tile_stride", "max_tiles",
                                       "vit_image_size", "tile_pooling"}}
    provenance.update(pooling=POOLING, stream=stream,
                      cache_sha256=file_sha256(cache_path))
    return features, provenance


def split_indices(rows, assignments, fold_id):
    split = assignments.loc[assignments["fold"] == fold_id]
    sets = {role: set(split.loc[split["role"] == role, "station_code"].astype(str))
            for role in ("train", "validation", "test")}
    if (any(not group for group in sets.values()) or len(sets["test"]) != 1
            or sets["train"] & sets["validation"] or sets["train"] & sets["test"]
            or sets["validation"] & sets["test"]):
        raise ValueError("Invalid or overlapping station split")
    indices = {role: np.flatnonzero(rows["station_code"].isin(stations).to_numpy())
               for role, stations in sets.items()}
    if any(len(values) == 0 for values in indices.values()):
        raise ValueError("Every split must have at least one image")
    return indices
