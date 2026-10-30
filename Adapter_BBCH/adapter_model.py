"""Training-only basis estimation and small residual adapters.

All bases use exactly the same parameter-free LayerNorm coordinates as inference.
No station identifier, stage label, or date is an input to the classifier.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("none", "transition", "random", "pca", "learned")


def normalized_features(features: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(features.float(), (features.shape[-1],))


def _check_features(features):
    if features.ndim != 2 or min(features.shape) < 1 or not torch.isfinite(features).all():
        raise ValueError("Expected nonempty finite features [images, feature_dim]")


def transition_basis(features, rows, num_classes, rank=6, min_stage_images=2,
                     min_stations=2):
    """Estimate a basis from TRAINING rows only; caller must split first.

    Average images within station-year/stage, differences within station-year,
    differences across years within station, then across stations equally.
    Soft targets train the classifier; hard interval labels define prototypes.
    Missing/unsupported adjacent boundaries are skipped, never bridged.
    """
    _check_features(features)
    if len(rows) != len(features):
        raise ValueError("Features and prototype rows must have identical ordering")
    if min(rank, min_stage_images, min_stations) < 1:
        raise ValueError("Rank and support thresholds must be positive")
    required = {"station_code", "station_year", "class_idx"}
    if not required.issubset(rows.columns) or rows[list(required)].isna().any().any():
        raise ValueError(f"Prototype rows require nonmissing {sorted(required)}")
    if not rows["class_idx"].isin(range(num_classes)).all():
        raise ValueError("Prototype class indices are outside the class order")
    z = normalized_features(features.detach().cpu()).double()
    table = rows.reset_index(drop=True).copy()
    # One season must not accidentally combine two physical stations.
    if table.groupby("station_year")["station_code"].nunique().max() != 1:
        raise ValueError("A station_year belongs to more than one station")
    prototypes = {}
    for (station, year, stage), indices in table.groupby(
            ["station_code", "station_year", "class_idx"], sort=True).indices.items():
        if len(indices) >= min_stage_images:
            prototypes[(str(station), str(year), int(stage))] = z[indices].mean(0)
    columns, boundaries = [], []
    seasons = table[["station_code", "station_year"]].astype(str).drop_duplicates()
    for stage in range(num_classes - 1):
        per_station = {}
        for station, year in seasons.itertuples(index=False, name=None):
            before = prototypes.get((station, year, stage))
            after = prototypes.get((station, year, stage + 1))
            if before is not None and after is not None:
                per_station.setdefault(station, []).append(after - before)
        station_deltas = [torch.stack(values).mean(0) for values in per_station.values()]
        supported = len(station_deltas) >= min_stations
        entry = {"from_idx": stage, "to_idx": stage + 1,
                 "stations": sorted(per_station), "station_count": len(per_station),
                 "supported": supported}
        if supported:
            deltas = torch.stack(station_deltas)
            delta = deltas.mean(0)
            entry["mean_norm"] = float(delta.norm())
            entry["station_variation_rms"] = float((deltas - delta).square().sum(1).mean().sqrt())
            columns.append(delta)
        boundaries.append(entry)
    if not columns:
        raise ValueError("No supported adjacent transitions in training data; lower the "
                         "prototype support thresholds or collect more training stages")
    matrix = torch.stack(columns, dim=1)
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular.max()
    available = int((singular > tolerance).sum())
    effective_rank = min(rank, available)
    if effective_rank < 1:
        raise ValueError("Training transition directions have zero numerical rank")
    # Deterministic sign convention matters because GELU is not sign symmetric.
    basis = u[:, :effective_rank].T.contiguous()
    for vector in basis:
        if vector[vector.abs().argmax()] < 0:
            vector.neg_()
    report = {"requested_rank": rank, "effective_rank": effective_rank,
              "available_rank": available, "singular_values": singular.tolist(),
              "min_stage_images": min_stage_images, "min_stations": min_stations,
              "training_images": len(rows),
              "training_stations": sorted(table["station_code"].astype(str).unique()),
              "boundaries": boundaries}
    return basis.float(), report


def control_basis(method, training_features, rank, seed):
    """Same coordinates/rank as the transition arm; PCA sees TRAINING only."""
    _check_features(training_features)
    dim = training_features.shape[1]
    if not 1 <= rank <= dim:
        raise ValueError("Basis rank must be between one and feature_dim")
    if method in {"random", "learned", "none"}:
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=generator), mode="reduced")
        return q.T.contiguous()
    if method == "pca":
        z = normalized_features(training_features.detach().cpu()).double()
        _, singular, vh = torch.linalg.svd(z - z.mean(0), full_matrices=False)
        tolerance = max(z.shape) * torch.finfo(z.dtype).eps * singular.max()
        if int((singular > tolerance).sum()) < rank:
            raise ValueError("Training PCA has less numerical rank than the matched adapter")
        basis = vh[:rank].clone()
        for vector in basis:
            if vector[vector.abs().argmax()] < 0:
                vector.neg_()
        return basis.float()
    raise ValueError(f"Unknown control basis: {method}")


def select_label_subset(rows, fraction=1.0, seed=42):
    """Select a deterministic, nested, stratified subset of TRAINING labels.

    Sampling is nested across fractions for a fixed seed. Each station-year/stage
    group retains at least one image, so the realized fraction can exceed the
    requested fraction, especially in small groups.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1]")
    if fraction == 1:
        return np.arange(len(rows))
    rng = np.random.default_rng(seed)
    indices = []
    for positions in rows.reset_index(drop=True).groupby(
            ["station_code", "station_year", "class_idx"], sort=True).indices.values():
        count = max(1, math.ceil(fraction * len(positions)))
        # Take a prefix of one seeded permutation per stratum. This keeps the
        # lower-budget subset inside every higher-budget subset for the same seed.
        indices.extend(rng.permutation(positions)[:count].tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


class AdapterClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, basis, method="transition",
                 hidden_dim=128, dropout=0.2):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown adapter method: {method}")
        if basis.ndim != 2 or basis.shape[1] != feature_dim or not torch.isfinite(basis).all():
            raise ValueError("Basis must be finite [rank, feature_dim]")
        if min(feature_dim, num_classes, hidden_dim, basis.shape[0]) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid model dimensions or dropout")
        self.config = dict(feature_dim=feature_dim, num_classes=num_classes,
                           method=method, hidden_dim=hidden_dim, dropout=dropout,
                           rank=basis.shape[0])
        self.method = method
        if method == "learned":
            self.basis = nn.Parameter(basis.detach().float().clone())
        else:
            self.register_buffer("basis", basis.detach().float().clone())
        # Initialize the head before the optional up-projection so the same seed
        # produces identical heads for every arm, including the no-adapter arm.
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        if method != "none":
            self.up = nn.Linear(basis.shape[0], feature_dim, bias=False)
            nn.init.zeros_(self.up.weight)
        else:
            self.up = None

    def forward(self, features):
        z = normalized_features(features)
        if self.up is not None:
            z = z + self.up(F.gelu(F.linear(z, self.basis)))
        return self.head(z)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        config = dict(checkpoint["model_config"])
        rank = config.pop("rank")
        basis = torch.zeros(rank, config["feature_dim"])
        model = cls(basis=basis, **config)
        model.load_state_dict(checkpoint["model"])
        return model
