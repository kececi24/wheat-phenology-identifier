"""Compare four losses on fixed, frozen DINOv3 image features."""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Adapter_BBCH.adapter_model import select_label_subset
from Adapter_BBCH.data import (
    CLASSES,
    discover_rows,
    file_sha256,
    load_features,
    split_indices,
)
from Adapter_BBCH.run_cached_training import (
    classification_metrics,
    prediction_table,
    seed_everything,
    write_json,
)
from Loss_BBCH.config import METHODS, parse_args
from Loss_BBCH.losses import (
    build_sequences,
    hard_ce,
    monotone_path_loss,
    ordinal_loss,
    soft_ce,
)


class ImageClassifier(nn.Module):
    """Small image-only classifier shared by every loss arm.

    The feature cache is pooled before this model. LayerNorm has no affine
    parameters, followed by Linear -> GELU -> Dropout -> Linear.
    """

    def __init__(self, feature_dim, hidden_dim=32, dropout=0.2,
                 num_classes=len(CLASSES)):
        super().__init__()
        if min(int(feature_dim), int(hidden_dim), int(num_classes)) < 1:
            raise ValueError("Model dimensions must be positive")
        if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.config = {
            "architecture": "parameter_free_layernorm_mlp_v1",
            "feature_dim": int(feature_dim),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "num_classes": int(num_classes),
            "layer_norm_affine": False,
        }
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim, elementwise_affine=False),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected image features [batch,{self.feature_dim}], got {tuple(features.shape)}"
            )
        return self.network(features)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        config = checkpoint["model_config"]
        model = cls(
            feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            num_classes=config["num_classes"],
        )
        model.load_state_dict(checkpoint["model"])
        return model


def _as_index_array(values, name):
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must be a one-dimensional array of unique row indices")
    return indices


def _prepare_sequences(rows, train_indices, selected_indices):
    """Build and validate training-only paths, returning a compact audit table."""
    train_indices = _as_index_array(train_indices, "train_indices")
    selected_indices = _as_index_array(selected_indices, "selected_indices")
    train_set = set(train_indices.tolist())
    selected_set = set(selected_indices.tolist())
    if not selected_set or not selected_set.issubset(train_set):
        raise ValueError("Selected labels must be a nonempty subset of fold training rows")
    if min(train_indices.min(), selected_indices.min()) < 0 or max(train_indices.max(), selected_indices.max()) >= len(rows):
        raise ValueError("Training sequence index is outside the metadata table")

    raw_sequences = build_sequences(rows, train_indices, selected_indices)
    sequences = []
    audit_rows = []
    covered = set()
    for sequence_id, raw in enumerate(raw_sequences, start=1):
        indices = _as_index_array(raw["indices"], "sequence indices")
        if len(indices) == 0 or not set(indices.tolist()).issubset(train_set):
            raise ValueError("A path sequence is empty or contains a non-training row")
        if covered.intersection(indices.tolist()):
            raise ValueError("A training image appears in more than one season path")
        covered.update(indices.tolist())
        allowed = torch.as_tensor(raw["allowed"])
        if allowed.dtype != torch.bool or tuple(allowed.shape) != (len(indices), len(CLASSES)):
            raise ValueError("Path allowed-state mask must be bool [sequence_length, 7]")
        if not allowed.any(dim=1).all():
            raise ValueError("Every image in a path must allow at least one class")

        season_rows = rows.iloc[indices]
        if (season_rows["station_code"].astype(str).nunique() != 1
                or season_rows["station_year"].astype(str).nunique() != 1):
            raise ValueError("A path sequence crosses station or season boundaries")
        dates = pd.to_datetime(season_rows["date"], errors="raise")
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("A path sequence must have unique chronological dates")
        station_code = str(raw["station_code"])
        station_year = str(raw["station_year"])
        if (season_rows["station_code"].astype(str).iloc[0] != station_code
                or season_rows["station_year"].astype(str).iloc[0] != station_year):
            raise ValueError("Path metadata disagrees with its indexed station season")

        sequences.append({
            "indices": indices,
            "allowed": allowed,
            "station_code": station_code,
            "station_year": station_year,
        })
        for offset, (row_index, allowed_row) in enumerate(zip(indices, allowed.tolist())):
            row = rows.iloc[int(row_index)]
            allowed_idx = [i for i, is_allowed in enumerate(allowed_row) if is_allowed]
            audit_rows.append({
                "sequence_id": sequence_id,
                "station_code": station_code,
                "station_year": station_year,
                "position": offset,
                "sequence_length": len(indices),
                "image_key": row["image_key"],
                "date": str(row["date"]),
                "selected_label_anchor": int(row_index) in selected_set,
                "allowed_class_indices": json.dumps(allowed_idx),
                "allowed_classes": "|".join(CLASSES[i] for i in allowed_idx),
            })
    if not sequences:
        raise ValueError("No anchored training seasons are available for the path objective")
    return sequences, pd.DataFrame(audit_rows)


def _label_coverage(args, full_training_rows, selected_training_rows, sampling_seed):
    full_count = len(full_training_rows)
    selected_count = len(selected_training_rows)
    fraction = selected_count / full_count
    return {
        "requested_percentage": float(args.label_percentage),
        "realized_percentage": 100.0 * fraction,
        "realized_fraction": fraction,
        "full_training_images": full_count,
        "selected_training_images": selected_count,
        "training_strata": int(full_training_rows.groupby(
            ["station_code", "station_year", "class_idx"], sort=True).ngroups),
        "sampling_policy_version": "nested_stratified_permutation_v1",
        "sampling_seed": int(sampling_seed),
        "sampling_policy": "ceil per station-year/stage stratum, minimum one image per stratum",
    }


def _fresh_output_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Output directory is not empty: {path}; choose a fresh --out-dir"
        )
    return path


def _fold_indices(rows, assignments, fold_id, args):
    indices = split_indices(rows, assignments, fold_id)
    full_train = indices["train"].copy()
    sampling_seed = args.label_seed + fold_id
    local_subset = select_label_subset(
        rows.iloc[full_train], args.label_percentage / 100.0, sampling_seed
    )
    selected_train = full_train[local_subset]
    if not len(selected_train):
        raise ValueError(f"Fold {fold_id} has no selected training labels")
    return indices, full_train, selected_train, sampling_seed


def _dry_run(rows, assignments, args):
    available = sorted(int(value) for value in assignments["fold"].unique())
    fold_ids = available if args.fold_id == 0 else [args.fold_id]
    if any(fold_id not in available for fold_id in fold_ids):
        raise ValueError(f"Requested fold is outside {available}")
    audits = []
    for fold_id in fold_ids:
        indices, full_train, selected_train, sampling_seed = _fold_indices(
            rows, assignments, fold_id, args
        )
        sequences, _ = _prepare_sequences(rows, full_train, selected_train)
        coverage = _label_coverage(
            args, rows.iloc[full_train], rows.iloc[selected_train], sampling_seed
        )
        audits.append({
            "fold": fold_id,
            "train_images": len(full_train),
            "validation_images": len(indices["validation"]),
            "test_images": len(indices["test"]),
            "selected_training_images": len(selected_train),
            "training_sequences": len(sequences),
            "path_positions": sum(len(sequence["indices"]) for sequence in sequences),
            "label_coverage": coverage,
        })
    print(json.dumps({
        "dry_run": True,
        "classes": CLASSES,
        "folds": audits,
        "feature_cache_loaded": False,
        "output_written": False,
    }, indent=2, allow_nan=False), flush=True)


def _localize_sequences(sequences, train_indices, device):
    global_to_local = {int(row_index): local for local, row_index in enumerate(train_indices)}
    localized = []
    for sequence in sequences:
        try:
            local_indices = np.asarray(
                [global_to_local[int(row_index)] for row_index in sequence["indices"]],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("Path sequence includes a row outside the training forward batch") from exc
        localized.append({
            "indices": torch.as_tensor(local_indices, dtype=torch.long, device=device),
            "allowed": sequence["allowed"].to(device=device),
        })
    return localized


def _train_loss(method, all_train_logits, selected_local, selected_hard,
                selected_soft, localized_sequences, path_weight):
    selected_logits = all_train_logits[selected_local]
    path_value = None
    if method == "hard_ce":
        supervised = hard_ce(selected_logits, selected_hard)
        objective = supervised
    elif method == "soft_ce":
        supervised = soft_ce(selected_logits, selected_soft)
        objective = supervised
    elif method == "ordinal":
        supervised = ordinal_loss(selected_logits, selected_soft)
        objective = supervised
    elif method == "transition":
        supervised = soft_ce(selected_logits, selected_soft)
        per_season = [
            monotone_path_loss(all_train_logits[sequence["indices"]], sequence["allowed"])
            for sequence in localized_sequences
        ]
        if not per_season:
            raise ValueError("Transition objective requires at least one anchored training season")
        path_value = torch.stack(per_season).mean()
        objective = supervised + path_weight * path_value
    else:
        raise ValueError(f"Unknown loss method: {method}")
    if not torch.isfinite(objective):
        raise RuntimeError(f"Nonfinite {method} objective")
    if path_value is not None and not torch.isfinite(path_value):
        raise RuntimeError("Nonfinite monotone path loss")
    return objective, supervised, path_value


@torch.no_grad()
def _evaluate(model, features, indices, rows, device):
    model.eval()
    logits = model(features[indices].to(device))
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite evaluation logits")
    probabilities = logits.softmax(dim=-1).cpu()
    hard = rows.iloc[indices]["class_idx"].to_numpy(dtype=np.int64)
    metrics = classification_metrics(hard, probabilities.argmax(dim=-1).numpy())
    return metrics, probabilities


def _train_method(args, method, rows, features, train_indices, selected_indices,
                  validation_indices, test_indices, sequences, fold_id, fold_dir,
                  feature_provenance, label_coverage, source_provenance):
    output = fold_dir / method
    output.mkdir(parents=True)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    global_to_local = {int(row_index): local for local, row_index in enumerate(train_indices)}
    selected_local = torch.as_tensor(
        [global_to_local[int(row_index)] for row_index in selected_indices],
        dtype=torch.long,
        device=args.device,
    )
    selected_rows = rows.iloc[selected_indices]
    selected_hard = torch.as_tensor(
        selected_rows["class_idx"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=args.device,
    )
    selected_soft = torch.as_tensor(
        np.asarray(selected_rows["active_target"].tolist(), dtype=np.float32),
        dtype=torch.float32,
        device=args.device,
    )
    if selected_soft.shape != (len(selected_indices), len(CLASSES)):
        raise ValueError("Selected soft targets do not match the fixed seven-class order")

    # Every arm starts from the same initialized weights and the same dropout RNG.
    model_seed = args.seed + fold_id
    seed_everything(model_seed)
    model = ImageClassifier(
        feature_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=len(CLASSES),
    ).to(args.device)
    model_config = dict(model.config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    seed_everything(model_seed)

    train_features = features[train_indices].to(args.device)
    localized_sequences = _localize_sequences(sequences, train_indices, args.device)
    history = []
    best_f1 = -math.inf
    stale_epochs = 0
    best_path = output / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Exactly one full-batch forward/backward/update per arm and epoch.
        train_logits = model(train_features)
        if not torch.isfinite(train_logits).all():
            raise RuntimeError("Nonfinite training logits")
        objective, supervised_loss, path_loss = _train_loss(
            method,
            train_logits,
            selected_local,
            selected_hard,
            selected_soft,
            localized_sequences,
            args.path_weight,
        )
        objective.backward()
        optimizer.step()

        train_probabilities = train_logits.detach()[selected_local].softmax(-1).cpu()
        train_metrics = classification_metrics(
            selected_hard.detach().cpu().numpy(),
            train_probabilities.argmax(-1).numpy(),
        )
        validation_metrics, validation_probabilities = _evaluate(
            model, features, validation_indices, rows, args.device
        )
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_objective_loss": float(objective.detach().cpu()),
            "train_supervised_loss": float(supervised_loss.detach().cpu()),
            "train_path_loss": (None if path_loss is None
                                else float(path_loss.detach().cpu())),
            "train_selected_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)

        val_f1 = validation_metrics["macro_f1"]
        print(
            f"fold={fold_id} method={method} epoch={epoch} "
            f"train_loss={record['train_objective_loss']:.5f} "
            f"val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1, stale_epochs = val_f1, 0
            checkpoint = {
                "format": "bbch_frozen_loss_classifier_v1",
                "classes": list(CLASSES),
                "model_config": model_config,
                "model": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "method": method,
                "epoch": epoch,
                "val_macro_f1": float(best_f1),
                "args": vars(args).copy(),
                "fold": fold_id,
                "model_seed": model_seed,
                "feature_provenance": feature_provenance,
                "source_provenance": source_provenance,
                "label_coverage": label_coverage,
                "path_weight": float(args.path_weight),
            }
            torch.save(checkpoint, best_path)
            write_json(output / "val_metrics.json", {
                **validation_metrics,
                "selected_epoch": epoch,
                "method": method,
                "label_coverage": label_coverage,
            })
            prediction_table(
                rows.iloc[validation_indices], validation_probabilities, fold_id, method
            ).to_csv(output / "val_predictions.csv", index=False)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != list(CLASSES):
        raise ValueError("Saved checkpoint class order differs from the current class order")
    model.load_state_dict(checkpoint["model"])
    # The held-out test station is forwarded once, after validation selection.
    test_metrics, test_probabilities = _evaluate(
        model, features, test_indices, rows, args.device
    )
    write_json(output / "test_metrics.json", {
        **test_metrics,
        "selected_epoch": checkpoint["epoch"],
        "method": method,
        "label_coverage": label_coverage,
    })
    pd.DataFrame(test_metrics["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        output / "confusion_matrix.csv"
    )
    pd.DataFrame.from_dict(test_metrics["per_class"], orient="index").to_csv(
        output / "per_class_metrics.csv"
    )
    test_predictions = prediction_table(
        rows.iloc[test_indices], test_probabilities, fold_id, method
    )
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    summary = {
        "fold": fold_id,
        "method": method,
        "best_epoch": checkpoint["epoch"],
        "val_macro_f1": checkpoint["val_macro_f1"],
        "requested_label_percentage": label_coverage["requested_percentage"],
        "realized_label_percentage": label_coverage["realized_percentage"],
        "full_training_images": label_coverage["full_training_images"],
        "selected_training_images": label_coverage["selected_training_images"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        **{
            key: value for key, value in test_metrics.items()
            if not isinstance(value, (dict, list))
        },
    }
    return summary, test_predictions


def run_experiment(args, rows, features, assignments, feature_provenance):
    """Run matched loss arms on aligned frozen image features."""
    if (features.ndim != 2 or len(features) != len(rows)
            or not torch.isfinite(features).all()):
        raise ValueError("Feature matrix must be finite [images, feature_dim] and row-aligned")
    class_indices = pd.to_numeric(rows["class_idx"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(class_indices).all()
            or not np.equal(class_indices, np.floor(class_indices)).all()
            or not np.isin(class_indices, np.arange(len(CLASSES))).all()):
        raise ValueError("Metadata class indices must be integers in the fixed seven-class range")
    available = sorted(int(value) for value in assignments["fold"].unique())
    fold_ids = available if args.fold_id == 0 else [args.fold_id]
    if any(fold_id not in available for fold_id in fold_ids):
        raise ValueError(f"Requested fold is outside {available}")
    methods = list(METHODS) if args.methods == ["all"] else list(args.methods)
    output = _fresh_output_directory(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_files = {
        "losses.py": ROOT / "Loss_BBCH" / "losses.py",
        "config.py": ROOT / "Loss_BBCH" / "config.py",
        "run_training.py": ROOT / "Loss_BBCH" / "run_training.py",
        "data.py": ROOT / "Adapter_BBCH" / "data.py",
        "adapter_model.py": ROOT / "Adapter_BBCH" / "adapter_model.py",
        "run_cached_training.py": ROOT / "Adapter_BBCH" / "run_cached_training.py",
        "multiscale_phenology.py": ROOT / "DINOv3_BBCH" / "multiscale_phenology.py",
    }
    source_hashes = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in source_files.items()
    }
    source_provenance = {
        "label_path": str(Path(args.label_path)),
        "label_sha256": file_sha256(args.label_path),
        "data_path": str(Path(args.data_path)),
        "embedding_cache": str(Path(args.embedding_cache)),
        "source_hashes": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
        },
    }
    if args.config:
        source_provenance["config_path"] = str(Path(args.config))
        source_provenance["config_sha256"] = file_sha256(args.config)
    rows.to_csv(output / "image_metadata.csv", index=False)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    write_json(output / "run_config.json", {
        "args": vars(args),
        "classes": list(CLASSES),
        "architecture": {
            "name": "parameter_free_layernorm_mlp_v1",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "layer_norm_affine": False,
        },
        "methods": methods,
        "feature_provenance": feature_provenance,
        "source_provenance": source_provenance,
        "schedule": "one full-batch optimizer update per epoch",
        "checkpoint_selection": "validation macro-F1 over the fixed seven classes",
        "test_evaluations_per_fold_method": 1,
    })

    summaries = []
    prediction_tables = []
    coverage_reports = []
    for fold_id in fold_ids:
        indices, full_train, selected_train, sampling_seed = _fold_indices(
            rows, assignments, fold_id, args
        )
        sequences, sequence_audit = _prepare_sequences(rows, full_train, selected_train)
        coverage = _label_coverage(
            args, rows.iloc[full_train], rows.iloc[selected_train], sampling_seed
        )
        coverage_reports.append(coverage)
        fold_dir = output / f"fold_{fold_id}"
        fold_dir.mkdir()
        split_report = {
            role: {
                "stations": sorted(rows.iloc[role_indices]["station_code"].astype(str).unique()),
                "images": len(role_indices),
            }
            for role, role_indices in indices.items()
        }
        split_report.update({
            "fold": fold_id,
            "label_coverage": coverage,
            "full_training_images": len(full_train),
            "selected_training_images": len(selected_train),
            "train_validation_test_station_disjoint": True,
            "training_sequences": len(sequences),
            "path_positions": int(sum(len(sequence["indices"]) for sequence in sequences)),
        })
        write_json(fold_dir / "split_info.json", split_report)
        rows.iloc[selected_train][[
            "image_key", "station_code", "station_year", "date", "label", "class_idx",
            "active_target",
        ]].to_csv(fold_dir / "selected_training_labels.csv", index=False)
        sequence_audit.to_csv(fold_dir / "sequence_anchor_audit.csv", index=False)

        for method in methods:
            summary, test_table = _train_method(
                args,
                method,
                rows,
                features,
                full_train,
                selected_train,
                indices["validation"],
                indices["test"],
                sequences,
                fold_id,
                fold_dir,
                feature_provenance,
                coverage,
                source_provenance,
            )
            summaries.append(summary)
            prediction_tables.append(test_table)
            pd.DataFrame(summaries).to_csv(output / "all_test_metrics.csv", index=False)

    combined = pd.concat(prediction_tables, ignore_index=True)
    combined.to_csv(output / "combined_test_predictions.csv", index=False)
    aggregate_rows = []
    for method in methods:
        table = combined.loc[combined["method"] == method]
        pooled = classification_metrics(table["true_idx"], table["pred_idx"])
        station_metrics = [
            classification_metrics(group["true_idx"], group["pred_idx"])
            for _, group in table.groupby("station_code", sort=True)
        ]
        aggregate = {
            "method": method,
            "folds": len(fold_ids),
            "complete_loso": len(fold_ids) == len(available),
            **{
                key: value for key, value in pooled.items()
                if not isinstance(value, (dict, list))
            },
        }
        for metric in (
            "accuracy", "macro_f1", "plus_minus_one_accuracy",
            "mean_absolute_stage_error", "quadratic_weighted_kappa",
        ):
            values = np.asarray([result[metric] for result in station_metrics], dtype=float)
            finite_values = values[np.isfinite(values)]
            aggregate[f"mean_station_{metric}"] = (
                float(finite_values.mean()) if len(finite_values) else float("nan")
            )
            if metric in {"accuracy", "macro_f1"}:
                aggregate[f"station_{metric}_std"] = (
                    float(finite_values.std()) if len(finite_values) else float("nan")
                )
        aggregate.update(
            requested_label_percentage=float(args.label_percentage),
            realized_label_percentage_mean=float(np.mean(
                [report["realized_percentage"] for report in coverage_reports]
            )),
            realized_label_percentage_min=float(np.min(
                [report["realized_percentage"] for report in coverage_reports]
            )),
            realized_label_percentage_max=float(np.max(
                [report["realized_percentage"] for report in coverage_reports]
            )),
        )
        aggregate_rows.append(aggregate)
        write_json(output / f"aggregate_{method}.json", {
            **aggregate,
            "per_class": pooled["per_class"],
            "confusion_matrix": pooled["confusion_matrix"],
        })
    pd.DataFrame(aggregate_rows).to_csv(output / "comparison_summary.csv", index=False)
    print(pd.DataFrame(aggregate_rows).to_string(index=False), flush=True)
    return aggregate_rows


def main(argv=None):
    args = parse_args(argv)
    rows, assignments = discover_rows(args)
    fold_count = int(assignments["fold"].max())
    if args.fold_id > fold_count:
        raise ValueError(f"Requested fold {args.fold_id} exceeds {fold_count} LOSO folds")
    print(
        f"In-season images={len(rows)}, stations={rows.station_code.nunique()}, "
        f"station-years={rows.station_year.nunique()}, LOSO folds={fold_count}",
        flush=True,
    )
    print(pd.crosstab(rows["station_code"], rows["label"]).reindex(
        columns=CLASSES, fill_value=0), flush=True)
    if args.dry_run:
        _dry_run(rows, assignments, args)
        return

    output = _fresh_output_directory(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device {args.device!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    features, feature_provenance = load_features(args.embedding_cache, rows, args.stream)
    # run_experiment accepts only an empty output directory; it has already been
    # created here to fail before expensive cache loading on a stale path.
    run_experiment(args, rows, features, assignments, feature_provenance)


if __name__ == "__main__":
    main()

