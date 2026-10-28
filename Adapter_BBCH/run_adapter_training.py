"""Image-only LOSO pilot: transition/random/PCA/learned adapters on frozen DINOv3."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Adapter_BBCH.adapter_model import (
    METHODS, AdapterClassifier, control_basis, select_label_subset, transition_basis,
)
from Adapter_BBCH.config import load_config_defaults
from Adapter_BBCH.data import (
    CLASSES, discover_rows, file_sha256, load_features, split_indices,
)
from DINOv3_BBCH.multiscale_phenology import HybridOrdinalLoss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (np.integer, np.floating)):
            return clean(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def classification_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty split")
    labels = np.arange(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0)
    # Kappa is undefined for a constant exact-match split; report null in JSON.
    kappa = (float("nan") if len(np.unique(np.concatenate([actual, predicted]))) == 1
             else float(cohen_kappa_score(actual, predicted, labels=labels, weights="quadratic")))
    return {"samples": len(actual), "accuracy": float((actual == predicted).mean()),
            "macro_f1": float(f1.mean()),
            "plus_minus_one_accuracy": float((np.abs(actual - predicted) <= 1).mean()),
            "mean_absolute_stage_error": float(np.abs(actual - predicted).mean()),
            "quadratic_weighted_kappa": kappa,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
            "per_class": {name: dict(precision=float(precision[i]), recall=float(recall[i]),
                                      f1=float(f1[i]), support=int(support[i]))
                          for i, name in enumerate(CLASSES)}}


def run_epoch(model, features, targets, class_indices, criterion, device,
              batch_size, optimizer=None, seed=42):
    training = optimizer is not None
    model.train(training)
    dataset = TensorDataset(features, targets, class_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    loss_sum, actual, predicted, probabilities = 0.0, [], [], []
    for x, target, hard_label in loader:
        x, target = x.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss encountered")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        probs = logits.detach().softmax(-1).cpu()
        loss_sum += float(loss.detach()) * len(x)
        actual.extend(hard_label.tolist())
        predicted.extend(probs.argmax(-1).tolist())
        probabilities.append(probs)
    metrics = classification_metrics(actual, predicted)
    metrics["loss"] = loss_sum / len(dataset)
    return metrics, torch.cat(probabilities)


def prediction_table(rows, probabilities, fold, method):
    result = rows[["image_key", "image_path", "station_code", "station_year", "date", "label"]].copy()
    result = result.rename(columns={"label": "true_label"})
    result["true_idx"] = rows["class_idx"].to_numpy()
    result["pred_idx"] = probabilities.argmax(-1).numpy()
    result["pred_label"] = [CLASSES[i] for i in result["pred_idx"]]
    result["fold"], result["method"] = fold, method
    for index, name in enumerate(CLASSES):
        result[f"prob_{name}"] = probabilities[:, index].numpy()
    return result


def train_method(args, method, basis, rows, features, indices, fold_id,
                 fold_dir, basis_report, provenance):
    output = fold_dir / method
    output.mkdir(parents=True, exist_ok=True)
    seed = args.seed + fold_id
    seed_everything(seed)
    model = AdapterClassifier(features.shape[1], len(CLASSES), basis,
                              method, args.hidden_dim, args.dropout).to(args.device)
    targets = torch.tensor(rows["active_target"].tolist(), dtype=torch.float32)
    hard = torch.tensor(rows["class_idx"].to_numpy(), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = HybridOrdinalLoss(power=2, cross_entropy_weight=args.ordinal_ce_weight)
    history, best_f1, stale = [], -math.inf, 0
    # Reset stochastic training after construction so every arm uses the same
    # shuffles and dropout seed despite different adapter initialization costs.
    seed_everything(seed)
    for epoch in range(1, args.epochs + 1):
        metrics = {}
        for role in ("train", "validation"):
            ix = indices[role]
            metrics[role], probs = run_epoch(
                model, features[ix], targets[ix], hard[ix], criterion, args.device,
                args.batch_size, optimizer if role == "train" else None,
                seed=seed + epoch)
            if role == "validation":
                val_probs = probs
        history.append(dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], **{
            f"{role}_{key}": value for role, values in metrics.items()
            for key, value in values.items() if not isinstance(value, (dict, list))}))
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        val_f1 = metrics["validation"]["macro_f1"]
        print(f"fold={fold_id} method={method} epoch={epoch} "
              f"train_f1={metrics['train']['macro_f1']:.4f} val_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            checkpoint = {"format": "bbch_transition_adapter_v1", "classes": CLASSES,
                          "model_config": model.config,
                          "model": {key: val.detach().cpu() for key, val in model.state_dict().items()},
                          "epoch": epoch, "val_macro_f1": best_f1,
                          "args": vars(args), "fold": fold_id,
                          "basis_report": basis_report, "feature_provenance": provenance,
                          "label_sha256": file_sha256(args.label_path),
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            torch.save(checkpoint, output / "best_model.pt")
            write_json(output / "val_metrics.json", metrics["validation"])
            prediction_table(rows.iloc[indices["validation"]], val_probs, fold_id, method).to_csv(
                output / "val_predictions.csv", index=False)
        else:
            stale += 1
        scheduler.step()
        if args.patience and stale >= args.patience:
            break
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # Test is evaluated exactly once, after validation checkpoint selection.
    ix = indices["test"]
    test_metrics, probabilities = run_epoch(
        model, features[ix], targets[ix], hard[ix], criterion, args.device, args.batch_size)
    write_json(output / "test_metrics.json", test_metrics)
    pd.DataFrame(test_metrics["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        output / "confusion_matrix.csv")
    pd.DataFrame.from_dict(test_metrics["per_class"], orient="index").to_csv(
        output / "per_class_metrics.csv")
    predictions = prediction_table(rows.iloc[ix], probabilities, fold_id, method)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    summary = {"fold": fold_id, "method": method, "best_epoch": checkpoint["epoch"],
               "val_macro_f1": checkpoint["val_macro_f1"],
               "rank": int(basis.shape[0]) if method != "none" else 0,
               "trainable_parameters": checkpoint["trainable_parameters"],
               **{key: value for key, value in test_metrics.items()
                  if not isinstance(value, (dict, list))}}
    return summary, predictions


def run_experiment(args, rows, features, assignments, provenance):
    """Also accepts synthetic inputs for remote smoke tests, without a backbone."""
    if features.ndim != 2 or len(features) != len(rows) or not torch.isfinite(features).all():
        raise ValueError("Feature matrix must be finite and aligned to metadata rows")
    available = sorted(assignments["fold"].unique().tolist())
    fold_ids = available if args.fold_id == 0 else [args.fold_id]
    if any(fold not in available for fold in fold_ids):
        raise ValueError(f"Requested fold is outside {available}")
    methods = list(METHODS) if args.methods == ["all"] else args.methods
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty; use a fresh --out-dir for each run")
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "image_metadata.csv", index=False)
    assignments.to_csv(out / "fold_assignments.csv", index=False)
    write_json(out / "run_config.json", {"args": vars(args), "classes": CLASSES,
               "label_sha256": file_sha256(args.label_path), "feature_provenance": provenance})
    summaries, predictions = [], []
    for fold_id in fold_ids:
        indices = split_indices(rows, assignments, fold_id)
        full_train = indices["train"].copy()
        subset = select_label_subset(rows.iloc[full_train], args.label_fraction, args.seed + fold_id)
        indices["train"] = full_train[subset]
        train_rows = rows.iloc[indices["train"]].reset_index(drop=True)
        train_features = features[indices["train"]]
        fold_dir = out / f"fold_{fold_id}"
        fold_dir.mkdir()
        split_report = {role: {"stations": sorted(rows.iloc[ix]["station_code"].unique()),
                               "images": len(ix)} for role, ix in indices.items()}
        split_report.update(full_training_images=len(full_train),
                            realized_label_fraction=len(train_rows) / len(full_train))
        write_json(fold_dir / "split_info.json", split_report)
        train_rows[["image_key", "station_code", "station_year", "class_idx"]].to_csv(
            fold_dir / "selected_training_labels.csv", index=False)
        # Fit the transition arm once. Its effective rank sets the matched
        # budget for every control, never fill a deficient basis with noise.
        basis_report = {"training_stations": sorted(train_rows["station_code"].unique())}
        bases = {}
        effective_rank = min(args.rank, features.shape[1])
        if "transition" in methods:
            bases["transition"], basis_report = transition_basis(
                train_features, train_rows, len(CLASSES), args.rank,
                args.min_stage_images, args.min_transition_stations)
            effective_rank = bases["transition"].shape[0]
        write_json(fold_dir / "basis_report.json", basis_report)
        for method in methods:
            if method not in bases:
                bases[method] = control_basis(method, train_features, effective_rank,
                                              args.seed + fold_id)
        torch.save(bases, fold_dir / "bases.pt")
        for method in methods:
            summary, table = train_method(args, method, bases[method], rows, features,
                indices, fold_id, fold_dir, basis_report, provenance)
            summaries.append(summary)
            predictions.append(table)
            pd.DataFrame(summaries).to_csv(out / "all_test_metrics.csv", index=False)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_csv(out / "combined_test_predictions.csv", index=False)
    comparison = []
    for method in methods:
        table = combined.loc[combined["method"] == method]
        pooled = classification_metrics(table["true_idx"], table["pred_idx"])
        station_metrics = [classification_metrics(group["true_idx"], group["pred_idx"])
                           for _, group in table.groupby("station_code")]
        aggregate = {"method": method, "folds": len(fold_ids),
                     "complete_loso": len(fold_ids) == len(available),
                     **{key: val for key, val in pooled.items() if not isinstance(val, (dict, list))}}
        for metric in ("accuracy", "macro_f1"):
            values = [station[metric] for station in station_metrics]
            aggregate[f"station_{metric}_mean"] = float(np.mean(values))
            aggregate[f"station_{metric}_std"] = float(np.std(values))
            aggregate[f"station_{metric}_min"] = float(np.min(values))
        comparison.append(aggregate)
        write_json(out / f"aggregate_{method}.json", {**aggregate,
                   "per_class": pooled["per_class"], "confusion_matrix": pooled["confusion_matrix"]})
    pd.DataFrame(comparison).to_csv(out / "comparison_summary.csv", index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML training configuration; explicit CLI options override its values")
    parser.add_argument("--label-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--embedding-cache", help="Original frozen DINOv3 cache, never a supervised fold cache")
    parser.add_argument("--out-dir", default="results_adapter_bbch_pilot")
    parser.add_argument("--stream", choices=["micro", "macro"], default="micro")
    parser.add_argument("--camera", choices=["AUTO", "K1", "K2"], default="AUTO")
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--ignore-status-csv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fold-id", type=int, default=1, help="One-based fold; 0 runs complete LOSO")
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--expected-stations", type=int)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=["all", *METHODS], default=["all"])
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--min-stage-images", type=int, default=2)
    parser.add_argument("--min-transition-stations", type=int, default=2)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5, help="Validation patience; 0 disables early stopping")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ordinal-ce-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="Small cached pilot can run on CPU; cuda also supported")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False,
                        help="Discover labels/folds only; no writes or cache load")
    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        parser.set_defaults(**load_config_defaults(parser, config_args.config))
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.rank, args.hidden_dim, args.min_stage_images,
           args.min_transition_stations, args.validation_groups) < 1 or args.fold_id < 0 or args.patience < 0:
        parser.error("Dimensions, epochs, batch size and support thresholds must be positive")
    if not 0 < args.label_fraction <= 1 or not 0 <= args.dropout < 1:
        parser.error("label-fraction must be in (0,1] and dropout in [0,1)")
    if (not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or not 0 <= args.ordinal_ce_weight <= 1 or args.transition_days < 0):
        parser.error("Invalid learning rate, weight decay, CE weight, or transition days")
    if len(set(args.methods)) != len(args.methods) or ("all" in args.methods and args.methods != ["all"]):
        parser.error("Use --methods all alone, or unique explicit method names")
    if not args.dry_run and not args.embedding_cache:
        parser.error("--embedding-cache is required unless --dry-run is used")
    return args


def main(argv=None):
    args = parse_args(argv)
    rows, assignments = discover_rows(args)
    fold_count = int(assignments["fold"].max())
    if args.fold_id > fold_count:
        raise ValueError(f"Requested fold {args.fold_id} exceeds {fold_count} LOSO folds")
    print(f"In-season images={len(rows)}, stations={rows.station_code.nunique()}, "
          f"station-years={rows.station_year.nunique()}, LOSO folds={fold_count}", flush=True)
    print(pd.crosstab(rows["station_code"], rows["label"]).reindex(columns=CLASSES, fill_value=0))
    if args.dry_run:
        return
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    features, provenance = load_features(args.embedding_cache, rows, args.stream)
    run_experiment(args, rows, features, assignments, provenance)


if __name__ == "__main__":
    main()
