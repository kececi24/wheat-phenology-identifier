"""Single-image DINOv3 LOSO: frozen, Q/V LoRA, standard and transition adapters."""

from __future__ import annotations

import gc
import math
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from Adapter_BBCH.adapter_model import select_label_subset, transition_basis
from Adapter_BBCH.block_config import parse_args
from Adapter_BBCH.block_model import (
    CHECKPOINT_FORMAT, METHODS, BlockAdapterClassifier, capture_layer_inputs,
    make_extractor, selected_blocks,
)
from Adapter_BBCH.data import CLASSES, discover_rows, file_sha256, split_indices
from Adapter_BBCH.image_data import make_loader
from Adapter_BBCH.run_cached_training import (
    LABEL_SAMPLING_POLICY, classification_metrics, label_coverage_report,
    prediction_table, seed_everything, write_json,
)
from DINOv3_BBCH.multiscale_phenology import HybridOrdinalLoss


def autocast(args):
    if args.amp == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)


def preprocessing(args, extractor):
    processor = extractor.processor
    return {"image_size": args.image_size, "tile_size": args.tile_size,
            "tile_stride": args.tile_stride, "max_tiles": args.max_tiles,
            "augment": args.augment,
            "mean": list(processor.image_mean), "std": list(processor.image_std),
            "tile_policy": "deterministic_evenly_spaced_edge_covering_v1"}


def software_provenance():
    import transformers
    sources = ("run_adapter_training.py", "block_config.py", "block_model.py", "image_data.py",
               "adapter_model.py", "data.py", "config.py", "run_cached_training.py")
    folder = Path(__file__).resolve().parent
    return {"torch_version": str(torch.__version__), "transformers_version": str(transformers.__version__),
            "source_sha256": {name: file_sha256(folder / name) for name in sources},
            "shared_dinov3_sha256": file_sha256(ROOT / "DINOv3_BBCH" / "multiscale_phenology.py")}


@torch.no_grad()
def estimate_bases(args, extractor, train_rows, settings):
    """Selected training images only, in the exact adapter-input coordinates."""
    extractor.eval().requires_grad_(False)
    _, indices = selected_blocks(extractor, args.last_blocks)
    features = {index: [] for index in indices}
    loader = make_loader(train_rows, settings, args, train=False)
    with capture_layer_inputs(extractor, args.last_blocks) as captured:
        for batch_number, batch in enumerate(loader, 1):
            for tiles, mask in zip(batch["tiles"], batch["tile_mask"]):
                sums = {index: [] for index in indices}
                for pixels in tiles[mask].split(args.tile_batch_size):
                    captured.clear()
                    extractor.backbone(pixel_values=pixels.to(args.device))
                    for index in indices:
                        z = captured[index].float()
                        # CLS plus spatial patches; registers excluded.
                        z = torch.cat([z[:, :1], z[:, 1 + extractor.num_register_tokens:]], dim=1)
                        sums[index].append(z.mean(1).cpu())
                for index in indices:
                    features[index].append(torch.cat(sums[index]).mean(0))
            if batch_number % 50 == 0:
                print(f"basis images={min(batch_number * args.batch_size, len(train_rows))}"
                      f"/{len(train_rows)}", flush=True)
    bases, reports = {}, {}
    for index in indices:
        bases[index], reports[index] = transition_basis(
            torch.stack(features[index]), train_rows, len(CLASSES), args.rank,
            args.min_stage_images, args.min_transition_stations, normalize=False)
        if bases[index].shape[0] != args.rank:
            raise ValueError(f"Layer {index}: supported rank is {bases[index].shape[0]}, requested {args.rank}. "
                             "Choose a smaller rank and matched LoRA rank; unsupported directions "
                             "are never filled with random noise.")
    return bases, {"coordinate_system": "pretrained_block_norm2_output_no_extra_normalization",
                   "image_summary": "mean_CLS_and_patch_tokens_then_mean_tiles", "layers": reports}


def run_epoch(model, loader, rows, criterion, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    n = len(rows)
    probabilities = torch.empty(n, len(CLASSES))
    seen = torch.zeros(n, dtype=torch.bool)
    totals = {"loss": 0.0, "classification_loss": 0.0, "prior_penalty": 0.0}
    for step, batch in enumerate(loader):
        size = len(batch["target"])
        # Sample-weighted accumulation, including a short final batch/group.
        group_start = (step // args.accumulation_steps) * args.accumulation_steps
        group_samples = min(args.accumulation_steps * args.batch_size,
                            n - group_start * args.batch_size)
        with torch.set_grad_enabled(training), autocast(args):
            logits = model(batch["tiles"].to(args.device, non_blocking=True),
                           batch["tile_mask"].to(args.device, non_blocking=True))
            classification_loss = criterion(logits, batch["target"].to(args.device))
            penalty = model.prior_penalty()
            loss = classification_loss + args.prior_lambda * penalty
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss encountered")
        if training:
            scaler.scale(loss * size / group_samples).backward()
            if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        ix = batch["index"]
        if seen[ix].any():
            raise RuntimeError("Duplicate image during epoch")
        seen[ix] = True
        probabilities[ix] = logits.detach().float().softmax(-1).cpu()
        for key, value in (("loss", loss), ("classification_loss", classification_loss),
                           ("prior_penalty", penalty)):
            totals[key] += float(value.detach()) * size
    if not seen.all():
        raise RuntimeError("An epoch did not evaluate every requested image")
    metrics = classification_metrics(rows["class_idx"], probabilities.argmax(-1).numpy())
    metrics.update({key: value / n for key, value in totals.items()})
    return metrics, probabilities


def train_method(args, method, bases, rows, indices, fold_id, fold_dir,
                 settings, backbone, basis_report, coverage, provenance):
    output = fold_dir / method
    output.mkdir()
    seed = args.seed + fold_id
    seed_everything(seed)
    extractor = make_extractor(backbone["name"], backbone["revision"])
    seed_everything(seed)  # head/pool initialization is independent of HF loading internals
    model = BlockAdapterClassifier(extractor, method=method, last_blocks=args.last_blocks,
        rank=args.rank, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        hidden_dim=args.hidden_dim, dropout=args.dropout, dense_grid_size=args.dense_grid_size,
        tile_batch_size=args.tile_batch_size, bases=bases).to(args.device)
    counts = model.parameter_counts()
    print(f"fold={fold_id} method={method} parameters={counts}", flush=True)
    role_rows = {role: rows.iloc[ix].reset_index(drop=True) for role, ix in indices.items()}
    val_loader = make_loader(role_rows["validation"], settings, args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    criterion = HybridOrdinalLoss(power=2, cross_entropy_weight=args.ordinal_ce_weight)
    best, stale, history = -math.inf, 0, []
    seed_everything(seed)
    for epoch in range(1, args.epochs + 1):
        loader = make_loader(role_rows["train"], settings, args, train=True, seed=seed + epoch)
        train_metrics, _ = run_epoch(model, loader, role_rows["train"], criterion, args, optimizer, scaler)
        val_metrics, val_probs = run_epoch(model, val_loader, role_rows["validation"], criterion, args)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **{
            f"{role}_{key}": value for role, metrics in (("train", train_metrics), ("validation", val_metrics))
            for key, value in metrics.items() if not isinstance(value, (dict, list))}})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"fold={fold_id} method={method} epoch={epoch} "
              f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best:
            best, stale = val_metrics["macro_f1"], 0
            checkpoint = {"format": CHECKPOINT_FORMAT, "classes": CLASSES,
                          "model_config": model.config, "model": model.adaptation_state(),
                          "backbone": backbone, "preprocessing": settings,
                          "epoch": epoch, "val_macro_f1": best, "fold": int(fold_id),
                          "args": vars(args), "basis_report": basis_report,
                          "software_provenance": provenance,
                          "label_coverage": coverage, "label_sha256": file_sha256(args.label_path), **counts}
            torch.save(checkpoint, output / "best_model.pt")
            write_json(output / "val_metrics.json", {**val_metrics, "label_coverage": coverage})
            prediction_table(role_rows["validation"], val_probs, fold_id, method).to_csv(
                output / "val_predictions.csv", index=False)
        else:
            stale += 1
        scheduler.step()
        if args.patience and stale >= args.patience:
            break
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_adaptation_state(checkpoint["model"])
    # Complete held-out station once, after validation checkpoint selection.
    test_rows = role_rows["test"]
    test_metrics, test_probs = run_epoch(model, make_loader(test_rows, settings, args),
                                        test_rows, criterion, args)
    write_json(output / "test_metrics.json", {**test_metrics, "label_coverage": coverage, **counts})
    pd.DataFrame(test_metrics["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        output / "confusion_matrix.csv")
    pd.DataFrame.from_dict(test_metrics["per_class"], orient="index").to_csv(output / "per_class_metrics.csv")
    predictions = prediction_table(test_rows, test_probs, fold_id, method)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    summary = {"fold": int(fold_id), "method": method, "best_epoch": checkpoint["epoch"],
               "val_macro_f1": best, "rank": args.rank if method in {"standard", "transition"}
               else args.lora_rank if method == "lora" else 0,
               "requested_label_percentage": coverage["requested_percentage"],
               "realized_label_percentage": coverage["realized_percentage"],
               "full_training_images": coverage["full_training_images"],
               "selected_training_images": coverage["selected_training_images"], **counts,
               **{k: v for k, v in test_metrics.items() if not isinstance(v, (dict, list))}}
    del model, extractor, optimizer, scheduler, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, predictions


def aggregate_results(out, summaries, predictions, available):
    table = pd.concat(predictions, ignore_index=True)
    table.to_csv(out / "combined_test_predictions.csv", index=False)
    comparison = []
    for method, group in table.groupby("method", sort=False):
        if group["image_key"].duplicated().any():
            raise RuntimeError("LOSO aggregate contains duplicate test images")
        metrics = classification_metrics(group["true_idx"], group["pred_idx"])
        folds = [s for s in summaries if s["method"] == method]
        report = {"method": method, "folds": len(folds), "complete_loso": len(folds) == len(available),
                  "requested_label_percentage": folds[0]["requested_label_percentage"],
                  "realized_label_percentage_mean": float(np.mean([s["realized_label_percentage"] for s in folds])),
                  "full_training_images_across_folds": sum(s["full_training_images"] for s in folds),
                  "selected_training_images_across_folds": sum(s["selected_training_images"] for s in folds),
                  **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}
        percentages = [s["realized_label_percentage"] for s in folds]
        report.update(realized_label_percentage_min=min(percentages),
                      realized_label_percentage_max=max(percentages),
                      realized_label_percentage_weighted=(100 * report["selected_training_images_across_folds"]
                                                          / report["full_training_images_across_folds"]))
        for key in ("adapter_parameters", "head_pool_parameters", "trainable_parameters"):
            report[key] = folds[0][key]
        for name in ("accuracy", "macro_f1"):
            scores = [classification_metrics(g["true_idx"], g["pred_idx"])[name]
                      for _, g in group.groupby("station_code")]
            report.update({f"station_{name}_mean": float(np.mean(scores)),
                           f"station_{name}_std": float(np.std(scores)),
                           f"station_{name}_min": float(np.min(scores))})
        comparison.append(report)
        write_json(out / f"aggregate_{method}.json", {**report, "per_class": metrics["per_class"],
                   "confusion_matrix": metrics["confusion_matrix"]})
    pd.DataFrame(comparison).to_csv(out / "comparison_summary.csv", index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


def run_experiment(args, rows, assignments):
    available = sorted(assignments["fold"].unique().tolist())
    folds = available if args.fold_id == 0 else [args.fold_id]
    if any(f not in available for f in folds):
        raise ValueError(f"Requested fold outside {available}")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing experiments are never overwritten")
    methods = list(METHODS) if args.methods == ["all"] else args.methods
    extractor = make_extractor(args.image_backbone, args.backbone_revision)
    backbone = {"name": args.image_backbone,
                "revision": getattr(extractor.backbone.config, "_commit_hash", None)}
    if not backbone["revision"]:
        raise ValueError("Could not resolve the public pretrained backbone commit")
    settings = preprocessing(args, extractor)
    if args.image_size % extractor.patch_size:
        raise ValueError("image_size must be divisible by the backbone patch size")
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "image_metadata.csv", index=False)
    assignments.to_csv(out / "fold_assignments.csv", index=False)
    provenance = software_provenance()
    if args.rank != 2 * args.lora_rank:
        print("Adapter parameter budgets differ: set rank = 2 * lora_rank to match Q/V LoRA.", flush=True)
    write_json(out / "run_config.json", {"format": CHECKPOINT_FORMAT, "args": vars(args),
               "classes": CLASSES, "backbone": backbone, "preprocessing": settings,
               "label_sha256": file_sha256(args.label_path), "sampling_policy": LABEL_SAMPLING_POLICY,
               "software_provenance": provenance,
               "adapter_budget_matched": args.rank == 2 * args.lora_rank,
               "basis_uses_selected_training_labels_only": True})
    summaries, predictions = [], []
    extractor.eval().requires_grad_(False)
    for fold_id in folds:
        indices = split_indices(rows, assignments, fold_id)
        full_train = indices["train"].copy()
        sampling_seed = args.label_seed + fold_id
        selected = select_label_subset(rows.iloc[full_train], args.label_fraction, sampling_seed)
        indices["train"] = full_train[selected]
        train_rows = rows.iloc[indices["train"]].reset_index(drop=True)
        coverage = label_coverage_report(args, rows.iloc[full_train], train_rows, sampling_seed)
        fold_dir = out / f"fold_{fold_id}"
        fold_dir.mkdir()
        write_json(fold_dir / "split_info.json", {
            **{role: {"stations": sorted(rows.iloc[ix].station_code.unique()), "images": len(ix)}
               for role, ix in indices.items()}, "label_coverage": coverage})
        train_rows[["image_key", "station_code", "station_year", "class_idx"]].to_csv(
            fold_dir / "selected_training_labels.csv", index=False)
        bases, basis_report = {}, {"not_requested": True}
        if "transition" in methods:
            extractor.to(args.device)
            bases, basis_report = estimate_bases(args, extractor, train_rows, settings)
        basis_report["label_coverage"] = coverage
        write_json(fold_dir / "basis_report.json", basis_report)
        torch.save(bases, fold_dir / "bases.pt")
        extractor.cpu()
        for method in methods:
            summary, table = train_method(args, method, bases, rows, indices, fold_id,
                                          fold_dir, settings, backbone, basis_report, coverage, provenance)
            summaries.append(summary)
            predictions.append(table)
            pd.DataFrame(summaries).to_csv(out / "all_test_metrics.csv", index=False)
    aggregate_results(out, summaries, predictions, available)


def main(argv=None):
    args = parse_args(argv)
    rows, assignments = discover_rows(args)
    if args.fold_id > int(assignments["fold"].max()):
        raise ValueError("Requested fold exceeds available LOSO folds")
    print(f"Images={len(rows)} stations={rows.station_code.nunique()} folds={assignments.fold.nunique()}", flush=True)
    print(pd.crosstab(rows.station_code, rows.label).reindex(columns=CLASSES, fill_value=0))
    if args.dry_run:
        return
    if torch.device(args.device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # CUDA without an explicit index means the current/default device.
        if torch.device(args.device).index is not None:
            torch.cuda.set_device(torch.device(args.device))
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bf16; choose fp16 or none")
    run_experiment(args, rows, assignments)


if __name__ == "__main__":
    main()
