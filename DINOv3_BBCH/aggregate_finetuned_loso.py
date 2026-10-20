import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
fold_start = int(sys.argv[2])
fold_end = int(sys.argv[3])
aggregate_dir = root / "aggregate"
aggregate_dir.mkdir(parents=True, exist_ok=True)

metric_rows = []
prediction_frames = []
validation_frames = []
history_frames = []
missing = []
for fold in range(fold_start, fold_end + 1):
    run_dir = root / f"run_fold_{fold}"
    fold_dir = run_dir / f"fold_{fold}"
    metric_path = fold_dir / "test_metrics.json"
    prediction_path = fold_dir / "test_predictions.csv"
    if not metric_path.is_file() or not prediction_path.is_file():
        missing.append(fold)
        continue
    metric_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
    predictions = pd.read_csv(prediction_path)
    predictions["fold"] = fold
    prediction_frames.append(predictions)
    val_path = run_dir / "all_val_metrics.csv"
    if val_path.is_file():
        validation_frames.append(pd.read_csv(val_path))
    history_path = fold_dir / "history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history["fold"] = fold
        history_frames.append(history)

if missing:
    raise SystemExit(f"Cannot aggregate; missing completed folds: {missing}")

metrics = pd.DataFrame(metric_rows).sort_values("fold")
predictions = pd.concat(prediction_frames, ignore_index=True)
metrics.to_csv(aggregate_dir / "all_test_metrics.csv", index=False)
predictions.to_csv(aggregate_dir / "all_test_predictions.csv", index=False)
if validation_frames:
    pd.concat(validation_frames, ignore_index=True).sort_values("fold").to_csv(
        aggregate_dir / "all_val_metrics.csv", index=False
    )
if history_frames:
    pd.concat(history_frames, ignore_index=True).sort_values(["fold", "epoch"]).to_csv(
        aggregate_dir / "all_history.csv", index=False
    )

classes = ["BBCH0", "BBCH1", "BBCH2", "BBCH3", "BBCH5", "BBCH6_7", "BBCH8"]


def evaluate_prediction_columns(frame, pred_idx_col, pred_label_col, date_score_col):
    true_indices = pd.to_numeric(frame["true_idx"], errors="raise").astype(int)
    pred_indices = pd.to_numeric(frame[pred_idx_col], errors="raise").astype(int)
    true_labels = frame["true_label"].astype(str)
    pred_labels = frame[pred_label_col].astype(str)

    confusion = pd.crosstab(true_labels, pred_labels)
    confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)

    per_class = []
    for label in classes:
        true_is_label = true_labels == label
        pred_is_label = pred_labels == label
        tp = int((true_is_label & pred_is_label).sum())
        fp = int((~true_is_label & pred_is_label).sum())
        fn = int((true_is_label & ~pred_is_label).sum())
        support = int(true_is_label.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    per_class_frame = pd.DataFrame(per_class)
    distance = (true_indices - pred_indices).abs()
    date_scores = pd.to_numeric(frame[date_score_col], errors="coerce")
    summary = {
        "folds": len(metrics),
        "samples": len(frame),
        "accuracy": float((true_indices == pred_indices).mean()),
        "plus_minus_1_accuracy": float((distance <= 1).mean()),
        "mean_absolute_stage_error": float(distance.mean()),
        "macro_f1": float(per_class_frame.f1.mean()),
        "date_window_accuracy": float(date_scores.mean()),
    }
    return confusion, per_class_frame, summary


confusion, per_class_frame, summary = evaluate_prediction_columns(
    predictions,
    pred_idx_col="pred_idx",
    pred_label_col="pred_label",
    date_score_col="date_window_score",
)
confusion.to_csv(aggregate_dir / "all_test_confusion_matrix.csv")
per_class_frame.to_csv(aggregate_dir / "all_test_per_class_metrics.csv", index=False)
summary.update(
    {
        "fold_accuracy_mean": float(metrics.test_accuracy.mean()),
        "fold_accuracy_std": float(metrics.test_accuracy.std(ddof=0)),
        "fold_macro_f1_mean": float(metrics.test_macro_f1.mean()),
        "fold_macro_f1_std": float(metrics.test_macro_f1.std(ddof=0)),
    }
)
(aggregate_dir / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("Raw aggregate:")
print(json.dumps(summary, indent=2))

decoded_columns = {
    "decoded_pred_idx",
    "decoded_pred_label",
    "decoded_date_window_score",
}
present_decoded_columns = decoded_columns.intersection(predictions.columns)
if present_decoded_columns and present_decoded_columns != decoded_columns:
    missing_decoded_columns = sorted(decoded_columns - present_decoded_columns)
    raise SystemExit(
        "Cannot aggregate monotonic predictions; missing decoded columns: "
        f"{missing_decoded_columns}"
    )

if decoded_columns.issubset(predictions.columns):
    monotonic_predictions = predictions.copy()
    monotonic_predictions["raw_pred_idx"] = monotonic_predictions["pred_idx"]
    monotonic_predictions["raw_pred_label"] = monotonic_predictions["pred_label"]
    monotonic_predictions["raw_date_window_score"] = monotonic_predictions[
        "date_window_score"
    ]
    monotonic_predictions["pred_idx"] = monotonic_predictions["decoded_pred_idx"]
    monotonic_predictions["pred_label"] = monotonic_predictions["decoded_pred_label"]
    monotonic_predictions["date_window_score"] = monotonic_predictions[
        "decoded_date_window_score"
    ]
    monotonic_predictions.to_csv(
        aggregate_dir / "all_test_monotonic_predictions.csv",
        index=False,
    )

    monotonic_confusion, monotonic_per_class, monotonic_summary = (
        evaluate_prediction_columns(
            monotonic_predictions,
            pred_idx_col="pred_idx",
            pred_label_col="pred_label",
            date_score_col="date_window_score",
        )
    )
    monotonic_confusion.to_csv(
        aggregate_dir / "all_test_monotonic_confusion_matrix.csv"
    )
    monotonic_per_class.to_csv(
        aggregate_dir / "all_test_monotonic_per_class_metrics.csv",
        index=False,
    )

    if "test_monotonic_accuracy" in metrics:
        monotonic_summary["fold_accuracy_mean"] = float(
            metrics.test_monotonic_accuracy.mean()
        )
        monotonic_summary["fold_accuracy_std"] = float(
            metrics.test_monotonic_accuracy.std(ddof=0)
        )
    if "test_monotonic_macro_f1" in metrics:
        monotonic_summary["fold_macro_f1_mean"] = float(
            metrics.test_monotonic_macro_f1.mean()
        )
        monotonic_summary["fold_macro_f1_std"] = float(
            metrics.test_monotonic_macro_f1.std(ddof=0)
        )
    monotonic_summary["accuracy_delta_vs_raw"] = (
        monotonic_summary["accuracy"] - summary["accuracy"]
    )
    monotonic_summary["macro_f1_delta_vs_raw"] = (
        monotonic_summary["macro_f1"] - summary["macro_f1"]
    )
    monotonic_summary["date_window_accuracy_delta_vs_raw"] = (
        monotonic_summary["date_window_accuracy"]
        - summary["date_window_accuracy"]
    )
    (aggregate_dir / "monotonic_summary.json").write_text(
        json.dumps(monotonic_summary, indent=2),
        encoding="utf-8",
    )
    print("Monotonic aggregate:")
    print(json.dumps(monotonic_summary, indent=2))
else:
    print(
        "No decoded prediction columns found; monotonic aggregate was not generated."
    )

print(f"Combined outputs: {aggregate_dir}")
