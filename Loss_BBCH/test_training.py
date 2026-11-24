"""Small remote-only integration test using synthetic frozen features."""
import json

import pandas as pd
import torch

from Adapter_BBCH.data import CLASSES
from Loss_BBCH.config import parse_args
from Loss_BBCH.run_training import run_experiment


def test_four_arms_checkpoint_and_heldout_outputs(tmp_path):
    label_file = tmp_path / "labels.csv"
    label_file.write_text("synthetic fixture\n", encoding="utf-8")
    output = tmp_path / "run"
    args = parse_args([
        "--label-path", str(label_file), "--out-dir", str(output),
        "--fold-id", "1", "--epochs", "2", "--hidden-dim", "4",
        "--dropout", "0", "--label-percentage", "50",
    ])
    records = []
    for station in ["A", "B", "C"]:
        for day in range(14):
            stage = day // 2
            records.append(dict(
                image_key=f"{station}/{day}", image_path=f"{station}/{day}.jpeg",
                station_code=station, station_year=f"{station}_2020",
                date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                label=CLASSES[stage], class_idx=stage,
                active_target=[float(k == stage) for k in range(7)],
            ))
    rows = pd.DataFrame(records)
    assignments = pd.DataFrame([
        dict(fold=1, station_code="A", role="train"),
        dict(fold=1, station_code="B", role="validation"),
        dict(fold=1, station_code="C", role="test"),
    ])
    features = torch.randn(42, 8, generator=torch.Generator().manual_seed(1))
    results = run_experiment(args, rows, features, assignments, {"fixture": True})
    assert len(results) == 4
    predictions = pd.read_csv(output / "combined_test_predictions.csv")
    assert set(predictions.station_code) == {"C"}
    assert predictions.groupby("method").size().tolist() == [14] * 4
    audit = pd.read_csv(output / "fold_1" / "sequence_anchor_audit.csv")
    assert set(audit.station_code) == {"A"}
    assert audit.selected_label_anchor.sum() == 7
    for method in args.methods:
        directory = output / "fold_1" / method
        checkpoint = torch.load(directory / "best_model.pt", weights_only=True)
        assert checkpoint["classes"] == CLASSES
        metrics = json.loads((directory / "test_metrics.json").read_text())
        assert metrics["samples"] == 14
        history = pd.read_csv(directory / "history.csv")
        assert checkpoint["epoch"] == int(history.loc[history.validation_macro_f1.idxmax(), "epoch"])
