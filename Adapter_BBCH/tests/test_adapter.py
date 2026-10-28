"""Remote tests: no pretrained weights, network, real images, or GPU required."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from Adapter_BBCH.adapter_model import (
    AdapterClassifier, control_basis, normalized_features, select_label_subset, transition_basis,
)
from Adapter_BBCH.data import CLASSES, image_key, load_features, pool_feature, split_indices
from Adapter_BBCH.run_adapter_training import parse_args, run_experiment


def synthetic_data():
    """Nonlinear seven-stage trajectories plus independent station offsets."""
    generator = torch.Generator().manual_seed(71)
    rows, features = [], []
    directions = torch.randn(7, 12, generator=generator)
    for station in range(4):
        offset = torch.randn(12, generator=generator) * 0.3
        for stage in range(7):
            for image in range(3):
                features.append(directions[stage] + offset +
                                torch.randn(12, generator=generator) * 0.02)
                key = f"{station:02d}.01/2020/K1/10X/{stage}_{image}.jpeg"
                target = [0.0] * 7
                target[stage] = 1.0
                rows.append(dict(station_code=f"{station:02d}.01",
                    station_year=f"{station:02d}.01_2020", class_idx=stage,
                    date=f"2020-0{stage + 1}-0{image + 1}", label=CLASSES[stage],
                    image_key=key, image_path=f"data/{key}", active_target=target))
    return pd.DataFrame(rows), torch.stack(features)


def assignments_for(rows):
    stations = sorted(rows.station_code.unique())
    assignments = []
    for fold in range(4):
        for position, station in enumerate(stations):
            role = ("test" if position == fold else
                    "validation" if position == (fold + 1) % 4 else "train")
            assignments.append(dict(fold=fold + 1, role=role, station_code=station))
    return pd.DataFrame(assignments)


class BasisTests(unittest.TestCase):
    def test_balances_station_years_and_stations(self):
        values = [torch.tensor([3., 0., 1., 0.]), torch.tensor([0., 3., 1., 0.]),
                  torch.tensor([1., 0., 0., 3.]), torch.tensor([0., 1., 3., 0.])]
        rows, features = [], []
        for station, year, count, pair in (
                ("A", "A_1", 2, values[:2]), ("A", "A_2", 2, values[:2]),
                ("B", "B_1", 12, values[2:])):
            for stage, vector in enumerate(pair):
                for _ in range(count):
                    rows.append(dict(station_code=station, station_year=year, class_idx=stage))
                    features.append(vector)
        basis, report = transition_basis(torch.stack(features), pd.DataFrame(rows), 2,
                                         rank=1, min_stage_images=2, min_stations=2)
        z = normalized_features(torch.stack(values))
        expected = ((z[1] - z[0]) + (z[3] - z[2])) / 2
        expected = expected / expected.norm()
        self.assertAlmostEqual(abs(float(basis[0] @ expected)), 1.0, places=5)
        self.assertEqual(report["boundaries"][0]["station_count"], 2)

    def test_missing_adjacent_stage_is_not_bridged(self):
        rows, x = synthetic_data()
        rows = rows.loc[rows.class_idx != 1]
        basis, report = transition_basis(x[rows.index], rows, 7, min_stations=2)
        self.assertFalse(report["boundaries"][0]["supported"])
        self.assertFalse(report["boundaries"][1]["supported"])
        self.assertLessEqual(len(basis), 4)
        self.assertTrue(torch.allclose(basis @ basis.T, torch.eye(len(basis)), atol=1e-5))

    def test_missing_support_fails_instead_of_using_held_out_rows(self):
        rows, x = synthetic_data()
        selected = rows.station_code == "00.01"
        with self.assertRaisesRegex(ValueError, "No supported"):
            transition_basis(x[selected.to_numpy()], rows.loc[selected], 7, min_stations=2)

    def test_label_subset_is_reproducible_and_keeps_groups(self):
        rows, _ = synthetic_data()
        first = select_label_subset(rows, 0.25, 42)
        self.assertTrue(np.array_equal(first, select_label_subset(rows, 0.25, 42)))
        selected = rows.iloc[first]
        self.assertEqual(len(selected), rows.groupby(["station_code", "class_idx"]).ngroups)
        self.assertLess(len(first), len(rows))

    def test_zero_initialized_residual_and_frozen_basis(self):
        _, x = synthetic_data()
        basis = control_basis("random", x, 3, 42)
        torch.manual_seed(1)
        adapted = AdapterClassifier(12, 7, basis, "transition", dropout=0)
        torch.manual_seed(1)
        baseline = AdapterClassifier(12, 7, basis, "none", dropout=0)
        self.assertTrue(torch.equal(adapted(x), baseline(x)))
        adapted(x).square().mean().backward()
        self.assertIsNone(adapted.basis.grad)
        self.assertGreater(float(adapted.up.weight.grad.abs().sum()), 0)
        self.assertNotIn("basis", dict(adapted.named_parameters()))

    def test_learned_basis_receives_gradients_after_first_update(self):
        _, x = synthetic_data()
        basis = control_basis("learned", x, 3, 42)
        model = AdapterClassifier(12, 7, basis, "learned", dropout=0)
        with torch.no_grad():
            model.up.weight.fill_(0.1)
        model(x).square().mean().backward()
        self.assertGreater(float(model.basis.grad.abs().sum()), 0)

    def test_checkpoint_reconstruction(self):
        _, x = synthetic_data()
        for method in ("transition", "random", "pca", "learned", "none"):
            basis = control_basis("random", x, 3, 42)
            model = AdapterClassifier(12, 7, basis, method).eval()
            restored = AdapterClassifier.from_checkpoint(
                {"model_config": model.config, "model": model.state_dict()}).eval()
            self.assertTrue(torch.equal(model(x), restored(x)))


class DataAndRunnerTests(unittest.TestCase):
    def test_mean_pooling_and_portable_paths(self):
        value = torch.arange(24).reshape(2, 3, 4).float()
        self.assertTrue(torch.equal(pool_feature(value), value.mean((0, 1))))
        self.assertEqual(image_key("/home/person/data/02.03/2015/K1/10X/a.jpeg"),
                         image_key("C:\\project\\data\\02.03\\2015\\K1\\10X\\a.jpeg"))

    def test_missing_or_supervised_cache_is_rejected(self):
        rows = pd.DataFrame({"image_key": ["02.03/2015/K1/10X/a.jpeg"]})
        frozen = dict(image_backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
                      pretrained=True, cache_format="dinov3_dense_v1", feature_dim=4, micro={})
        with patch("Adapter_BBCH.data.torch.load", return_value=frozen):
            with self.assertRaisesRegex(ValueError, "missing 1/1"):
                load_features("unused.pt", rows, "micro")
        with patch("Adapter_BBCH.data.torch.load", return_value={**frozen, "image_backbone": "/fold/backbone"}):
            with self.assertRaisesRegex(ValueError, "fold-safe"):
                load_features("unused.pt", rows, "micro")

    def test_overlap_is_rejected(self):
        rows, _ = synthetic_data()
        assignments = assignments_for(rows)
        split = split_indices(rows, assignments, 1)
        self.assertEqual(len(split["train"]), 42)
        duplicate = pd.DataFrame([dict(fold=1, role="train", station_code="00.01")])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            split_indices(rows, pd.concat([assignments, duplicate]), 1)

    def test_cli_rejects_invalid_budget(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--dry-run", "--label-fraction", "0"])

    def test_training_smoke_basis_scope_and_selected_checkpoint(self):
        rows, x = synthetic_data()
        assignments = assignments_for(rows)
        with tempfile.TemporaryDirectory() as folder:
            label_path = Path(folder) / "labels.csv"
            label_path.write_text("synthetic fixture only", encoding="utf-8")
            args = parse_args(["--label-path", str(label_path), "--embedding-cache", "synthetic",
                               "--out-dir", str(Path(folder) / "run"), "--epochs", "2",
                               "--rank", "3", "--hidden-dim", "8", "--dropout", "0"])
            comparison = run_experiment(args, rows, x, assignments, {"synthetic": True})
            self.assertEqual(len(comparison), 5)
            fold = Path(args.out_dir) / "fold_1"
            report = json.loads((fold / "basis_report.json").read_text())
            self.assertEqual(report["training_stations"], ["02.01", "03.01"])
            self.assertEqual({entry["rank"] for entry in comparison if entry["method"] != "none"}, {3})
            for method in ("none", "transition", "random", "pca", "learned"):
                directory = fold / method
                checkpoint = torch.load(directory / "best_model.pt", weights_only=True)
                model = AdapterClassifier.from_checkpoint(checkpoint).eval()
                test_ix = split_indices(rows, assignments, 1)["test"]
                with torch.no_grad():
                    expected = model(x[test_ix]).softmax(-1).numpy()
                saved = pd.read_csv(directory / "test_predictions.csv")
                self.assertTrue(np.allclose(saved[[f"prob_{name}" for name in CLASSES]],
                                            expected, atol=1e-6))
                self.assertFalse(comparison[0]["complete_loso"])


if __name__ == "__main__":
    unittest.main()
