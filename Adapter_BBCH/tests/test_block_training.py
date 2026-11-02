"""Remote-only end-to-end exercise of raw-image training and saved artifacts."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from Adapter_BBCH.block_config import parse_args
from Adapter_BBCH.block_model import DenseExtractor, METHODS
from Adapter_BBCH.data import CLASSES
from Adapter_BBCH.run_adapter_training import run_experiment


class BlockTrainingTests(unittest.TestCase):
    def test_four_arms_budget_selection_and_checkpoint_artifacts(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                self.exercise(Path(folder))
        finally:
            torch.set_num_threads(previous_threads)

    def exercise(self, root):
        rng = np.random.default_rng(11)
        rows = []
        for station in range(4):
            for stage in range(7):
                for day in range(2):
                    path = root / f"{station}_{stage}_{day}.png"
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                    rows.append({"image_path": str(path), "image_key": path.name,
                                 "station_code": str(station), "station_year": f"{station}_2020",
                                 "date": f"2020-0{stage + 1}-0{day + 1}", "label": CLASSES[stage],
                                 "class_idx": stage, "active_target": [float(i == stage) for i in range(7)]})
        rows = pd.DataFrame(rows)
        assignments = pd.DataFrame([{"fold": 1, "station_code": str(i),
                                      "role": "train" if i < 2 else "validation" if i == 2 else "test"}
                                     for i in range(4)])
        label_file = root / "labels.csv"
        label_file.write_text("synthetic labels for provenance", encoding="utf-8")
        output = root / "output"
        args = parse_args(["--device", "cpu", "--num-workers", "0", "--epochs", "2",
                           "--batch-size", "3", "--accumulation-steps", "2", "--patience", "0",
                           "--image-size", "32", "--tile-size", "32", "--tile-stride", "32",
                           "--tile-batch-size", "2", "--max-tiles", "1", "--rank", "4",
                           "--lora-rank", "2", "--hidden-dim", "16", "--min-stage-images", "1",
                           "--label-percentage", "25", "--label-path", str(label_file),
                           "--out-dir", str(output)])
        config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, image_size=32, patch_size=16, num_register_tokens=2,
            drop_path_rate=0., attention_dropout=0.)
        config._commit_hash = "synthetic-no-network"
        base = DenseExtractor(DINOv3ViTModel(config),
                              SimpleNamespace(image_mean=[.5] * 3, image_std=[.5] * 3))
        with patch("Adapter_BBCH.run_adapter_training.make_extractor",
                   side_effect=lambda *a: copy.deepcopy(base)):
            run_experiment(args, rows, assignments)
        selected = pd.read_csv(output / "fold_1" / "selected_training_labels.csv")
        self.assertEqual(len(selected), 14)  # ceil .25 * 2 = one per training stratum
        self.assertTrue(set(selected.station_code.astype(str)) <= {"0", "1"})
        report = json.loads((output / "fold_1" / "basis_report.json").read_text())
        for layer in report["layers"].values():
            self.assertEqual(layer["training_images"], 14)
            self.assertEqual(layer["training_stations"], ["0", "1"])
        summary = pd.read_csv(output / "comparison_summary.csv")
        self.assertEqual(set(summary.method), set(METHODS))
        for method in METHODS:
            arm = output / "fold_1" / method
            checkpoint = torch.load(arm / "best_model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["label_coverage"]["selected_training_images"], 14)
            self.assertEqual(checkpoint["label_coverage"]["realized_percentage"], 50.)
            self.assertEqual(checkpoint["backbone"]["revision"], "synthetic-no-network")
            self.assertEqual(checkpoint["adapter_parameters"], 0 if method == "none" else 512)
            self.assertIn(checkpoint["epoch"], (1, 2))
            self.assertEqual(len(pd.read_csv(arm / "val_predictions.csv")), 14)
            self.assertEqual(len(pd.read_csv(arm / "test_predictions.csv")), 14)
        combined = pd.read_csv(output / "combined_test_predictions.csv")
        self.assertEqual(len(combined), 4 * 14)
        self.assertTrue((combined.station_code == 3).all())


if __name__ == "__main__":
    unittest.main()
