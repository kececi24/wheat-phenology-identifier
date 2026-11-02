"""Optional remote checks. Tiny real HF DINOv3, no network/pretrained downloads."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from Adapter_BBCH.adapter_model import transition_basis
from Adapter_BBCH.block_model import (
    CHECKPOINT_FORMAT, METHODS, BlockAdapterClassifier, BlockWithAdapter,
    Bottleneck, DenseExtractor, capture_layer_inputs, from_checkpoint,
)
from Adapter_BBCH.run_adapter_training import estimate_bases


def setUpModule():
    global _previous_threads
    _previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_previous_threads)


def tiny_extractor():
    config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, image_size=32, patch_size=16,
        num_register_tokens=2, attention_dropout=0., drop_path_rate=0.)
    return DenseExtractor(DINOv3ViTModel(config))


class BlockModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.base = tiny_extractor()
        self.pixels = torch.randn(2, 2, 3, 32, 32)
        self.mask = torch.tensor([[True, True], [True, False]])
        self.bases = {i: torch.linalg.qr(torch.randn(32, 4)).Q[:, :4].T for i in (1, 2)}

    def model(self, method):
        extractor = copy.deepcopy(self.base)
        torch.manual_seed(44)
        return BlockAdapterClassifier(extractor, method=method, last_blocks=2,
            rank=4, lora_rank=2, lora_alpha=2., hidden_dim=16, dropout=0.,
            tile_batch_size=2, bases=self.bases)

    def test_all_arms_start_identically_and_match_parameter_budget(self):
        reference = self.model("none").eval()
        with torch.no_grad():
            expected = reference(self.pixels, self.mask)
        for method in METHODS:
            model = self.model(method).eval()
            with torch.no_grad():
                actual = model(self.pixels, self.mask)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            count = model.parameter_counts()["adapter_parameters"]
            self.assertEqual(count, 0 if method == "none" else 2 * 2 * 32 * 4)
            # Masked padded pixels must not affect image predictions.
            changed = self.pixels.clone()
            changed[1, 1] = 1000
            with torch.no_grad():
                torch.testing.assert_close(model(changed, self.mask), actual)

    def test_only_adapters_and_head_receive_gradients_and_affect_output(self):
        for method in ("lora", "standard", "transition"):
            model = self.model(method).train()
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
            target = torch.tensor([1, 5])
            before = model(self.pixels, self.mask).detach()
            for _ in range(2):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(self.pixels, self.mask), target)
                (loss + .01 * model.prior_penalty()).backward()
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, name)
                optimizer.step()
            parameters = dict(model.named_parameters())
            for name, value in frozen.items():
                torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)
            self.assertFalse(torch.allclose(before, model(self.pixels, self.mask)))
            up_gradients = [p.grad for n, p in model.extractor.named_parameters() if "up.weight" in n and p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in up_gradients))
            down_gradients = [p.grad for n, p in model.extractor.named_parameters() if "down" in n and p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in down_gradients))
            self.assertTrue(all(m._z is None for m in model.modules() if isinstance(m, BlockWithAdapter)))

    def test_compact_checkpoint_restores_every_arm_and_prior_buffers(self):
        for method in METHODS:
            model = self.model(method).eval()
            with torch.no_grad():
                for p in model.extractor.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * .02)
                expected = model(self.pixels, self.mask)
            checkpoint = {"format": CHECKPOINT_FORMAT, "model_config": model.config,
                          "model": model.adaptation_state(),
                          "backbone": {"name": "fake", "revision": "fake-commit"}}
            self.assertFalse(any("embeddings" in key for key in checkpoint["model"]))
            with patch("Adapter_BBCH.block_model.make_extractor", return_value=copy.deepcopy(self.base)):
                restored = from_checkpoint(checkpoint, "cpu")
            with torch.no_grad():
                torch.testing.assert_close(restored(self.pixels, self.mask), expected)
            corrupted = dict(checkpoint["model"])
            corrupted.pop(next(iter(corrupted)))
            with self.assertRaises(ValueError):
                restored.load_adaptation_state(corrupted)

    def test_prior_regularizer_penalizes_only_transition_displacement(self):
        model = self.model("transition")
        self.assertEqual(float(model.prior_penalty()), 0.)
        for module in model.modules():
            if isinstance(module, Bottleneck):
                with torch.no_grad():
                    module.down.fill_(.5)
        self.assertAlmostEqual(float(model.prior_penalty()), 4 * 32 * .25)
        self.assertEqual(float(self.model("standard").prior_penalty()), 0.)

    def test_prior_estimation_uses_actual_per_layer_training_coordinates(self):
        torch.manual_seed(73)
        extractor = copy.deepcopy(self.base).eval()
        rows = pd.DataFrame([{"station_code": station, "station_year": f"{station}_2020",
                              "class_idx": stage}
                             for station in ("a", "b") for stage in range(7) for _ in range(2)])
        pixels = torch.randn(len(rows), 3, 32, 32)
        with torch.no_grad(), capture_layer_inputs(extractor, 2) as captured:
            extractor.backbone(pixel_values=pixels)
            expected_features = {i: torch.cat([z[:, :1], z[:, 3:]], 1).mean(1).clone()
                                 for i, z in captured.items()}
        args = SimpleNamespace(last_blocks=2, rank=6, min_stage_images=2,
            min_transition_stations=2, batch_size=len(rows), tile_batch_size=7, device="cpu")
        loader = [{"tiles": pixels[:, None], "tile_mask": torch.ones(len(rows), 1, dtype=torch.bool)}]
        with patch("Adapter_BBCH.run_adapter_training.make_loader", return_value=loader) as factory:
            bases, report = estimate_bases(args, extractor, rows, {})
        self.assertIs(factory.call_args.args[0], rows)
        self.assertFalse(factory.call_args.kwargs["train"])
        for i, features in expected_features.items():
            expected, _ = transition_basis(features, rows, 7, 6, 2, 2, normalize=False)
            torch.testing.assert_close(bases[i], expected, rtol=1e-3, atol=1e-3)
            self.assertEqual(report["layers"][i]["training_images"], len(rows))
            self.assertFalse(report["layers"][i]["normalize_image_features"])
        self.assertFalse(torch.allclose(bases[1], bases[2]))
        # Temporary hooks are cleaned up after estimation.
        from Adapter_BBCH.block_model import selected_blocks
        blocks, indices = selected_blocks(extractor, 2)
        self.assertTrue(all(not blocks[i].norm2._forward_hooks for i in indices))


if __name__ == "__main__":
    unittest.main()
