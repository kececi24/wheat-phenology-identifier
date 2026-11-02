"""Optional remote schema checks for the raw-image block adapter runner."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from Adapter_BBCH.block_config import parse_args


class BlockConfigTests(unittest.TestCase):
    def yaml_args(self, text, *overrides):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return parse_args(["--config", str(path), *overrides])

    def test_supplied_configs_and_defaults(self):
        root = Path(__file__).resolve().parents[1]
        for filename, budget in (("config.yaml", 100), ("config25.yaml", 25), ("config50.yaml", 50)):
            args = parse_args(["--config", str(root / "config" / filename), "--dry-run"])
            self.assertEqual(args.methods, ["none", "lora", "standard", "transition"])
            self.assertEqual(args.label_percentage, budget)
            self.assertEqual(args.rank, 2 * args.lora_rank)
            self.assertEqual(args.fold_id, 0)
            self.assertTrue(args.dry_run)
        self.assertEqual(parse_args([]).fold_id, 1)

    def test_cli_overrides_yaml_aliases_and_booleans(self):
        args = self.yaml_args("label_percentage: 50\nmethods: [all]\naugment: true\nseed: 42\n",
                             "--label-fraction", ".25", "--methods", "none", "lora",
                             "--no-augment", "--seed", "43")
        self.assertEqual(args.label_percentage, 25)
        self.assertEqual(args.methods, ["none", "lora"])
        self.assertEqual(args.fold_seed, 42)
        self.assertEqual(args.label_seed, 42)
        self.assertEqual(args.seed, 43)
        self.assertFalse(args.augment)

    def test_rejects_old_cache_settings_and_invalid_schema(self):
        invalid = ["embedding_cache: old.pt", "methods: [pca]", "rank: true",
                   "rank: 6\nrank: 2", "label_percentage: null", "label_percentage: 0",
                   "label_percentage: 25\nlabel_fraction: .25", "methods: [all, lora]",
                   "prior_lambda: -1", "lora_rank: 0", "max_tiles: -1", "amp: invalid",
                   "device: cpu\namp: fp16", "image_backbone: local-supervised-backbone",
                   "methods: [none, none]", "batch_size: 0"]
        for text in invalid:
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.yaml_args(text)


if __name__ == "__main__":
    unittest.main()
