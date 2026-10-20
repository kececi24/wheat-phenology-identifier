"""Configuration tests that do not load data or download backbone weights."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_multiscale_training import parse_args


class TrainingConfigTests(unittest.TestCase):
    def parse_yaml(self, content, *overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(content, encoding="utf-8")
            return parse_args(["--config", str(path), *overrides])

    def assert_config_error(self, content, message):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.parse_yaml(content)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_cli_only_defaults_are_preserved(self):
        args = parse_args([])
        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.batch_size, 2)
        self.assertTrue(args.amp)
        self.assertFalse(args.use_location_metadata)

    def test_yaml_types_aliases_and_optional_values(self):
        args = self.parse_yaml(
            "epochs: 3\nbatch-size: 8\nlr: 1e-4\namp: false\n"
            "use_location_metadata: true\nlabel_path: labels.csv\nembedding_cache: null\n"
        )
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.lr, 1e-4)
        self.assertFalse(args.amp)
        self.assertTrue(args.use_location_metadata)
        self.assertEqual(args.excel_path, "labels.csv")
        self.assertIsNone(args.embedding_cache)

    def test_cli_overrides_yaml_in_both_boolean_directions(self):
        args = self.parse_yaml(
            "epochs: 3\namp: false\ntensorboard: true\nuse_days_since_planting: true\n",
            "--epochs", "9", "--amp", "--no-tensorboard", "--no-days-since-planting",
        )
        self.assertEqual(args.epochs, 9)
        self.assertTrue(args.amp)
        self.assertFalse(args.tensorboard)
        self.assertFalse(args.use_days_since_planting)
        # Parsing another run must not inherit YAML defaults from this run.
        self.assertEqual(parse_args([]).epochs, 20)

    def test_rejects_invalid_documents_and_options(self):
        cases = [
            ("", "must contain a mapping"),
            ("- epochs\n- 3", "must contain a mapping"),
            ("epochs: [", "cannot load config"),
            ("epohs: 3", "unknown training option"),
            ("config: nested.yaml", "unknown training option"),
            ("epochs: 3\nepochs: 4", "duplicate configuration key"),
            ("batch_size: 3\nbatch-size: 4", "duplicate training option"),
            ("excel_path: a.csv\nlabel_path: b.csv", "duplicate training option"),
            ("epochs: 1.5", "must be int"),
            ("epochs: true", "must be int"),
            ('amp: "false"', "must be bool"),
            ("amp: null", "cannot be null"),
            ("data_path: 123", "must be str"),
            ("lr: false", "must be float"),
            ("lr: .nan", "must be finite"),
            ("loss: invalid", "must be one of"),
            ("ordinal_power: 3", "must be one of"),
        ]
        for content, message in cases:
            with self.subTest(content=content):
                self.assert_config_error(content, message)

    def test_missing_config_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config", str(Path(directory) / "missing.yaml")])

    def test_yaml_uses_existing_cross_option_validation(self):
        self.assert_config_error("ordinal_ce_weight: 2", "must be in [0, 1]")
        self.assert_config_error("monotonic_decoding: viterbi", "requires --exclude-offseason")
        args = self.parse_yaml("monotonic_decoding: viterbi", "--exclude-offseason")
        self.assertTrue(args.exclude_offseason)

    def test_examples_match_the_documented_training_presets(self):
        args = parse_args(["--config", str(ROOT / "configs" / "train.yaml")])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.loss, "hybrid")
        self.assertEqual(args.temporal_feature_fusion, "legacy")
        loso = parse_args(["--config", str(ROOT / "configs" / "train_loso.yaml")])
        self.assertEqual(loso.fold_strategy, "loso")
        self.assertEqual(loso.test_groups, 1)
        self.assertEqual(loso.expected_stations, 15)
        self.assertEqual(loso.temporal_feature_fusion, "gated")

    def test_finetuned_runner_generates_compatible_training_yaml(self):
        import yaml
        from finetuned_loso_runner import fold_commands, load_experiment

        config = load_experiment(ROOT / "configs/finetuned_loso/default.yaml")
        _, training, _, _, cache, _ = fold_commands(config, 2)
        args = self.parse_yaml(yaml.safe_dump(training))
        self.assertEqual(args.only_fold, 2)
        self.assertEqual(args.embedding_cache, str(cache))
        self.assertEqual(args.fold_strategy, "loso")
        self.assertEqual(args.epochs, 30)


if __name__ == "__main__":
    unittest.main()
