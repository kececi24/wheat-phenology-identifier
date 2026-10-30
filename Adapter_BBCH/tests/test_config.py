"""Optional remote checks for config errors and CLI precedence; not run locally."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from Adapter_BBCH.run_adapter_training import parse_args


class ConfigTests(unittest.TestCase):
    def parse_yaml(self, contents, *overrides):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            return parse_args(["--config", str(path), *overrides])

    def test_typed_values_lists_null_and_scientific_notation(self):
        args = self.parse_yaml("dry_run: true\nmethods: [transition, random]\n"
                              "rank: 3\nlr: 3e-4\nexpected_stations: null\n"
                              "ignore_status_csv: true\nfold-id: 0\n")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.methods, ["transition", "random"])
        self.assertEqual(args.rank, 3)
        self.assertAlmostEqual(args.lr, 0.0003)
        self.assertIsNone(args.expected_stations)
        self.assertTrue(args.ignore_status_csv)
        self.assertEqual(args.fold_id, 0)

    def test_cli_overrides_scalars_lists_and_booleans_in_both_directions(self):
        args = self.parse_yaml("dry_run: true\nignore_status_csv: true\n"
                              "embedding_cache: frozen.pt\nrank: 3\nmethods: [all]\n",
                              "--no-dry-run", "--no-ignore-status-csv",
                              "--rank", "2", "--methods", "pca", "random")
        self.assertFalse(args.dry_run)
        self.assertFalse(args.ignore_status_csv)
        self.assertEqual(args.rank, 2)
        self.assertEqual(args.methods, ["pca", "random"])
        args = self.parse_yaml("dry_run: false\nignore_status_csv: false\n",
                              "--dry-run", "--ignore-status-csv")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.ignore_status_csv)

    def test_rejects_typo_duplicate_key_wrong_types_and_invalid_budgets(self):
        invalid = ["epochs_typo: 10", "rank: 2\nrank: 3", "fold-id: 1\nfold_id: 2",
                   "rank: true", 'dry_run: "false"', "methods: all", "methods: []",
                   "methods: [missing]", "rank: null", "rank: -1", "lr: .nan",
                   "methods: [all, pca]", "label_fraction: 0", "label_fraction: false",
                   "label_percentage: true", "label_percentage: null",
                   "stream: both",
                   "- rank: 2", "config: other.yaml"]
        for text in invalid:
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_yaml(text, "--dry-run")

    def test_cli_only_defaults_and_packaged_config(self):
        self.assertEqual(parse_args(["--dry-run"]).rank, 6)
        path = Path(__file__).resolve().parents[1] / "config.yaml"
        args = parse_args(["--config", str(path), "--dry-run"])
        self.assertEqual(args.methods, ["all"])
        self.assertEqual(args.fold_id, 1)
        self.assertEqual(args.embedding_cache, "results_adapter_frozen_cache/vit_embeddings.pt")
        self.assertEqual(args.label_percentage, 100)
        args = parse_args(["--config", str(path), "--dry-run", "--label-fraction", "0.25"])
        self.assertEqual(args.label_percentage, 25)
        self.assertEqual(args.label_fraction, 0.25)

    def test_label_percentage_and_fraction_alias_precedence(self):
        args = self.parse_yaml("label_percentage: 50\n", "--dry-run", "--label-fraction", "0.25")
        self.assertEqual(args.label_percentage, 25)
        self.assertEqual(args.label_fraction, 0.25)
        args = self.parse_yaml("label_fraction: 0.5\n", "--dry-run", "--label-percentage", "25")
        self.assertEqual(args.label_percentage, 25)
        self.assertEqual(args.label_fraction, 0.25)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse_yaml("label_percentage: 50\nlabel_fraction: 0.5\n", "--dry-run")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--dry-run", "--label-percentage", "25", "--label-fraction", "0.25"])

    def test_rejects_nonfinite_or_out_of_range_label_budgets(self):
        for option, value in (("--label-percentage", "0"), ("--label-percentage", "100.1"),
                              ("--label-percentage", "nan"), ("--label-percentage", "inf"),
                              ("--label-fraction", "nan")):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args(["--dry-run", option, value])

    def test_missing_config_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config", str(Path(folder) / "missing.yaml"), "--dry-run"])


if __name__ == "__main__":
    unittest.main()
