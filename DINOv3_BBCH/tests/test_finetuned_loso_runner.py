import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import finetuned_loso_runner as runner


class FinetunedLOSORunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = runner.load_experiment(ROOT / "configs/finetuned_loso/default.yaml")
        self.config["pipeline"].update(result_root=str(self.directory / "results"), fold_end=1)
        self.path = self.write_config("first", self.config)

    def write_config(self, name, config):
        path = self.directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_selects_multiple_names_or_all_yaml_files(self):
        second = self.write_config("second", self.config)
        (self.directory / "ignore.txt").touch()
        self.assertEqual(runner.select_configs(self.directory, ["second", "first.yaml"]), [second, self.path])
        self.assertEqual(runner.select_configs(self.directory, None), [self.path, second])
        for names in (["first", "first.yaml"], ["missing"], ["../outside.yaml"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                runner.select_configs(self.directory, names)

    def test_rejects_duplicate_keys_and_shared_option_overrides(self):
        self.path.write_text("pipeline: {}\npipeline: {}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.load_experiment(self.path)
        for stage in ("finetune", "precompute", "training"):
            config = copy.deepcopy(self.config)
            config[stage]["fold_seed"] = 123
            path = self.write_config(stage, config)
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "pipeline-managed"):
                runner.load_experiment(path)

    def test_rejects_colliding_result_roots_before_launch(self):
        self.write_config("second", self.config)
        with patch.object(runner, "run_batch") as batch, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(["--config-dir", str(self.directory), "--parallel", "2"]), 1)
            batch.assert_not_called()

    def test_can_enable_epoch_jsonl_on_resume_without_changing_training_settings(self):
        root = Path(self.config["pipeline"]["result_root"])
        root.mkdir()
        old_config = copy.deepcopy(self.config)
        old_config["training"].pop("epoch_jsonl", None)
        (root / "pipeline.yaml").write_text(yaml.safe_dump(old_config), encoding="utf-8")
        self.config["training"]["epoch_jsonl"] = True
        self.write_config("first", self.config)
        with patch.object(runner, "run_batch", return_value=False) as batch:
            self.assertEqual(runner.main(["--config-dir", str(self.directory), "--configs", "first"]), 0)
            batch.assert_called_once()
        self.config["training"]["epochs"] += 1
        self.write_config("first", self.config)
        with patch.object(runner, "run_batch") as batch, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(["--config-dir", str(self.directory), "--configs", "first"]), 1)
            batch.assert_not_called()

    def test_builds_fold_safe_commands_and_resumes_checkpoint(self):
        commands, training, config_path, backbone, cache, run = runner.fold_commands(self.config, 3)
        self.assertEqual(commands[0][commands[0].index("--fold-id") + 1], "3")
        self.assertEqual(commands[1][commands[1].index("--image-backbone") + 1], str(backbone))
        self.assertEqual(training["embedding_cache"], str(cache))
        self.assertEqual(training["only_fold"], 3)
        self.assertEqual(training["fold_strategy"], "loso")
        self.assertEqual(training["fold_seed"], self.config["pipeline"]["fold_seed"])
        self.assertEqual(commands[2][-2:], ["--config", str(config_path)])
        checkpoint = run / "fold_3/last_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        self.assertEqual(runner.fold_commands(self.config, 3)[1]["resume_checkpoint"], str(checkpoint))

    def test_generated_stage_flags_match_existing_script_parsers(self):
        commands = runner.fold_commands(self.config, 1)[0]
        for command in commands[:2]:
            with self.subTest(script=command[1]):
                # Execute only argparse definitions, avoiding model imports and downloads.
                tree = ast.parse(Path(command[1]).read_text(encoding="utf-8"))
                main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
                statements = []
                for statement in main.body:
                    if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "args" for target in statement.targets
                    ):
                        break
                    statements.append(statement)
                namespace = {"argparse": argparse, "os": os, "DINO_DEFAULT_BACKBONE": "unused",
                             "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
                exec(compile(ast.Module(body=statements, type_ignores=[]), command[1], "exec"), namespace)
                namespace["parser"].parse_args(command[2:])

    def test_dry_run_has_no_authentication_processes_or_output_writes(self):
        with patch.object(runner, "check_backbone_access") as auth, patch.object(runner.subprocess, "run") as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.path, self.config, dry_run=True)
            auth.assert_not_called()
            execute.assert_not_called()
        self.assertFalse(Path(self.config["pipeline"]["result_root"]).exists())

    def test_stage_order_cleanup_and_completed_fold_skip(self):
        labels = self.directory / "labels.csv"
        labels.touch()
        self.config["pipeline"].update(label_path=str(labels), data_path=str(self.directory), gpu_id=2)
        _, _, _, _, cache, run = runner.fold_commands(self.config, 1)
        calls = []

        def execute(command, **kwargs):
            script = Path(command[1]).name
            calls.append(script)
            if script == "precompute_multiscale_embeddings.py":
                cache.touch()
            if script == "run_multiscale_training.py":
                done = run / "fold_1/test_metrics.json"
                done.parent.mkdir(parents=True)
                done.write_text("{}")
                resolved = yaml.safe_load(Path(command[-1]).read_text())
                self.assertEqual(resolved["only_fold"], 1)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")

        with patch.object(runner, "check_backbone_access"), patch.object(runner.subprocess, "run", side_effect=execute):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_experiment(self.path, self.config)
                self.assertFalse(cache.exists())
                runner.run_experiment(self.path, self.config)
        self.assertEqual(calls, ["finetune_dinov3_backbone.py", "precompute_multiscale_embeddings.py",
                                 "run_multiscale_training.py", "aggregate_finetuned_loso.py",
                                 "aggregate_finetuned_loso.py"])

    def test_failed_training_keeps_cache_and_skips_aggregation(self):
        labels = self.directory / "labels.csv"
        labels.touch()
        self.config["pipeline"].update(label_path=str(labels), data_path=str(self.directory))
        _, _, _, backbone, cache, _ = runner.fold_commands(self.config, 1)
        backbone.mkdir(parents=True)
        for name in ("config.json", "wheat_finetune_metadata.json"):
            (backbone / name).touch()
        cache.parent.mkdir(parents=True)
        cache.touch()
        with patch.object(runner, "check_backbone_access"), patch.object(
            runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "training")
        ) as execute, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            runner.run_experiment(self.path, self.config)
        self.assertTrue(cache.exists())
        self.assertEqual(execute.call_count, 1)

    def test_parallel_limit_and_failure_reporting_with_real_processes(self):
        # Cheap stand-in workers record their lifetimes; no GPU or training imports.
        worker = self.directory / "worker.py"
        worker.write_text(
            "import json, pathlib, sys, time\n"
            "path = pathlib.Path(sys.argv[1])\n"
            "start = time.monotonic()\n"
            "time.sleep(0.4)\n"
            "path.write_text(json.dumps([start, time.monotonic()]))\n"
            "sys.exit(int(sys.argv[2]))\n", encoding="utf-8"
        )
        real_popen = subprocess.Popen
        records = []

        def start_worker(command, **kwargs):
            record = self.directory / f"times_{len(records)}.json"
            records.append(record)
            return real_popen([sys.executable, str(worker), str(record), str(int(len(records) == 2))], **kwargs)

        experiments = []
        for index in range(3):
            config = copy.deepcopy(self.config)
            config["pipeline"]["result_root"] = str(self.directory / f"results_{index}")
            experiments.append((self.path, config))
        with patch.object(runner.subprocess, "Popen", side_effect=start_worker), contextlib.redirect_stdout(io.StringIO()):
            failed = runner.run_batch(experiments, 2)
        self.assertTrue(failed)
        times = [json.loads(path.read_text()) for path in records]
        self.assertLess(max(times[0][0], times[1][0]), min(times[0][1], times[1][1]))
        self.assertGreaterEqual(times[2][0], min(times[0][1], times[1][1]))
        for _, config in experiments:
            root = Path(config["pipeline"]["result_root"])
            self.assertTrue((root / "pipeline.log").is_file())
            self.assertTrue((root / "pipeline.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
