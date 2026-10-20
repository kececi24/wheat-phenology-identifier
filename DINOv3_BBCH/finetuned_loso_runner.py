"""Run independent YAML-defined fine-tuning/cache/temporal LOSO experiments."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_DEFAULTS = {
    "label_path": "labeling_truncated3.csv", "data_path": "data",
    "image_backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "expected_stations": 15, "fold_start": 1, "fold_end": None,
    "validation_groups": 2, "fold_seed": 42, "seed": 42,
    "stream": "micro", "camera": "AUTO", "exclude_offseason": True,
    "gpu_id": 0, "keep_caches": False,
}
OWNED_OPTIONS = {
    "label_path", "excel_path", "data_path", "out_dir", "cache_path", "image_backbone",
    "fold_id", "only_fold", "fold_strategy", "fold_group_by", "folds",
    "fold_seed", "seed", "expected_stations", "validation_groups", "test_groups",
    "embedding_cache", "resume_checkpoint", "device", "stream", "camera",
    "exclude_offseason", "config",
}
NEGATIVE_FLAGS = {
    "finetune": {"exclude_offseason": "include-offseason",
                 "unfreeze_final_norm": "freeze-final-norm", "dense_include_cls": "no-dense-cls"},
    "precompute": {"pretrained": "no-pretrained", "dense_features": "no-dense-features",
                   "dense_include_cls": "no-dense-include-cls"},
}


class ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML option names must be strings")
            name = key.replace("-", "_")
            if name in result:
                raise ValueError(f"Duplicate YAML option: {key}")
            result[name] = self.construct_object(value_node, deep=deep)
        return result


def load_experiment(path):
    with path.open(encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=ConfigLoader)
    sections = {"pipeline", "finetune", "precompute", "training"}
    if not isinstance(config, dict) or set(config) != sections:
        raise ValueError(f"{path}: expected sections {sorted(sections)}")
    for section, options in config.items():
        if not isinstance(options, dict):
            raise ValueError(f"{path}: {section} must be a mapping")
    pipeline = {**PIPELINE_DEFAULTS, **config["pipeline"]}
    unknown = set(pipeline) - (set(PIPELINE_DEFAULTS) | {"result_root"})
    if unknown:
        raise ValueError(f"{path}: unknown pipeline options: {sorted(unknown)}")
    for key in ("label_path", "data_path", "image_backbone", "result_root", "stream", "camera"):
        if not isinstance(pipeline.get(key), str) or not pipeline[key].strip():
            raise ValueError(f"{path}: pipeline.{key} must be a nonempty string")
    if pipeline["fold_end"] is None:
        pipeline["fold_end"] = pipeline["expected_stations"]
    for key in ("expected_stations", "fold_start", "fold_end", "validation_groups", "fold_seed", "seed"):
        if type(pipeline[key]) is not int:
            raise ValueError(f"{path}: pipeline.{key} must be an integer")
    if not 1 <= pipeline["fold_start"] <= pipeline["fold_end"] <= pipeline["expected_stations"]:
        raise ValueError(f"{path}: invalid fold range")
    if not 1 <= pipeline["validation_groups"] < pipeline["expected_stations"] - 1:
        raise ValueError(f"{path}: validation_groups must leave at least one training station")
    for key in ("keep_caches", "exclude_offseason"):
        if type(pipeline[key]) is not bool:
            raise ValueError(f"{path}: pipeline.{key} must be boolean")
    if pipeline["stream"] not in {"micro", "macro"}:
        raise ValueError(f"{path}: pipeline.stream must be micro or macro")
    if type(pipeline["gpu_id"]) not in (int, str) or not str(pipeline["gpu_id"]).strip():
        raise ValueError(f"{path}: pipeline.gpu_id must identify a GPU")
    config["pipeline"] = pipeline
    for stage in ("finetune", "precompute", "training"):
        conflicts = OWNED_OPTIONS.intersection(config[stage])
        if conflicts:
            raise ValueError(f"{path}: {stage} cannot override pipeline-managed options: {sorted(conflicts)}")
        if any(value is not None and type(value) not in (str, int, float, bool)
               for value in config[stage].values()):
            raise ValueError(f"{path}: {stage} options must be scalar values")
    # Check boolean translation before launching any experiment.
    for stage in ("finetune", "precompute"):
        cli_arguments(stage, config[stage])
    return config


def cli_arguments(stage, options):
    arguments = []
    for key, value in options.items():
        flag = key.replace("_", "-")
        if value is None:
            continue
        if type(value) is bool:
            if value:
                arguments.append(f"--{flag}")
            elif key in NEGATIVE_FLAGS[stage]:
                arguments.append(f"--{NEGATIVE_FLAGS[stage][key]}")
            elif key != "ignore_status_csv":
                raise ValueError(f"{stage}.{key}: no supported false flag")
        else:
            arguments.extend((f"--{flag}", str(value)))
    return arguments


def fold_commands(config, fold):
    p = config["pipeline"]
    root = Path(p["result_root"])
    backbone = root / "backbones" / f"fold_{fold}" / "backbone"
    cache = root / "caches" / f"fold_{fold}" / "vit_embeddings.pt"
    run = root / "temporal" / f"run_fold_{fold}"
    shared = {key: p[key] for key in ("label_path", "data_path", "stream", "camera")}
    split = {key: p[key] for key in ("fold_seed", "validation_groups", "expected_stations", "seed")}
    fine = {**config["finetune"], **shared, **split, "image_backbone": p["image_backbone"],
            "out_dir": str(root / "backbones"), "fold_id": fold,
            "exclude_offseason": p["exclude_offseason"], "device": "cuda:0"}
    precompute = {**config["precompute"], **shared, "image_backbone": str(backbone),
                  "out_dir": str(cache.parent), "device": "cuda:0"}
    training = {**config["training"], **shared, **split, "out_dir": str(run),
                "embedding_cache": str(cache), "folds": p["expected_stations"],
                "fold_strategy": "loso", "fold_group_by": "station", "only_fold": fold,
                "test_groups": 1, "exclude_offseason": p["exclude_offseason"], "device": "cuda:0"}
    checkpoint = run / f"fold_{fold}" / "last_checkpoint.pt"
    if checkpoint.is_file():
        training["resume_checkpoint"] = str(checkpoint)
    training_config = run / "training.yaml"
    commands = [
        [sys.executable, str(SCRIPT_DIR / "finetune_dinov3_backbone.py"), *cli_arguments("finetune", fine)],
        [sys.executable, str(SCRIPT_DIR / "precompute_multiscale_embeddings.py"), *cli_arguments("precompute", precompute)],
        [sys.executable, str(SCRIPT_DIR / "run_multiscale_training.py"), "--config", str(training_config)],
    ]
    return commands, training, training_config, backbone, cache, run


def check_backbone_access(model_id):
    if Path(model_id).is_dir():
        if not (Path(model_id) / "config.json").is_file():
            raise ValueError(f"Local backbone has no config.json: {model_id}")
        return
    from huggingface_hub import HfApi, get_token, whoami
    token = get_token()
    if not token:
        raise ValueError("No Hugging Face token visible. Run hf auth login or export HF_TOKEN.")
    account = whoami(token=token)["name"]
    HfApi().model_info(model_id, token=token)
    print(f"Authenticated backbone access confirmed for {model_id} as {account}.", flush=True)


def run_experiment(path, config, dry_run=False):
    p = config["pipeline"]
    root = Path(p["result_root"])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(p["gpu_id"])}
    if not dry_run:
        if not Path(p["label_path"]).is_file() or not Path(p["data_path"]).is_dir():
            raise ValueError(f"{path}: label_path must be a file and data_path must be a directory")
        check_backbone_access(p["image_backbone"])
    for fold in range(p["fold_start"], p["fold_end"] + 1):
        commands, training, training_config, backbone, cache, run = fold_commands(config, fold)
        done = run / f"fold_{fold}" / "test_metrics.json"
        print(f"[{path.name}] Fold {fold}/{p['expected_stations']} (physical GPU {p['gpu_id']})", flush=True)
        if dry_run:
            for command in commands:
                print(shlex.join(command), flush=True)
            print(yaml.safe_dump(training, sort_keys=False), flush=True)
            continue
        if not done.is_file():
            if not all((backbone / name).is_file() for name in ("config.json", "wheat_finetune_metadata.json")):
                subprocess.run(commands[0], check=True, env=env, cwd=PROJECT_ROOT)
            if not cache.is_file():
                cache.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(commands[1], check=True, env=env, cwd=PROJECT_ROOT)
            training_config.parent.mkdir(parents=True, exist_ok=True)
            training_config.write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
            subprocess.run(commands[2], check=True, env=env, cwd=PROJECT_ROOT)
        if not done.is_file():
            raise RuntimeError(f"Fold {fold} did not produce {done}; refusing cache cleanup")
        if not p["keep_caches"] and cache.is_file():
            cache.unlink()
    aggregate = [sys.executable, str(SCRIPT_DIR / "aggregate_finetuned_loso.py"),
                 str(root / "temporal"), str(p["fold_start"]), str(p["fold_end"])]
    if dry_run:
        print(shlex.join(aggregate), flush=True)
    else:
        subprocess.run(aggregate, check=True, cwd=PROJECT_ROOT)


def select_configs(folder, names):
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Config directory not found: {folder}")
    if not names:
        paths = sorted(path for path in folder.iterdir() if path.suffix in {".yaml", ".yml"} and path.is_file())
    else:
        paths = []
        for name in names:
            path = folder / name
            if not path.suffix:
                matches = [candidate for suffix in (".yaml", ".yml") if (candidate := path.with_suffix(suffix)).is_file()]
                if len(matches) != 1:
                    raise ValueError(f"Config name must resolve to exactly one YAML file: {name}")
                path = matches[0]
            if path.resolve().parent != folder or path.suffix not in {".yaml", ".yml"} or not path.is_file():
                raise ValueError(f"Select a YAML filename inside {folder}: {name}")
            paths.append(path.resolve())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Select at least one config, without duplicates")
    return paths


def run_batch(experiments, parallel):
    pending = iter(experiments)
    active = {}
    failed = []
    try:
        while True:
            while len(active) < parallel:
                item = next(pending, None)
                if item is None:
                    break
                path, config = item
                root = Path(config["pipeline"]["result_root"])
                root.mkdir(parents=True, exist_ok=True)
                snapshot = root / "pipeline.yaml"
                snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                log_path = root / "pipeline.log"
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                                "--worker", str(snapshot.resolve())],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT,
                                               start_new_session=os.name != "nt")
                active[process] = (path, log_path)
                print(f"Started {path.name}; log: {log_path}", flush=True)
            if not active:
                break
            for process, (path, log_path) in list(active.items()):
                code = process.poll()
                if code is not None:
                    del active[process]
                    print(f"{path.name}: {'completed' if code == 0 else 'FAILED'}; log: {log_path}", flush=True)
                    if code:
                        failed.append(path.name)
            time.sleep(0.1)
    finally:
        # Workers and their training subprocesses share a process group on Linux.
        for process in active:
            if os.name == "nt":
                process.terminate()
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return bool(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=SCRIPT_DIR / "configs" / "finetuned_loso")
    parser.add_argument("--configs", nargs="+", help="Selected YAML filenames or stems; omitted means all files in the folder")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Maximum simultaneous experiments (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training, authentication, or output writes")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    try:
        if args.worker:
            run_experiment(args.worker, load_experiment(args.worker))
            return 0
        if args.parallel < 1:
            raise ValueError("--parallel must be at least 1")
        experiments = [(path, load_experiment(path)) for path in select_configs(args.config_dir, args.configs)]
        roots = [Path(config["pipeline"]["result_root"]).resolve() for _, config in experiments]
        for index, root in enumerate(roots):
            if any(root == other or root in other.parents or other in root.parents for other in roots[:index]):
                raise ValueError("Selected configs must have separate, non-overlapping result_root directories")
            snapshot = root / "pipeline.yaml"
            if snapshot.is_file() and load_experiment(snapshot) != experiments[index][1]:
                raise ValueError(f"{root}: existing pipeline settings differ; choose a new result_root")
        if args.dry_run:
            for path, config in experiments:
                run_experiment(path, config, dry_run=True)
            return 0
        return int(run_batch(experiments, args.parallel))
    except (OSError, ValueError, RuntimeError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
