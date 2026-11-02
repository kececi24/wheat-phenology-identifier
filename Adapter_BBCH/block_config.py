"""CLI/YAML schema for transformer-block experiments."""
import argparse
import math

import torch

from Adapter_BBCH.block_model import METHODS
from Adapter_BBCH.config import load_config_defaults


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Image-only DINOv3 transformer-block PEFT LOSO")
    parser.add_argument("--config", help="Flat YAML; explicit CLI options override YAML values")
    parser.add_argument("--label-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_adapter_bbch_blocks_100")
    parser.add_argument("--image-backbone", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--backbone-revision", help="Optional HF commit/tag; resolved commit is saved")
    parser.add_argument("--stream", choices=["micro", "macro"], default="micro")
    parser.add_argument("--camera", choices=["AUTO", "K1", "K2"], default="AUTO")
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--ignore-status-csv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fold-id", type=int, default=1, help="One-based fold; 0 = complete LOSO")
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--expected-stations", type=int)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=42,
                        help="Independent seed for label subsets; hold fixed for training-seed repeats")
    parser.add_argument("--methods", nargs="+", choices=["all", *METHODS], default=["all"])
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--rank", type=int, default=6, help="Standard/transition bottleneck width")
    parser.add_argument("--lora-rank", type=int, default=3, help="Q/V rank; rank/2 matches parameter budget")
    parser.add_argument("--lora-alpha", type=float, default=3.0)
    parser.add_argument("--prior-lambda", type=float, default=.01,
                        help="Coefficient of mean over layers of sum(E**2); tune with validation only")
    parser.add_argument("--min-stage-images", type=int, default=2)
    parser.add_argument("--min-transition-stations", type=int, default=2)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--label-percentage", type=float)
    budget.add_argument("--label-fraction", type=float, help="Legacy fraction alias")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--dense-grid-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=112)
    parser.add_argument("--max-tiles", type=int, default=8, help="Same deterministic subset for every role; 0 = all")
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ordinal-ce-weight", type=float, default=.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False,
                        help="Discover metadata/folds only; no model loading or output writes")
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        defaults = load_config_defaults(parser, preliminary.config)
        if any(key in defaults and defaults[key] is None for key in ("label_fraction", "label_percentage")):
            parser.error("Label budget cannot be null")
        if "label_fraction" in defaults and "label_percentage" in defaults:
            parser.error("Set only one label budget spelling in YAML")
        if preliminary.label_fraction is not None or preliminary.label_percentage is not None:
            defaults.pop("label_fraction", None)
            defaults.pop("label_percentage", None)
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    positive = (args.epochs, args.batch_size, args.accumulation_steps, args.tile_batch_size,
                args.image_size, args.tile_size, args.tile_stride, args.dense_grid_size,
                args.rank, args.lora_rank, args.last_blocks, args.hidden_dim,
                args.min_stage_images, args.min_transition_stations, args.validation_groups)
    if min(positive) < 1 or min(args.fold_id, args.patience, args.num_workers,
                               args.max_tiles, args.transition_days) < 0:
        parser.error("Dimensions and support thresholds must be positive; counts cannot be negative")
    if args.expected_stations is not None and args.expected_stations < 3:
        parser.error("expected_stations must be at least three")
    if min(args.seed, args.fold_seed, args.label_seed) < 0:
        parser.error("Seeds must be nonnegative")
    if args.label_percentage is None and args.label_fraction is None:
        args.label_percentage = 100.0
    if args.label_percentage is not None:
        if not math.isfinite(args.label_percentage) or not 0 < args.label_percentage <= 100:
            parser.error("label_percentage must be finite and in (0,100]")
        args.label_fraction = args.label_percentage / 100
    else:
        if not math.isfinite(args.label_fraction) or not 0 < args.label_fraction <= 1:
            parser.error("label_fraction must be finite and in (0,1]")
        args.label_percentage = args.label_fraction * 100
    if not 0 <= args.dropout < 1 or not 0 <= args.ordinal_ce_weight <= 1:
        parser.error("Invalid dropout or ordinal_ce_weight")
    if any(not math.isfinite(v) or v < 0 for v in (args.prior_lambda, args.weight_decay)):
        parser.error("Regularization coefficients must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.lora_alpha)):
        parser.error("lr and lora_alpha must be finite and positive")
    if ("all" in args.methods and args.methods != ["all"]) or len(set(args.methods)) != len(args.methods):
        parser.error("Use [all] alone or unique method names")
    if not args.image_backbone.startswith("facebook/dinov3-"):
        parser.error("Use a public pretrained facebook/dinov3-* base, not a supervised local backbone")
    if torch.device(args.device).type not in {"cpu", "cuda"}:
        parser.error("device must be cpu or cuda[:index]")
    if args.amp != "none" and torch.device(args.device).type != "cuda":
        parser.error("AMP requires CUDA; use amp: none for CPU")
    return args
