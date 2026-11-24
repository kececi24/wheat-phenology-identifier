"""Command-line and strict YAML configuration for the frozen-feature loss study."""

from __future__ import annotations

import argparse
import math

from Adapter_BBCH.config import load_config_defaults


METHODS = ("hard_ce", "soft_ce", "ordinal", "transition")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Flat YAML configuration; explicitly supplied CLI options take precedence",
    )
    parser.add_argument("--label-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument(
        "--embedding-cache",
        default="results_adapter_frozen_cache/vit_embeddings.pt",
        help="Original frozen pretrained DINOv3 cache",
    )
    parser.add_argument("--out-dir", default="results_loss_bbch_100")
    parser.add_argument("--stream", choices=("micro", "macro"), default="micro")
    parser.add_argument("--camera", choices=("AUTO", "K1", "K2"), default="AUTO")
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument(
        "--ignore-status-csv", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fold-id", type=int, default=0,
        help="One-based LOSO fold; 0 runs all folds",
    )
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--expected-stations", type=int)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=("all", *METHODS), default=list(METHODS))
    parser.add_argument("--label-percentage", type=float, default=100.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25,
                        help="Validation macro-F1 patience; 0 disables early stopping")
    parser.add_argument("--path-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=False,
        help="Audit discovered folds and path anchors without loading features or writing output",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        parser.set_defaults(**load_config_defaults(parser, preliminary.config))
    args = parser.parse_args(argv)

    if args.fold_id < 0 or args.validation_groups < 1:
        parser.error("fold-id must be nonnegative and validation-groups must be positive")
    if args.expected_stations is not None and args.expected_stations < 1:
        parser.error("expected-stations must be positive when provided")
    if min(args.fold_seed, args.label_seed, args.seed) < 0:
        parser.error("fold-seed, label-seed, and seed must be nonnegative")
    if args.transition_days < 0:
        parser.error("transition-days must be nonnegative")
    if not math.isfinite(args.label_percentage) or not 0 < args.label_percentage <= 100:
        parser.error("label-percentage must be finite and in (0,100]")
    if args.hidden_dim < 1 or args.epochs < 1 or args.patience < 0:
        parser.error("hidden-dim and epochs must be positive; patience must be nonnegative")
    if (not math.isfinite(args.dropout) or not 0 <= args.dropout < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.path_weight) or args.path_weight < 0):
        parser.error("dropout, learning rate, weight decay, and path weight are invalid")
    if not args.device or not args.device.strip():
        parser.error("device must be a nonempty torch device string")
    if len(set(args.methods)) != len(args.methods) or (
        "all" in args.methods and args.methods != ["all"]
    ):
        parser.error("Use --methods all alone, or unique explicit method names")
    if not args.dry_run and not args.embedding_cache:
        parser.error("--embedding-cache is required unless --dry-run is used")
    return args

