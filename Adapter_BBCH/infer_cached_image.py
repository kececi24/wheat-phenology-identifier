"""Predict one cached image using a saved adapter; no labels or metadata needed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Adapter_BBCH.adapter_model import AdapterClassifier
from Adapter_BBCH.data import CLASSES, image_key, load_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--image-path", required=True,
                        help="Image path or portable station/year/camera/scale/file identity")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "bbch_transition_adapter_v1" or checkpoint["classes"] != CLASSES:
        raise ValueError("Unsupported checkpoint format or BBCH class order")
    expected = checkpoint["feature_provenance"]
    rows = pd.DataFrame({"image_key": [image_key(args.image_path)]})
    features, actual = load_features(args.embedding_cache, rows, expected["stream"])
    # Cache contents/hash may differ for new images; extraction settings may not.
    for key in ("pooling", "stream", "image_backbone", "feature_dim", "pretrained",
                "cache_format", "preprocess", "dense_features", "tiling"):
        if actual[key] != expected[key]:
            raise ValueError(f"Inference cache differs from training extraction setting: {key}")
    model = AdapterClassifier.from_checkpoint(checkpoint).to(args.device).eval()
    with torch.inference_mode():
        probs = model(features.to(args.device)).softmax(-1)[0].cpu()
    predicted = int(probs.argmax())
    print(json.dumps({"image": rows.iloc[0]["image_key"], "class": CLASSES[predicted],
                      "confidence": float(probs[predicted]),
                      "probabilities": dict(zip(CLASSES, probs.tolist()))}, indent=2))


if __name__ == "__main__":
    main()
