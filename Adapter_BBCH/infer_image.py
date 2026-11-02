"""Predict one raw image from a selected transformer-block adapter checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from Adapter_BBCH.block_model import from_checkpoint
from Adapter_BBCH.image_data import TiledImages, collate_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = from_checkpoint(checkpoint, args.device)
    # Dummy target only satisfies the shared dataset interface; it is never a model input.
    rows = pd.DataFrame([{"image_path": args.image, "active_target": [1.] + [0.] * 6,
                          "class_idx": 0}])
    item = TiledImages(rows, checkpoint["preprocessing"], train=False)[0]
    batch = collate_images([item])
    with torch.inference_mode():
        probs = model(batch["tiles"].to(args.device), batch["tile_mask"].to(args.device)).softmax(-1)[0].cpu()
    classes = checkpoint["classes"]
    print(json.dumps({"label": classes[probs.argmax().item()],
                      "probabilities": dict(zip(classes, probs.tolist()))}, indent=2))


if __name__ == "__main__":
    main()
