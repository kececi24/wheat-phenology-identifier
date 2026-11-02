"""Strict, deterministic image tiling; no held-out image is silently dropped."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def tile_boxes(width, height, size, stride):
    # Same edge-covering tile geometry as DINOv3_BBCH/precompute_*.
    def starts(length):
        if length <= size:
            return [0]
        positions = list(range(0, length - size + 1, stride))
        if positions[-1] != length - size:
            positions.append(length - size)
        return positions
    return [(x, y, x + size, y + size) for y in starts(height) for x in starts(width)]


def choose_boxes(boxes, max_tiles):
    if max_tiles == 0 or len(boxes) <= max_tiles:
        return boxes
    if max_tiles == 1:
        return [boxes[len(boxes) // 2]]
    step = (len(boxes) - 1) / (max_tiles - 1)
    return [boxes[i] for i in sorted({round(j * step) for j in range(max_tiles)})]


class TiledImages(Dataset):
    def __init__(self, rows, settings, train=False):
        self.rows = rows.reset_index(drop=True)
        self.settings = settings
        steps = [T.Resize((settings["image_size"], settings["image_size"]))]
        if train and settings.get("augment", False):
            steps.extend([T.RandomHorizontalFlip(), T.ColorJitter(brightness=.2, contrast=.2,
                                                                saturation=.15, hue=.03)])
        self.transform = T.Compose([*steps, T.ToTensor(),
                                    T.Normalize(settings["mean"], settings["std"])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        # Fail loudly for unreadable inputs in ALL roles; never drop a test day.
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            boxes = choose_boxes(tile_boxes(image.width, image.height,
                                            self.settings["tile_size"],
                                            self.settings["tile_stride"]),
                                 self.settings["max_tiles"])
            tiles = torch.stack([self.transform(image.crop(box)) for box in boxes])
        return {"tiles": tiles, "target": torch.tensor(row.active_target, dtype=torch.float32),
                "hard": int(row.class_idx), "index": index}


def collate_images(items):
    count = max(len(item["tiles"]) for item in items)
    first = items[0]["tiles"]
    tiles = first.new_zeros(len(items), count, *first.shape[1:])
    mask = torch.zeros(len(items), count, dtype=torch.bool)
    for i, item in enumerate(items):
        tiles[i, :len(item["tiles"])] = item["tiles"]
        mask[i, :len(item["tiles"])] = True
    return {"tiles": tiles, "tile_mask": mask,
            "target": torch.stack([item["target"] for item in items]),
            "hard": torch.tensor([item["hard"] for item in items]),
            "index": torch.tensor([item["index"] for item in items])}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(rows, settings, args, train=False, seed=42):
    return DataLoader(TiledImages(rows, settings, train), batch_size=args.batch_size,
                      shuffle=train, num_workers=args.num_workers,
                      pin_memory=torch.device(args.device).type == "cuda",
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collate_images)
