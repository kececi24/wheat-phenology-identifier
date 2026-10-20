"""
Clean dataset folders which might have mixed up 1x and 10x images. Currently only read-only and no in-place changes.
"""

from pathlib import Path
import os
from typing import TypedDict
from collections import Counter
import yaml
import argparse
import csv

from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MAGNIFICATION_LABELS = {"1X": 0, "10X": 1}

class ImageRecord(TypedDict):
    path: str
    station: str
    year: str
    camera: str
    magnification: str


def collect_trusted_images(trusted_folders: list[Path]) -> list[ImageRecord]:
    """Collect images from explicitly verified magnification folders."""

    valid_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".webp", ".bmp"}
    metadata_list: list[ImageRecord] = []

    for folder in trusted_folders:
        parts = folder.parts
        if len(parts) < 4:
            continue

        station, year, camera, magnification = parts[-4:]

        if magnification not in {"1X", "10X"}:
            raise ValueError(f"Unexpected magnification folder: {folder}")

        with os.scandir(folder) as entries:
            for entry in entries:
                if(
                    entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_extensions
                ):
                    metadata_list.append(
                        {
                            "path": Path(entry.path).as_posix(),
                            "station": station,
                            "year": year,
                            "camera": camera,
                            "magnification": magnification
                        }
                    )

    return metadata_list


class ImageSorterDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None) -> None:
        # Store the records and optional image transformation.
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record: ImageRecord = self.records[index]
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            return image, MAGNIFICATION_LABELS[record["magnification"]]

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database image cleaner model training")
    parser.add_argument("--config", type=Path, required=True, help="Location of .yaml config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()


    config_path = args.config.resolve()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    save_dir = (config_path.parent / config["output"]["model_dir"]).resolve()
    data_config = config["data"]
    data_root = (config_path.parent / data_config["root"]).resolve()
    folders = [data_root / folder for folder in data_config["trusted_folders"]]

    records = collect_trusted_images(folders)
    print(f"Found {len(records)} images")

    print(Counter(record["magnification"] for record in records))

    model_name = config["model"]["name"]
    processor = AutoImageProcessor.from_pretrained(model_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])

    device_name = config["training"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print("Device: ", device)

    groups = [
        f"{record['station']}_{record['year']}"
        for record in records
    ]

    splitter = GroupKFold(n_splits=config["data"]["folds"])

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(records, groups=groups),
        start=1
    ):
        fold_dir = save_dir / f"fold_{fold}"

        fold_seed = config["training"]["seed"] + fold - 1
        torch.manual_seed(fold_seed)
        
        train_records = [records[t] for t in train_indices]
        val_records = [records[v] for v in val_indices]

        train_groups = {(r["station"], r["year"]) for r in train_records}
        val_groups = {(r["station"], r["year"]) for r in val_records}
        assert train_groups.isdisjoint(val_groups)
        assert {r["magnification"] for r in train_records} == {"1X", "10X"}
        assert {r["magnification"] for r in val_records} == {"1X", "10X"}

        print(f"Fold {fold}")
        print("Train:", Counter(r["magnification"] for r in train_records))
        print("Validation:", Counter(r["magnification"] for r in val_records)) 

        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "1X", 1: "10X"},
            label2id=MAGNIFICATION_LABELS,
            ignore_mismatched_sizes=True
        )

        freeze_backbone = config["model"]["freeze_backbone"]
        for parameter in model.base_model.parameters():
            parameter.requires_grad = not freeze_backbone

        model = model.to(device)

        performance = config["performance"]
        num_workers = performance["num_workers"]
        pin_memory = performance["pin_memory"] and device.type == "cuda"
        precision = performance["mixed_precision"]

        if precision not in {"none", "fp16"}:
            raise ValueError(f"Unsupported mixed precission: {precision}")

        use_amp = device.type == "cuda" and precision == "fp16"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        loader_options = {
            "batch_size": config["training"]["batch_size"],
            "num_workers": num_workers,
            "pin_memory": pin_memory
        }
        if num_workers > 0:
            loader_options["persistent_workers"] = performance["persistent_workers"]
            loader_options["prefetch_factor"] = performance["prefetch_factor"]

        train_dataset = ImageSorterDataset(train_records, transform=transform)
        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            **loader_options
        )

        val_dataset = ImageSorterDataset(val_records, transform=transform)
        val_loader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            **loader_options
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        best_val_loss = float("inf")
        best_epoch = 0
        patience = config["training"]["patience"]

        with(fold_dir / "split.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["path", "station", "year", "magnification", "split"]
            )
            writer.writeheader()
            writer.writerows({**record, "split": "train"} for record in train_records)
            writer.writerows({**record, "split": "validation"} for record in val_records)

        for epoch in range(config["training"]["epochs"]):
            print("Epoch: ", epoch + 1)
            model.train()

            if freeze_backbone:
                model.base_model.eval()

            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp
                ):
                    logits = model(pixel_values=images).logits
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size

                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += batch_size

            print("Training Loss: ", train_loss / total)
            print("Training Accuracy: ", correct / total)

            model.eval()

            val_loss = 0.0
            correct = 0
            total = 0
            errors = []
            record_offset = 0
            validation_results = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=pin_memory)
                    labels = labels.to(device, non_blocking=pin_memory)

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=use_amp
                    ):
                        logits = model(pixel_values=images).logits
                        loss = criterion(logits, labels)

                    probabilities = logits.float().softmax(dim=1)
                    confidences, predictions = probabilities.max(dim=1)

                    batch_size = labels.size(0)
                    val_loss += loss.item() * batch_size
            
                    correct += (predictions == labels).sum().item()
                    total += batch_size

                    for i in range(labels.size(0)):
                        result = {
                            "path": val_records[record_offset + i]["path"],
                            "true_label": labels[i].item(),
                            "predicted_label" : predictions[i].item(),
                            "confidence": confidences[i].item()
                        }

                        validation_results.append(result)

                        if result["true_label"] != result["predicted_label"]:
                            errors.append(result)
                            
                    record_offset += batch_size

            print("Validation Loss: ", val_loss / total)
            print("Validation Accuracy: ", correct / total)
            if args.verbose:
                print("Validation Errors: ", len(errors))
                for error in errors:
                    print(error)

            val_average_loss = val_loss / total
            if val_average_loss < best_val_loss:
                best_val_loss = val_average_loss
                best_epoch = epoch

                model.save_pretrained(fold_dir)
                processor.save_pretrained(fold_dir)

                with(fold_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=["path", "true_label", "predicted_label", "confidence"]
                    )
                    writer.writeheader()
                    writer.writerows(validation_results)

                print(f"Saved best model at epoch {epoch + 1}")
            else:
                if epoch - best_epoch >= patience:
                    print(f"Patience for epoch ran out at epoch: {epoch + 1}")
                    break

    with(save_dir / "loaded_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)
        print(f"Saved loaded configuration at: {save_dir}")
    