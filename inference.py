"""Inference and evaluation"""

import argparse
import torch
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation", "multitask"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()

# LOAD MODEL 

def load_model(task, model_path, device):

    if task == "classification":
        model = VGG11Classifier(num_classes=37)

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    elif task == "localization":
        model = VGG11Localizer()

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    elif task == "segmentation":
        model = VGG11UNet()

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    elif task == "multitask":
        model = MultiTaskPerceptionModel(load_pretrained=True)

    else:
        raise ValueError("Invalid task")

    model.to(device)
    model.eval()

    return model

# CLASSIFICATION

def evaluate_classification(model, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Classification Accuracy: {acc:.4f}")


# LOCALIZATION

def evaluate_localization(model, loader, device):
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["bbox"].to(device)

            preds = model(images)
            loss = torch.mean((preds - targets) ** 2)

            total_loss += loss.item()

    print(f"Localization MSE: {total_loss / len(loader):.4f}")


# SEGMENTATION

def evaluate_segmentation(model, loader, device):
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

    acc = correct_pixels / total_pixels
    print(f"Segmentation Pixel Accuracy: {acc:.4f}")

# MULTITASK

def evaluate_multitask(model, loader, device):
    correct_cls = 0
    total_cls = 0
    total_loc_loss = 0
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)

            cls_out = outputs["classification"]
            box_out = outputs["localization"]
            seg_out = outputs["segmentation"]

            # Classification
            preds = cls_out.argmax(dim=1)
            correct_cls += (preds == labels).sum().item()
            total_cls += labels.size(0)

            # Localization
            loc_loss = torch.mean((box_out - boxes) ** 2)
            total_loc_loss += loc_loss.item()

            # Segmentation
            seg_preds = seg_out.argmax(dim=1)
            correct_pixels += (seg_preds == masks).sum().item()
            total_pixels += masks.numel()

    cls_acc = correct_cls / total_cls
    loc_loss = total_loc_loss / len(loader)
    seg_acc = correct_pixels / total_pixels

    print(f"Multitask Classification Acc: {cls_acc:.4f}")
    print(f"Multitask Localization MSE: {loc_loss:.4f}")
    print(f"Multitask Segmentation Acc: {seg_acc:.4f}")

# MAIN

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="test",
        task=args.task,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.task, args.model_path, device)

    if args.task == "classification":
        evaluate_classification(model, loader, device)

    elif args.task == "localization":
        evaluate_localization(model, loader, device)

    elif args.task == "segmentation":
        evaluate_segmentation(model, loader, device)

    elif args.task == "multitask":
        evaluate_multitask(model, loader, device)


if __name__ == "__main__":
    main()