"""Unified training script for all tasks."""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


# ARGUMENT PARSING
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train visual perception models")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation", "multitask"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()

# DATA LOADING

def build_dataloaders(data_root, task, batch_size):
    """Create train and validation dataloaders for given task."""
    train_dataset = OxfordIIITPetDataset(root=data_root, split="train", task=task)
    val_dataset = OxfordIIITPetDataset(root=data_root, split="val", task=task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# MODEL + LOSS

def build_model_and_loss(task, device):
    """Initialize model and corresponding loss function."""

    # -------- Classification --------
    if task == "classification":
        model = VGG11Classifier(num_classes=37).to(device)
        criterion = nn.CrossEntropyLoss()

    # -------- Localization --------
    elif task == "localization":
        model = VGG11Localizer().to(device)

        mse_loss = nn.MSELoss()
        iou_loss = IoULoss()

        # Combined loss: regression + IoU
        def criterion(pred_boxes, target_boxes):
            return mse_loss(pred_boxes, target_boxes) + iou_loss(pred_boxes, target_boxes)

    # -------- Segmentation --------
    elif task == "segmentation":
        model = VGG11UNet(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()

    # -------- Multi-task --------
    elif task == "multitask":
        model = MultiTaskPerceptionModel().to(device)

        cls_loss_fn = nn.CrossEntropyLoss()
        box_mse_loss = nn.MSELoss()
        box_iou_loss = IoULoss()
        seg_loss_fn = nn.CrossEntropyLoss()

        # Combined loss across all tasks
        def criterion(outputs, batch):
            cls_loss = cls_loss_fn(outputs["classification"], batch["label"])
            loc_loss = box_mse_loss(outputs["localization"], batch["bbox"]) + box_iou_loss(
                outputs["localization"], batch["bbox"]
            )
            seg_loss = seg_loss_fn(outputs["segmentation"], batch["mask"])
            return cls_loss + loc_loss + seg_loss

    else:
        raise ValueError("Invalid task")

    return model, criterion


# IOU METRIC (LOCALIZATION)

def box_iou_xywh(pred_boxes, target_boxes, eps=1e-6):
    """Compute IoU for bounding boxes."""
    pred_w = torch.clamp(pred_boxes[:, 2], min=0.0)
    pred_h = torch.clamp(pred_boxes[:, 3], min=0.0)

    target_w = torch.clamp(target_boxes[:, 2], min=0.0)
    target_h = torch.clamp(target_boxes[:, 3], min=0.0)

    pred_x1 = pred_boxes[:, 0] - pred_w / 2
    pred_y1 = pred_boxes[:, 1] - pred_h / 2
    pred_x2 = pred_boxes[:, 0] + pred_w / 2
    pred_y2 = pred_boxes[:, 1] + pred_h / 2

    target_x1 = target_boxes[:, 0] - target_w / 2
    target_y1 = target_boxes[:, 1] - target_h / 2
    target_x2 = target_boxes[:, 0] + target_w / 2
    target_y2 = target_boxes[:, 1] + target_h / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, 0) * torch.clamp(inter_y2 - inter_y1, 0)
    union = (
        (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        + (target_x2 - target_x1) * (target_y2 - target_y1)
        - inter_area
        + eps
    )

    return inter_area / union


# TRAIN FUNCTION

def train_one_epoch(model, loader, optimizer, criterion, task, device):
    """Run one training epoch."""
    model.train()

    total_loss = 0.0
    total_correct, total_samples = 0, 0
    total_pixels_correct, total_pixels = 0, 0
    total_iou, total_iou_batches = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        optimizer.zero_grad()

        # ---- Classification ----
        if task == "classification":
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        # ---- Localization ----
        elif task == "localization":
            targets = batch["bbox"].to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_iou += box_iou_xywh(outputs, targets).mean().item()
            total_iou_batches += 1

        # ---- Segmentation ----
        elif task == "segmentation":
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            total_pixels_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

        # ---- Multitask ----
        elif task == "multitask":
            batch["label"] = batch["label"].to(device)
            batch["bbox"] = batch["bbox"].to(device)
            batch["mask"] = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, batch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    metrics = {"loss": total_loss / len(loader)}

    if total_samples > 0:
        metrics["accuracy"] = total_correct / total_samples
    if total_pixels > 0:
        metrics["pixel_acc"] = total_pixels_correct / total_pixels
    if total_iou_batches > 0:
        metrics["iou"] = total_iou / total_iou_batches

    return metrics


# CHECKPOINT NAME FIX

def get_save_path(task):
    """Return correct filename required by autograder."""
    if task == "classification":
        return "checkpoints/classifier.pth"
    if task == "localization":
        return "checkpoints/localizer.pth"
    if task == "segmentation":
        return "checkpoints/unet.pth"
    if task == "multitask":
        return "checkpoints/multitask.pth"
    raise ValueError("Invalid task")


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = build_dataloaders(args.data_root, args.task, args.batch_size)
    model, criterion = build_model_and_loss(args.task, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)

    wandb.init(project="da6401_assignment_2", name=f"{args.task}_run")

    best_val_loss = float("inf")
    save_path = get_save_path(args.task)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.task, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {train_metrics['loss']:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"]
        })

        # Save best model
        if train_metrics["loss"] < best_val_loss:
            best_val_loss = train_metrics["loss"]

            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_val_loss,
            }, save_path)

    print("Saved to:", save_path)


if __name__ == "__main__":
    main()
