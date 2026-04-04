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


def build_dataloaders(data_root, task, batch_size):
    """Build train and validation dataloaders."""
    train_dataset = OxfordIIITPetDataset(
        root=data_root,
        split="train",
        task=task,
    )

    val_dataset = OxfordIIITPetDataset(
        root=data_root,
        split="val",
        task=task,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def build_model_and_loss(task, device):
    """Create model and loss function(s) for the selected task."""
    if task == "classification":
        model = VGG11Classifier(num_classes=37).to(device)
        criterion = nn.CrossEntropyLoss()

    elif task == "localization":
        model = VGG11Localizer().to(device)
        mse_loss = nn.MSELoss()
        iou_loss = IoULoss()

        def criterion(pred_boxes, target_boxes):
            return mse_loss(pred_boxes, target_boxes) + iou_loss(pred_boxes, target_boxes)

    elif task == "segmentation":
        model = VGG11UNet(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()

    elif task == "multitask":
        model = MultiTaskPerceptionModel().to(device)

        cls_loss_fn = nn.CrossEntropyLoss()
        box_mse_loss = nn.MSELoss()
        box_iou_loss = IoULoss()
        seg_loss_fn = nn.CrossEntropyLoss()

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


def box_iou_xywh(pred_boxes, target_boxes, eps=1e-6):
    """
    Compute IoU for boxes in [x_center, y_center, width, height] format.
    This is used only as a metric.
    """
    pred_w = torch.clamp(pred_boxes[:, 2], min=0.0)
    pred_h = torch.clamp(pred_boxes[:, 3], min=0.0)
    target_w = torch.clamp(target_boxes[:, 2], min=0.0)
    target_h = torch.clamp(target_boxes[:, 3], min=0.0)

    pred_x1 = pred_boxes[:, 0] - pred_w / 2.0
    pred_y1 = pred_boxes[:, 1] - pred_h / 2.0
    pred_x2 = pred_boxes[:, 0] + pred_w / 2.0
    pred_y2 = pred_boxes[:, 1] + pred_h / 2.0

    target_x1 = target_boxes[:, 0] - target_w / 2.0
    target_y1 = target_boxes[:, 1] - target_h / 2.0
    target_x2 = target_boxes[:, 0] + target_w / 2.0
    target_y2 = target_boxes[:, 1] + target_h / 2.0

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    pred_area = torch.clamp(pred_x2 - pred_x1, min=0.0) * torch.clamp(pred_y2 - pred_y1, min=0.0)
    target_area = torch.clamp(target_x2 - target_x1, min=0.0) * torch.clamp(target_y2 - target_y1, min=0.0)

    union_area = pred_area + target_area - inter_area + eps
    return inter_area / union_area


def train_one_epoch(model, loader, optimizer, criterion, task, device):
    """Train for one epoch and return metrics."""
    model.train()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0

    total_pixels_correct = 0
    total_pixels = 0

    total_iou = 0.0
    total_iou_batches = 0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad()

        if task == "classification":
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        elif task == "localization":
            target_boxes = batch["bbox"].to(device)

            outputs = model(images)
            loss = criterion(outputs, target_boxes)

            batch_iou = box_iou_xywh(outputs, target_boxes).mean().item()
            total_iou += batch_iou
            total_iou_batches += 1

        elif task == "segmentation":
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            total_pixels_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

        elif task == "multitask":
            batch["label"] = batch["label"].to(device)
            batch["bbox"] = batch["bbox"].to(device)
            batch["mask"] = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, batch)

            cls_preds = torch.argmax(outputs["classification"], dim=1)
            total_correct += (cls_preds == batch["label"]).sum().item()
            total_samples += batch["label"].size(0)

            seg_preds = torch.argmax(outputs["segmentation"], dim=1)
            total_pixels_correct += (seg_preds == batch["mask"]).sum().item()
            total_pixels += batch["mask"].numel()

            batch_iou = box_iou_xywh(outputs["localization"], batch["bbox"]).mean().item()
            total_iou += batch_iou
            total_iou_batches += 1

        else:
            raise ValueError("Invalid task")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    metrics = {
        "loss": total_loss / len(loader),
    }

    if total_samples > 0:
        metrics["accuracy"] = total_correct / total_samples

    if total_pixels > 0:
        metrics["pixel_acc"] = total_pixels_correct / total_pixels

    if total_iou_batches > 0:
        metrics["iou"] = total_iou / total_iou_batches

    return metrics


def validate_one_epoch(model, loader, criterion, task, device):
    """Run validation for one epoch and return metrics."""
    model.eval()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0

    total_pixels_correct = 0
    total_pixels = 0

    total_iou = 0.0
    total_iou_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            if task == "classification":
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            elif task == "localization":
                target_boxes = batch["bbox"].to(device)

                outputs = model(images)
                loss = criterion(outputs, target_boxes)

                batch_iou = box_iou_xywh(outputs, target_boxes).mean().item()
                total_iou += batch_iou
                total_iou_batches += 1

            elif task == "segmentation":
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                total_pixels_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

            elif task == "multitask":
                batch["label"] = batch["label"].to(device)
                batch["bbox"] = batch["bbox"].to(device)
                batch["mask"] = batch["mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, batch)

                cls_preds = torch.argmax(outputs["classification"], dim=1)
                total_correct += (cls_preds == batch["label"]).sum().item()
                total_samples += batch["label"].size(0)

                seg_preds = torch.argmax(outputs["segmentation"], dim=1)
                total_pixels_correct += (seg_preds == batch["mask"]).sum().item()
                total_pixels += batch["mask"].numel()

                batch_iou = box_iou_xywh(outputs["localization"], batch["bbox"]).mean().item()
                total_iou += batch_iou
                total_iou_batches += 1

            else:
                raise ValueError("Invalid task")

            total_loss += loss.item()

    metrics = {
        "loss": total_loss / len(loader),
    }

    if total_samples > 0:
        metrics["accuracy"] = total_correct / total_samples

    if total_pixels > 0:
        metrics["pixel_acc"] = total_pixels_correct / total_pixels

    if total_iou_batches > 0:
        metrics["iou"] = total_iou / total_iou_batches

    return metrics


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = build_dataloaders(args.data_root, args.task, args.batch_size)
    model, criterion = build_model_and_loss(args.task, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)

    wandb.init(project="da6401_assignment_2", name=f"{args.task}_run")
    wandb.watch(model, log="gradients", log_freq=100)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.task, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, args.task, device)

        print(f"Epoch [{epoch + 1}/{args.epochs}]")

        if args.task == "classification":
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
            )

        elif args.task == "localization":
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}"
            )

        elif args.task == "segmentation":
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | Train Pixel Acc: {train_metrics['pixel_acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Pixel Acc: {val_metrics['pixel_acc']:.4f}"
            )

        elif args.task == "multitask":
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Cls Acc: {train_metrics['accuracy']:.4f} | "
                f"Train IoU: {train_metrics['iou']:.4f} | "
                f"Train Pixel Acc: {train_metrics['pixel_acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Cls Acc: {val_metrics['accuracy']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"Val Pixel Acc: {val_metrics['pixel_acc']:.4f}"
            )

        wandb_log = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }

        if "accuracy" in train_metrics:
            wandb_log["train_accuracy"] = train_metrics["accuracy"]
            wandb_log["val_accuracy"] = val_metrics["accuracy"]

        if "pixel_acc" in train_metrics:
            wandb_log["train_pixel_acc"] = train_metrics["pixel_acc"]
            wandb_log["val_pixel_acc"] = val_metrics["pixel_acc"]

        if "iou" in train_metrics:
            wandb_log["train_iou"] = train_metrics["iou"]
            wandb_log["val_iou"] = val_metrics["iou"]

        wandb.log(wandb_log)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_val_loss,
                },
                f"checkpoints/{args.task}.pth",
            )

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
