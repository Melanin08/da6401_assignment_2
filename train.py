"""Unified training script for all tasks."""

import argparse
import os
import random
import time

import numpy as np
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


# Parse command-line inputs
def parse_args():
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
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability for classification")
    parser.add_argument(
        "--transfer_mode",
        type=str,
        default="full",
        choices=["strict", "partial", "full"],
        help="Transfer learning strategy for segmentation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible experiments")
    parser.add_argument(
        "--no_batchnorm",
        action="store_false",
        dest="batchnorm",
        help="Disable BatchNorm layers in the backbone",
    )
    parser.set_defaults(batchnorm=True)

    return parser.parse_args()


# Build train/validation data loaders
def build_dataloaders(data_root, task, batch_size):
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


# Build model + loss for each task
def build_model_and_loss(task, device, use_batchnorm=True, dropout_p=0.5):
    if task == "classification":
        model = VGG11Classifier(
            num_classes=37,
            use_batchnorm=use_batchnorm,
            dropout_p=dropout_p
        ).to(device)
        criterion = nn.CrossEntropyLoss()

    elif task == "localization":
        model = VGG11Localizer(use_batchnorm=use_batchnorm).to(device)

        mse_loss = nn.MSELoss()
        iou_loss = IoULoss()

        # Required loss for localization
        def criterion(pred_boxes, target_boxes):
            return mse_loss(pred_boxes, target_boxes) + iou_loss(pred_boxes, target_boxes)

    elif task == "segmentation":
        model = VGG11UNet(num_classes=3, use_batchnorm=use_batchnorm).to(device)
        criterion = nn.CrossEntropyLoss()

    elif task == "multitask":
        model = MultiTaskPerceptionModel(use_batchnorm=use_batchnorm).to(device)

        cls_loss_fn = nn.CrossEntropyLoss()
        box_mse_loss = nn.MSELoss()
        box_iou_loss = IoULoss()
        seg_loss_fn = nn.CrossEntropyLoss()

        # Weighted multitask loss so localization MSE does not dominate
        def criterion(outputs, batch):
            cls_loss = cls_loss_fn(outputs["classification"], batch["label"])
            loc_loss = box_mse_loss(outputs["localization"], batch["bbox"]) + box_iou_loss(
                outputs["localization"], batch["bbox"]
            )
            seg_loss = seg_loss_fn(outputs["segmentation"], batch["mask"])
            return cls_loss + 0.001 * loc_loss + seg_loss

    else:
        raise ValueError("Invalid task")

    return model, criterion


def apply_transfer_learning_strategy(model, task, transfer_mode):
    """
    Applies transfer learning strategy only for segmentation.
    strict  -> freeze full encoder
    partial -> freeze early encoder blocks, train deeper blocks + decoder
    full    -> train everything
    """
    if task != "segmentation":
        return

    if not hasattr(model, "encoder"):
        return

    if transfer_mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif transfer_mode == "strict":
        for param in model.parameters():
            param.requires_grad = True
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif transfer_mode == "partial":
        for param in model.parameters():
            param.requires_grad = True

        # Freeze early blocks only
        for block in [model.encoder.block1, model.encoder.block2, model.encoder.block3]:
            for param in block.parameters():
                param.requires_grad = False


# IoU metric for box evaluation
# Boxes are [x_center, y_center, width, height]
def box_iou_xywh(pred_boxes, target_boxes, eps=1e-6):
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


# Run one epoch
# train=True  -> training mode
# train=False -> validation mode
def run_one_epoch(model, loader, optimizer, criterion, task, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0

    total_pixels_correct = 0
    total_pixels = 0

    total_iou = 0.0
    total_iou_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in loader:
            images = batch["image"].to(device)

            if train:
                optimizer.zero_grad()

            # ----- Classification -----
            if task == "classification":
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            # ----- Localization -----
            elif task == "localization":
                targets = batch["bbox"].to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                total_iou += box_iou_xywh(outputs, targets).mean().item()
                total_iou_batches += 1

            # ----- Segmentation -----
            elif task == "segmentation":
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                total_pixels_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

            # ----- Multitask -----
            elif task == "multitask":
                labels = batch["label"].to(device)
                boxes = batch["bbox"].to(device)
                masks = batch["mask"].to(device)

                multitask_batch = {
                    "label": labels,
                    "bbox": boxes,
                    "mask": masks,
                }

                outputs = model(images)
                loss = criterion(outputs, multitask_batch)

                cls_preds = torch.argmax(outputs["classification"], dim=1)
                total_correct += (cls_preds == labels).sum().item()
                total_samples += labels.size(0)

                seg_preds = torch.argmax(outputs["segmentation"], dim=1)
                total_pixels_correct += (seg_preds == masks).sum().item()
                total_pixels += masks.numel()

                total_iou += box_iou_xywh(outputs["localization"], boxes).mean().item()
                total_iou_batches += 1

            else:
                raise ValueError("Invalid task")

            if train:
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


def register_third_conv_hook(model, activation_container):
    if not hasattr(model, "encoder"):
        return None

    def hook(module, inp, out):
        activation_container["third_conv"] = out.detach().cpu()

    return model.encoder.block3[0].register_forward_hook(hook)


def capture_third_conv_activation(model, fixed_images):
    activations = {}
    handle = register_third_conv_hook(model, activations)

    model.eval()
    with torch.no_grad():
        _ = model(fixed_images)

    if handle is not None:
        handle.remove()

    if "third_conv" not in activations:
        return {}

    act = activations["third_conv"].cpu().numpy()
    return {
        "third_conv_activation_hist": wandb.Histogram(act),
        "third_conv_activation_mean": act.mean().item() if hasattr(act, "mean") else float(act.mean()),
        "third_conv_activation_std": act.std().item() if hasattr(act, "std") else float(act.std()),
    }


# Required checkpoint filenames
def get_save_path(task):
    if task == "classification":
        return "checkpoints/classifier.pth"
    if task == "localization":
        return "checkpoints/localizer.pth"
    if task == "segmentation":
        return "checkpoints/unet.pth"
    if task == "multitask":
        return "checkpoints/multitask.pth"
    raise ValueError("Invalid task")


# Main training entry
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = build_dataloaders(args.data_root, args.task, args.batch_size)
    model, criterion = build_model_and_loss(
        args.task,
        device,
        use_batchnorm=args.batchnorm,
        dropout_p=args.dropout_p
    )

    apply_transfer_learning_strategy(model, args.task, args.transfer_mode)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)

    fixed_batch = next(iter(val_loader))
    fixed_images = fixed_batch["image"].to(device)

    wandb.init(
        project="da6401_assignment_2",
        name=f"{args.task}_{args.transfer_mode}_bn_{'on' if args.batchnorm else 'off'}_dropout_{args.dropout_p}_lr_{args.lr}",
        group=f"{args.task}_bn_comparison",
        config={
            "task": args.task,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "batchnorm": args.batchnorm,
            "dropout_p": args.dropout_p,
            "transfer_mode": args.transfer_mode,
            "seed": args.seed,
        },
    )
    wandb.watch(model, log="all", log_freq=100)

    best_val_loss = float("inf")
    save_path = get_save_path(args.task)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_metrics = run_one_epoch(
            model, train_loader, optimizer, criterion, args.task, device, train=True
        )
        val_metrics = run_one_epoch(
            model, val_loader, optimizer, criterion, args.task, device, train=False
        )

        epoch_time = time.time() - epoch_start

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
            "epoch_time_sec": epoch_time,
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

        activation_logs = capture_third_conv_activation(model, fixed_images)
        wandb_log.update(activation_logs)

        wandb.log(wandb_log)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_val_loss,
                },
                save_path,
            )

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
