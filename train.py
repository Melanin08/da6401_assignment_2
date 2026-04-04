"""Unified Training Script for all tasks"""

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

# ARGUMENTS


def parse_args():
    parser = argparse.ArgumentParser(description="Train Models")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation", "multitask"]
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


# MODEL + LOSS SETUP

def get_model_and_loss(task, device):

    if task == "classification":
        model = VGG11Classifier(num_classes=37).to(device)
        criterion = nn.CrossEntropyLoss()

    elif task == "localization":
        model = VGG11Localizer().to(device)
        mse_loss = nn.MSELoss()
        iou_loss = IoULoss()

        def criterion(pred, target):
            return mse_loss(pred, target) + iou_loss(pred, target)

    elif task == "segmentation":
        model = VGG11UNet(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()

    elif task == "multitask":
        model = MultiTaskPerceptionModel().to(device)

        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        iou = IoULoss()

        def criterion(outputs, batch):
            cls_loss = ce(outputs["classification"], batch["label"])
            loc_loss = mse(outputs["localization"], batch["bbox"]) + iou(outputs["localization"], batch["bbox"])
            seg_loss = ce(outputs["segmentation"], batch["mask"])

            return cls_loss + loc_loss + seg_loss

    else:
        raise ValueError("Invalid task")

    return model, criterion

# TRAIN LOOP


def train_one_epoch(model, loader, optimizer, criterion, task, device):
    model.train()
    total_loss = 0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad()

        if task == "classification":
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

        elif task == "localization":
            targets = batch["bbox"].to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

        elif task == "segmentation":
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

        elif task == "multitask":
            for k in ["label", "bbox", "mask"]:
                batch[k] = batch[k].to(device)
            outputs = model(images)
            loss = criterion(outputs, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# VALIDATION LOOP


def validate(model, loader, criterion, task, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            if task == "classification":
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

            elif task == "localization":
                targets = batch["bbox"].to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

            elif task == "segmentation":
                masks = batch["mask"].to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

            elif task == "multitask":
                for k in ["label", "bbox", "mask"]:
                    batch[k] = batch[k].to(device)
                outputs = model(images)
                loss = criterion(outputs, batch)

            total_loss += loss.item()

    return total_loss / len(loader)


# MAIN

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATA
    train_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="train",
        task=args.task
    )

    val_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="val",
        task=args.task
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # MODEL
    model, criterion = get_model_and_loss(args.task, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # WANDB
    wandb.init(project="da6401_assignment_2", name=f"{args.task}_run")
    wandb.watch(model, log="gradients", log_freq=100)

    # CHECKPOINT
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    # TRAINING
    for epoch in range(args.epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.task, device)
        val_loss = validate(model, val_loader, criterion, args.task, device)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # SAVE BEST
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": val_loss
            }, f"checkpoints/{args.task}.pth")

    print(f"\nTraining Finished. Best Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
