"""Training script for segmentation """

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATASET
    train_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="train",
        task="segmentation"
    )

    val_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="val",
        task="segmentation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # MODEL
    model = VGG11UNet(num_classes=3).to(device)

    # LOSS (pixel-wise classification)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # WANDB
    wandb.init(project="da6401_assignment_2", name="vgg11_unet")
    wandb.watch(model, log="gradients", log_freq=100)

    # CHECKPOINT DIR
    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")

    # TRAINING LOOP
    for epoch in range(args.epochs):

        # TRAIN
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)  # [B, H, W]

            optimizer.zero_grad()

            outputs = model(images)  # [B, C, H, W]

            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # PRINT
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # WANDB LOG
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # SAVE BEST MODEL
        if val_loss < best_loss:
            best_loss = val_loss

            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": val_loss
            }, "checkpoints/unet.pth")

        # OPTIONAL: SAVE LAST
        torch.save({
            "state_dict": model.state_dict()
        }, "checkpoints/unet_last.pth")

    print(f"\nTraining Finished. Best Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()