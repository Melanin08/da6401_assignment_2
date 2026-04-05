"""W&B experiments for BatchNorm and Dropout analysis."""

import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.layers import CustomDropout


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG11BackboneToggleBN(nn.Module):
    def __init__(self, in_channels=3, use_bn=True):
        super().__init__()
        self.block1 = ConvBlockBN(in_channels, 64, use_bn)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = ConvBlockBN(64, 128, use_bn)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3a = ConvBlockBN(128, 256, use_bn)
        self.conv3b = ConvBlockBN(256, 256, use_bn)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.block4 = nn.Sequential(
            ConvBlockBN(256, 512, use_bn),
            ConvBlockBN(512, 512, use_bn),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.block5 = nn.Sequential(
            ConvBlockBN(512, 512, use_bn),
            ConvBlockBN(512, 512, use_bn),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))
        return x


class VGG11ClassifierToggleBN(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5, use_bn=True):
        super().__init__()
        self.encoder = VGG11BackboneToggleBN(in_channels=in_channels, use_bn=use_bn)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--use_batchnorm", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="bn_dropout_run")
    return parser.parse_args()


def get_loaders(data_root, batch_size):
    train_dataset = OxfordIIITPetDataset(root=data_root, split="train", task="classification")
    val_dataset = OxfordIIITPetDataset(root=data_root, split="val", task="classification")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_dataset


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

    return {
        "loss": total_loss / len(loader),
        "accuracy": total_correct / total_samples,
    }


def log_activation_histogram(model, dataset, device, run_prefix):
    model.eval()

    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)

    captured = {}

    def hook_fn(module, inp, out):
        captured["act"] = out.detach().cpu().flatten().numpy()

    handle = model.encoder.conv3a.block[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image)

    handle.remove()

    if "act" in captured:
        wandb.log({
            f"{run_prefix}_conv3_activation_hist": wandb.Histogram(captured["act"])
        })


def main():
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, val_dataset = get_loaders(args.data_root, args.batch_size)

    model = VGG11ClassifierToggleBN(
        num_classes=37,
        dropout_p=args.dropout_p,
        use_bn=bool(args.use_batchnorm),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    wandb.init(
        project="da6401_assignment_2",
        name=args.run_name,
        config=vars(args),
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        start = time.time()

        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        epoch_time = time.time() - start

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time_sec": epoch_time,
        })

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if epoch == args.epochs - 1:
            log_activation_histogram(model, val_dataset, device, "same_input")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {"state_dict": model.state_dict(), "best_metric": best_val_loss},
                f"checkpoints/{args.run_name}.pth",
            )

    wandb.finish()


if __name__ == "__main__":
    main()