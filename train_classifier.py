"""Training script for classification """

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # DATASET
    train_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="train",
        task="classification"
    )

    val_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="val",
        task="classification"
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

    model = VGG11Classifier(num_classes=37).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # WANDB
   
    wandb.init(project="da6401_assignment_2", name="vgg11_classifier")
    wandb.watch(model, log="gradients", log_freq=100)

   
    # CHECKPOINT DIR
    os.makedirs("checkpoints", exist_ok=True)

    best_acc = 0.0

    # TRAINING LOOP
    for epoch in range(args.epochs):

        # TRAIN 
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # VALIDATION 
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # PRINT
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # WANDB LOG
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": val_acc
            }, "checkpoints/classifier.pth")

    # OPTIONAL: SAVE LAST MODEL once after training to avoid partial write issues.
    last_checkpoint_path = os.path.join("checkpoints", "classifier_last.pth")
    tmp_checkpoint_path = last_checkpoint_path + ".tmp"
    torch.save({
        "state_dict": model.state_dict()
    }, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, last_checkpoint_path)

    print(f"\nTraining Finished. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()