import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import UNet


# =========================
# INIT
# =========================
wandb.init(project="da6401_assignment_2", name="segmentation_eval")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# LOAD DATASET
# =========================
dataset = OxfordIIITPetDataset(
    root="data/oxford-iiit-pet",
    split="test",
    load_mask=True,
    load_bbox=False
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)


# =========================
# LOAD MODEL
# =========================
model = UNet(num_classes=3)

checkpoint = torch.load("checkpoints/unet.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()


# =========================
# METRICS
# =========================
def pixel_accuracy(pred, target):
    pred = pred.argmax(1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def dice_score(pred, target):
    pred = pred.argmax(1)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred == target).sum().item()
    return (2 * intersection) / (pred.numel() + target.numel() + 1e-8)


# =========================
# W&B TABLE
# =========================
table = wandb.Table(columns=["Image", "Ground Truth", "Prediction"])

pixel_acc_list = []
dice_list = []


# =========================
# LOOP
# =========================
count = 0

for img, _, mask in loader:
    if count >= 5:
        break

    img = img.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        pred = model(img)

    # metrics
    pa = pixel_accuracy(pred, mask)
    dc = dice_score(pred, mask)

    pixel_acc_list.append(pa)
    dice_list.append(dc)

    # prepare images
    img_np = img[0].cpu().permute(1, 2, 0).numpy()
    gt_np = mask[0].cpu().numpy()
    pred_np = pred.argmax(1)[0].cpu().numpy()

    table.add_data(
        wandb.Image(img_np, caption="Original"),
        wandb.Image(gt_np, caption="Ground Truth"),
        wandb.Image(pred_np, caption="Prediction")
    )

    count += 1


# =========================
# LOG
# =========================
wandb.log({
    "Segmentation Samples": table,
    "Pixel Accuracy": np.mean(pixel_acc_list),
    "Dice Score": np.mean(dice_list)
})

wandb.finish()

print("Done logging segmentation results!")
