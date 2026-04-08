import os
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet


# =========================
# INIT
# =========================
wandb.init(project="da6401_assignment_2", name="segmentation_eval")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# PATHS
# =========================
data_root = "data/oxford-iiit-pet"
ckpt_path = "checkpoints/unet.pth"

if not os.path.exists(data_root):
    raise FileNotFoundError(f"Dataset root not found: {data_root}")

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


# =========================
# LOAD DATASET
# =========================
dataset = OxfordIIITPetDataset(
    root=data_root,
    split="test",
    task="segmentation"
)

print("Dataset size:", len(dataset))
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check dataset path.")

loader = DataLoader(dataset, batch_size=1, shuffle=True)


# =========================
# LOAD MODEL
# =========================
model = VGG11UNet(num_classes=3)

checkpoint = torch.load(ckpt_path, map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()


# =========================
# METRICS
# =========================
def pixel_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def dice_score(pred, target, num_classes=3, eps=1e-8):
    pred = pred.argmax(dim=1)
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum().item()
        denom = pred_cls.sum().item() + target_cls.sum().item()

        if denom > 0:
            dice = (2.0 * intersection + eps) / (denom + eps)
            dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return float(np.mean(dice_scores))


def tensor_to_display_image(image_tensor):
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


# =========================
# COLORIZE MASKS
# =========================
def colorize_mask(mask):
    """
    Convert mask with labels {0,1,2} into RGB colors.
    0 = black
    1 = green
    2 = red
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    color_mask[mask == 0] = [0, 0, 0]        # background
    color_mask[mask == 1] = [0, 255, 0]      # class 1
    color_mask[mask == 2] = [255, 0, 0]      # class 2

    return color_mask


# =========================
# W&B TABLE
# =========================
table = wandb.Table(columns=["Original Image", "Ground Truth Trimap", "Predicted Trimap"])

pixel_acc_list = []
dice_list = []


# =========================
# LOOP
# =========================
count = 0

with torch.no_grad():
    for batch in loader:
        if count >= 5:
            break

        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        pred = model(img)

        # Metrics
        pa = pixel_accuracy(pred, mask)
        dc = dice_score(pred, mask)

        pixel_acc_list.append(pa)
        dice_list.append(dc)

        # Visuals
        img_np = tensor_to_display_image(img[0])
        gt_np = batch["mask"][0].cpu().numpy()
        pred_np = pred.argmax(dim=1)[0].cpu().numpy()

        gt_color = colorize_mask(gt_np)
        pred_color = colorize_mask(pred_np)

        table.add_data(
            wandb.Image(img_np, caption="Original"),
            wandb.Image(gt_color, caption="Ground Truth Trimap"),
            wandb.Image(pred_color, caption="Predicted Trimap"),
        )

        count += 1


# =========================
# LOG
# =========================
mean_pa = float(np.mean(pixel_acc_list))
mean_dice = float(np.mean(dice_list))

wandb.log({
    "Segmentation Samples": table,
    "Validation Pixel Accuracy": mean_pa,
    "Validation Dice Score": mean_dice,
})

print(f"Validation Pixel Accuracy: {mean_pa:.4f}")
print(f"Validation Dice Score: {mean_dice:.4f}")

wandb.finish()
print("Done logging segmentation results!")
