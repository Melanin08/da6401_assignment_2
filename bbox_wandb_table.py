
# OBJECT DETECTION: CONFIDENCE & IoU VISUALIZATION 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer


# Initialize Weights & Biases

# This will log results (images, IoU, confidence) to W&B dashboard
wandb.init(project="da6401_assignment_2", name="object_detection_confidence_iou")


# Define Paths

ckpt_path = "checkpoints/localizer.pth"       
data_root = "data/oxford-iiit-pet"             


# Check if files exist

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

if not os.path.exists(data_root):
    raise FileNotFoundError(f"Dataset root not found: {data_root}")



# Device setup (GPU / CPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Load trained localization model

model = VGG11Localizer()

checkpoint = torch.load(ckpt_path, map_location=device)

# Handle both checkpoint formats
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()   # set model to inference mode


# Load dataset (TEST split)

dataset = OxfordIIITPetDataset(
    root=data_root,
    split="test",
    task="localization"
)

print("Dataset size:", len(dataset))

# Safety check
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check dataset path or bbox loading.")


# DataLoader (1 image at a time for visualization)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


# HELPER FUNCTIONS


# Convert [cx, cy, w, h] → [x1, y1, x2, y2]
def xywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


# Compute Intersection over Union (IoU)
def compute_iou(box1, box2, eps=1e-6):
    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)

    # Intersection area
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # Union area
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])

    union = area1 + area2 - inter_area + eps

    return inter_area / union


# Convert tensor → image for visualization
def tensor_to_img(t):
    img = t.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


# Draw predicted and ground truth bounding boxes
def draw_boxes(image_tensor, pred_box, gt_box, iou, confidence):
    img = tensor_to_img(image_tensor)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)

    # Predicted box → RED
    px1, py1, px2, py2 = xywh_to_xyxy(pred_box)
    ax.add_patch(plt.Rectangle(
        (px1, py1),
        px2 - px1,
        py2 - py1,
        edgecolor="red",
        linewidth=3,
        fill=False
    ))

    # Ground truth box → GREEN (dashed)
    gx1, gy1, gx2, gy2 = xywh_to_xyxy(gt_box)
    ax.add_patch(plt.Rectangle(
        (gx1, gy1),
        gx2 - gx1,
        gy2 - gy1,
        edgecolor="lime",
        linewidth=3,
        fill=False,
        linestyle="--"
    ))

    # Show IoU + confidence
    ax.set_title(f"Confidence: {confidence:.2f} | IoU: {iou:.2f}")
    ax.axis("off")

    # Convert plot → image array for W&B
    fig.canvas.draw()
    img_out = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return img_out


# Proxy confidence (since model does not output real confidence)
def proxy_confidence(pred_box):
    cx, cy, w, h = pred_box

    # Size-based score
    area = max(w, 0.0) * max(h, 0.0)
    area_score = np.clip(area / (224.0 * 224.0), 0.0, 1.0)

    # Center-based score
    cx_score = 1.0 - min(abs(cx - 112.0) / 112.0, 1.0)
    cy_score = 1.0 - min(abs(cy - 112.0) / 112.0, 1.0)
    center_score = 0.5 * (cx_score + cy_score)

    # Final confidence
    conf = 0.5 * area_score + 0.5 * center_score
    return float(np.clip(conf, 0.0, 1.0))


# CREATE W&B TABLE

table = wandb.Table(columns=["Index", "Image", "Confidence", "IoU", "Failure Case"])


# Store results for later logging
results = []
lowest_iou = 999.0
failure_index = -1


# RUN INFERENCE ON 10 TEST IMAGES

for idx, batch in enumerate(loader):
    if idx >= 10:
        break

    image = batch["image"].to(device)
    gt_box = batch["bbox"][0].cpu().numpy()

    # Model prediction
    with torch.no_grad():
        pred_box = model(image)[0].cpu().numpy()

    # Compute IoU and confidence
    iou = compute_iou(pred_box, gt_box)
    confidence = proxy_confidence(pred_box)

    # Print values (debug + understanding)
    print(f"Sample {idx}")
    print("GT box:", gt_box)
    print("Pred box:", pred_box)
    print("IoU:", iou)
    print("Confidence:", confidence)
    print("-" * 50)

    results.append((idx, batch, pred_box, gt_box, iou, confidence))

    # Track worst case (lowest IoU)
    if iou < lowest_iou:
        lowest_iou = iou
        failure_index = idx


# LOG RESULTS TO W&B TABLE

for idx, batch, pred_box, gt_box, iou, confidence in results:
    note = ""

    # Mark failure case
    if idx == failure_index:
        note = "Failure Case"

    img_vis = draw_boxes(batch["image"][0], pred_box, gt_box, iou, confidence)

    table.add_data(
        idx,
        wandb.Image(img_vis),
        confidence,
        iou,
        note
    )


# Upload table to W&B
wandb.log({"Detection_Table": table})


# Final output
print(f"Failure case: sample {failure_index} with IoU = {lowest_iou:.2f}")

wandb.finish()
print("Done. Check W&B.")
