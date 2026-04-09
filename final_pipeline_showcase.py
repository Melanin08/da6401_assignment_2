import os
import requests
import torch
import numpy as np
import wandb

from PIL import Image, ImageDraw
from torchvision import transforms
from io import BytesIO

from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

# W&B INIT

wandb.init(project="da6401_assignment_2", name="final_pipeline_showcase")

# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# PATHS

localizer_ckpt = "checkpoints/localizer.pth"
segmenter_ckpt = "checkpoints/unet.pth"

for path in [localizer_ckpt, segmenter_ckpt]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")


# LOAD MODELS

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


localizer = load_checkpoint(VGG11Localizer(), localizer_ckpt)
segmenter = load_checkpoint(VGG11UNet(num_classes=3), segmenter_ckpt)


# PREPROCESS

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def denorm(img):
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.clip(img * std + mean, 0, 1)



# BOX UTILS

def xywh_to_xyxy(box):
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def resize_box(box, ow, oh):
    cx, cy, w, h = box
    return [cx * ow / 224.0, cy * oh / 224.0, w * ow / 224.0, h * oh / 224.0]


def clamp(x1, y1, x2, y2, w, h):
    return (
        max(0, min(w - 1, int(round(x1)))),
        max(0, min(h - 1, int(round(y1)))),
        max(0, min(w - 1, int(round(x2)))),
        max(0, min(h - 1, int(round(y2))))
    )



# SEGMENTATION VIS

def colorize(mask):
    c = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    c[mask == 0] = [0, 0, 0]
    c[mask == 1] = [0, 255, 0]
    c[mask == 2] = [255, 0, 0]
    return c


def overlay(img, mask, alpha=0.4):
    mask = mask.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * img + alpha * mask, 0, 1)



# DOWNLOAD

def download(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


# 3 NOVEL INTERNET IMAGES

urls = [
    "https://images.unsplash.com/photo-1517849845537-4d257902454a?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?auto=format&fit=crop&w=800&q=80",
]


# TABLE

table = wandb.Table(columns=[
    "Original Image",
    "Bounding Box Output",
    "Cropped Subject",
    "Segmentation Output"
])


# PIPELINE

for i, url in enumerate(urls):
    img = download(url)
    w, h = img.size

    inp = preprocess(img).unsqueeze(0).to(device)

    # Localization
    with torch.no_grad():
        box = localizer(inp)[0].cpu().numpy()

    box = resize_box(box, w, h)
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    x1, y1, x2, y2 = clamp(x1, y1, x2, y2, w, h)

    # Bounding box image
    boxed = img.copy()
    ImageDraw.Draw(boxed).rectangle([x1, y1, x2, y2], outline="red", width=4)

    # Crop for classifier stage showcase
    crop = img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img

    # Segmentation
    with torch.no_grad():
        seg = segmenter(inp).argmax(1)[0].cpu().numpy()

    disp = denorm(inp[0])
    seg_img = overlay(disp, colorize(seg))

    # Add to table 
    table.add_data(
        wandb.Image(img, caption=f"Original {i+1}"),
        wandb.Image(boxed, caption=f"BBox {i+1}"),
        wandb.Image(crop, caption=f"Crop {i+1}"),
        wandb.Image(seg_img, caption=f"Segmentation {i+1}")
    )


    wandb.log({
        f"original_image_{i+1}": wandb.Image(img),
        f"bbox_output_{i+1}": wandb.Image(boxed),
        f"crop_output_{i+1}": wandb.Image(crop),
        f"segmentation_output_{i+1}": wandb.Image(seg_img),
    })

# log table too
wandb.log({"Final Pipeline Showcase Table": table})

wandb.finish()
print("Done ✅ Check both Media and Table in W&B.")
