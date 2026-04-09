import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from PIL import Image, ImageDraw
from torchvision import transforms

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


# =====================================
# W&B INIT
# =====================================
wandb.init(project="da6401_assignment_2", name="final_pipeline_showcase")


# =====================================
# DEVICE
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================
# PATHS
# =====================================
classifier_ckpt = "checkpoints/classifier.pth"
localizer_ckpt = "checkpoints/localizer.pth"
segmenter_ckpt = "checkpoints/unet.pth"

for path in [classifier_ckpt, localizer_ckpt, segmenter_ckpt]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")


# =====================================
# CLASS NAMES (Oxford-IIIT Pet 37 classes)
# =====================================
breed_names = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay",
    "boxer", "British_Shorthair", "chihuahua", "Egyptian_Mau",
    "english_cocker_spaniel", "english_setter", "german_shorthaired",
    "great_pyrenees", "havanese", "japanese_chin", "keeshond",
    "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
    "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
    "Siamese", "Sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier"
]


# =====================================
# LOAD MODELS
# =====================================
def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


classifier = load_checkpoint(VGG11Classifier(num_classes=37), classifier_ckpt)
localizer = load_checkpoint(VGG11Localizer(), localizer_ckpt)
segmenter = load_checkpoint(VGG11UNet(num_classes=3), segmenter_ckpt)


# =====================================
# PREPROCESS
# =====================================
image_size = 224

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def denorm_image_tensor(img_tensor):
    """
    Convert normalized tensor back to [0,1] image for display.
    """
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img


def xywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    return x1, y1, x2, y2


def resize_box_from_224_to_original(box_224, orig_w, orig_h):
    cx, cy, w, h = box_224
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0
    return [
        cx * scale_x,
        cy * scale_y,
        w * scale_x,
        h * scale_y
    ]


def colorize_mask(mask):
    """
    0 = black
    1 = green
    2 = red
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask == 0] = [0, 0, 0]
    color_mask[mask == 1] = [0, 255, 0]
    color_mask[mask == 2] = [255, 0, 0]
    return color_mask


def overlay_mask_on_image(image_np, color_mask, alpha=0.45):
    """
    image_np: [H,W,3] float in [0,1]
    color_mask: [H,W,3] uint8
    """
    color_mask = color_mask.astype(np.float32) / 255.0
    overlay = (1 - alpha) * image_np + alpha * color_mask
    overlay = np.clip(overlay, 0, 1)
    return overlay


# =====================================
# NOVEL INTERNET IMAGES
# Replace these with any 3 pet image URLs if needed
# =====================================
image_urls = [
    "https://images.unsplash.com/photo-1517849845537-4d257902454a?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?auto=format&fit=crop&w=800&q=80",
]


# =====================================
# DOWNLOAD IMAGE
# =====================================
def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(requests.compat.BytesIO(response.content)).convert("RGB")


# =====================================
# PIPELINE
# =====================================
table = wandb.Table(columns=[
    "Original Image",
    "Localized Image",
    "Cropped Subject",
    "Segmentation Output",
    "Predicted Class",
    "Class Index"
])

generalization_notes = []

for idx, url in enumerate(image_urls):
    print(f"Processing image {idx + 1}...")

    # Download original
    original_img = download_image(url)
    orig_w, orig_h = original_img.size

    # Model input
    input_tensor = preprocess(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Localization
        pred_box_224 = localizer(input_tensor)[0].cpu().numpy()

        # Classification on full image first (fallback)
        cls_logits_full = classifier(input_tensor)
        cls_idx_full = cls_logits_full.argmax(dim=1).item()

        # Segmentation on full image
        seg_logits = segmenter(input_tensor)
        seg_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()

    # Convert localization box to original image size
    pred_box_orig = resize_box_from_224_to_original(pred_box_224, orig_w, orig_h)
    x1, y1, x2, y2 = xywh_to_xyxy(pred_box_orig)
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, orig_w, orig_h)

    # Ensure crop is valid
    if x2 <= x1 or y2 <= y1:
        crop_img = original_img
        crop_note = "Invalid bbox, used full image"
    else:
        crop_img = original_img.crop((x1, y1, x2, y2))
        crop_note = "Used predicted bbox crop"

    # Classify crop
    crop_tensor = preprocess(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_logits_crop = classifier(crop_tensor)
        cls_idx_crop = cls_logits_crop.argmax(dim=1).item()

    pred_class_name = breed_names[cls_idx_crop] if 0 <= cls_idx_crop < len(breed_names) else str(cls_idx_crop)

    # Create localized image with bbox
    localized_img = original_img.copy()
    draw = ImageDraw.Draw(localized_img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    # Prepare segmentation overlay
    input_disp = denorm_image_tensor(input_tensor[0])
    seg_color = colorize_mask(seg_mask)
    seg_overlay = overlay_mask_on_image(input_disp, seg_color, alpha=0.45)

    # Log to W&B
    table.add_data(
        wandb.Image(original_img, caption=f"Novel Image {idx + 1}"),
        wandb.Image(localized_img, caption="Predicted Bounding Box"),
        wandb.Image(crop_img, caption=crop_note),
        wandb.Image(seg_overlay, caption="Predicted Segmentation Overlay"),
        pred_class_name,
        cls_idx_crop
    )

    # Short note for later report use
    generalization_notes.append(
        f"Image {idx + 1}: predicted class = {pred_class_name}, crop_note = {crop_note}"
    )

wandb.log({
    "Final Pipeline Showcase": table,
    "Generalization Notes": wandb.Table(columns=["Note"], data=[[n] for n in generalization_notes])
})

wandb.finish()
print("Done. Check W&B report/run for final pipeline showcase.")
