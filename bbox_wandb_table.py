import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer

# CONFIG
# =========================
DATA_ROOT = "data/oxford-iiit-pet"
CHECKPOINT_PATH = "localizer.pth"
NUM_SAMPLES = 10
BATCH_SIZE = 1


# =========================
# HELPERS
# =========================
def tensor_to_display_image(image_tensor):
    """
    Convert [C,H,W] tensor to HWC numpy image in [0,1] for plotting.
    """
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image_min = image.min()
    image_max = image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return image


def xywh_to_xyxy(box):
    """
    Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
    """
    cx, cy, w, h = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def compute_iou_xywh(pred_box, gt_box, eps=1e-6):
    px1, py1, px2, py2 = xywh_to_xyxy(pred_box)
    gx1, gy1, gx2, gy2 = xywh_to_xyxy(gt_box)

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    pred_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
    gt_area = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)

    union_area = pred_area + gt_area - inter_area + eps
    return inter_area / union_area


def clamp_box_to_image(box, image_size=224):
    """
    Clamp box coordinates to image bounds after converting to xyxy.
    """
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    x1 = max(0.0, min(image_size - 1.0, x1))
    y1 = max(0.0, min(image_size - 1.0, y1))
    x2 = max(0.0, min(image_size - 1.0, x2))
    y2 = max(0.0, min(image_size - 1.0, y2))
    return x1, y1, x2, y2


def box_validity_score(box, image_size=224):
    """
    A proxy confidence score for a regression-only localizer.

    Since the localizer does not output a true detection confidence,
    we compute a simple proxy based on whether the predicted center and box size
    are plausible and stay inside the image. This keeps the report honest.
    """
    cx, cy, w, h = box

    # center plausibility
    center_ok_x = 1.0 - min(abs(cx - image_size / 2.0) / (image_size / 2.0), 1.0)
    center_ok_y = 1.0 - min(abs(cy - image_size / 2.0) / (image_size / 2.0), 1.0)
    center_score = 0.5 * (center_ok_x + center_ok_y)

    # size plausibility
    w_score = min(max(w / image_size, 0.0), 1.0)
    h_score = min(max(h / image_size, 0.0), 1.0)

    # prefer medium-scale boxes over extreme tiny/huge ones
    size_score = 1.0 - abs(((w_score * h_score) - 0.25)) / 0.25
    size_score = max(0.0, min(1.0, size_score))

    # image-bound score
    x1, y1, x2, y2 = clamp_box_to_image(box, image_size=image_size)
    inside_width = max(x2 - x1, 0.0)
    inside_height = max(y2 - y1, 0.0)
    inside_area = inside_width * inside_height
    pred_area = max(w, 0.0) * max(h, 0.0) + 1e-6
    inside_score = max(0.0, min(1.0, inside_area / pred_area))

    confidence = 0.4 * center_score + 0.3 * size_score + 0.3 * inside_score
    return float(max(0.0, min(1.0, confidence)))


def render_bbox_overlay(image_tensor, pred_box, gt_box, title_text=""):
    """
    Returns a rendered RGB numpy image with:
    - GT box in green
    - Pred box in red
    - title showing confidence + IoU
    """
    image = tensor_to_display_image(image_tensor)

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(image)

    # Ground truth box: green
    gx1, gy1, gx2, gy2 = xywh_to_xyxy(gt_box)
    gt_rect = plt.Rectangle(
        (gx1, gy1),
        gx2 - gx1,
        gy2 - gy1,
        linewidth=2,
        edgecolor="green",
        facecolor="none",
    )
    ax.add_patch(gt_rect)

    # Predicted box: red
    px1, py1, px2, py2 = xywh_to_xyxy(pred_box)
    pred_rect = plt.Rectangle(
        (px1, py1),
        px2 - px1,
        py2 - py1,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(pred_rect)

    ax.set_title(title_text)
    ax.axis("off")

    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return rendered


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wandb.init(
        project="da6401_assignment_2",
        name="object_detection_confidence_iou",
        config={
            "data_root": DATA_ROOT,
            "checkpoint": CHECKPOINT_PATH,
            "num_samples": NUM_SAMPLES,
        },
    )

    dataset = OxfordIIITPetDataset(
        root=DATA_ROOT,
        split="test",
        task="localization",
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VGG11Localizer().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    table = wandb.Table(columns=[
        "Index",
        "Image with Boxes",
        "Confidence Score",
        "IoU",
        "Failure Note",
    ])

    failure_case = None

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= NUM_SAMPLES:
                break

            image = batch["image"].to(device)
            gt_box = batch["bbox"][0].cpu().numpy()
            image_tensor = batch["image"][0]

            pred_box = model(image)[0].cpu().numpy()

            iou = compute_iou_xywh(pred_box, gt_box)
            confidence = box_validity_score(pred_box)

            failure_note = ""

            # Mark a likely failure case
            if (confidence > 0.65 and iou < 0.30) or iou < 0.10:
                failure_note = "Possible failure case"
                if failure_case is None:
                    failure_case = {
                        "index": idx,
                        "confidence": confidence,
                        "iou": iou,
                    }

            title_text = f"Conf: {confidence:.2f} | IoU: {iou:.2f}"
            rendered = render_bbox_overlay(image_tensor, pred_box, gt_box, title_text)

            table.add_data(
                idx,
                wandb.Image(rendered),
                float(confidence),
                float(iou),
                failure_note,
            )

    wandb.log({"Object Detection Table": table})

    if failure_case is not None:
        print("\nFailure case found:")
        print(
            f"Sample {failure_case['index']} | "
            f"Confidence: {failure_case['confidence']:.2f} | "
            f"IoU: {failure_case['iou']:.2f}"
        )
    else:
        print("\nNo strong failure case found among the first 10 samples.")
        print("You can increase NUM_SAMPLES or inspect the lowest-IoU rows manually in W&B.")

    wandb.finish()
    print("Done. Check the W&B run for the table.")


if __name__ == "__main__":
    main()
