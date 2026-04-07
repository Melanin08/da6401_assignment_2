"""Inference, evaluation, and visualization utilities."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation", "multitask"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    # Optional visualization mode
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=[
            "eval",
            "feature_maps",
            "bbox_examples",
            "seg_examples",
            "multitask_examples",
        ],
        help="Choose evaluation or visualization mode",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of examples to visualize",
    )

    return parser.parse_args()


# LOAD MODEL

def load_model(task, model_path, device):
    if task == "classification":
        model = VGG11Classifier(num_classes=37)

    elif task == "localization":
        model = VGG11Localizer()

    elif task == "segmentation":
        model = VGG11UNet(num_classes=3)

    elif task == "multitask":
        model = MultiTaskPerceptionModel(load_pretrained=False)

    else:
        raise ValueError("Invalid task")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    return model



# HELPER FUNCTIONS

def tensor_to_display_image(image_tensor):
    """
    Convert [C,H,W] tensor to numpy image for display.
    Handles normalized images approximately by rescaling to [0,1].
    """
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()

    # Robust display scaling
    image_min = image.min()
    image_max = image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)

    return image


def xywh_to_xyxy(box):
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



# CLASSIFICATION

def evaluate_classification(model, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Classification Accuracy: {acc:.4f}")


def visualize_feature_maps(model, dataset, device, sample_index=0, max_maps=8):
    """
    For report question 2.4:
    Visualize feature maps from first conv layer and last conv layer.
    """
    sample = dataset[sample_index]
    image = sample["image"].unsqueeze(0).to(device)

    first_maps = []
    last_maps = []

    def first_hook(module, inp, out):
        first_maps.append(out.detach().cpu())

    def last_hook(module, inp, out):
        last_maps.append(out.detach().cpu())

    # Adjust these if your encoder block names differ
    h1 = model.encoder.block1[0].register_forward_hook(first_hook)
    h2 = model.encoder.block5[-2].register_forward_hook(last_hook)

    with torch.no_grad():
        _ = model(image)

    h1.remove()
    h2.remove()

    display_img = tensor_to_display_image(sample["image"])

    plt.figure(figsize=(4, 4))
    plt.imshow(display_img)
    plt.title("Input Image")
    plt.axis("off")
    plt.show()

    if len(first_maps) > 0:
        fmap = first_maps[0][0]
        n = min(max_maps, fmap.shape[0])

        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        for i in range(n):
            axes[i].imshow(fmap[i], cmap="gray")
            axes[i].axis("off")
        plt.suptitle("First Convolutional Layer Feature Maps")
        plt.show()

    if len(last_maps) > 0:
        fmap = last_maps[0][0]
        n = min(max_maps, fmap.shape[0])

        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        for i in range(n):
            axes[i].imshow(fmap[i], cmap="gray")
            axes[i].axis("off")
        plt.suptitle("Last Convolutional Layer Feature Maps")
        plt.show()



# LOCALIZATION

def evaluate_localization(model, loader, device):
    total_loss = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["bbox"].to(device)

            preds = model(images)

            loss = torch.mean((preds - targets) ** 2)
            total_loss += loss.item()

            batch_ious = []
            for i in range(preds.size(0)):
                pred_box = preds[i].cpu().numpy()
                gt_box = targets[i].cpu().numpy()
                batch_ious.append(compute_iou_xywh(pred_box, gt_box))

            total_iou += float(np.mean(batch_ious))
            total_batches += 1

    print(f"Localization MSE: {total_loss / len(loader):.4f}")
    print(f"Localization Mean IoU: {total_iou / total_batches:.4f}")


def visualize_bbox(image_tensor, pred_box, gt_box=None, title="Bounding Box Visualization"):
    image = tensor_to_display_image(image_tensor)

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(image)

    # Predicted box in red
    px1, py1, px2, py2 = xywh_to_xyxy(pred_box)
    pred_rect = plt.Rectangle(
        (px1, py1),
        px2 - px1,
        py2 - py1,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="Prediction",
    )
    ax.add_patch(pred_rect)

    # Ground truth box in green
    if gt_box is not None:
        gx1, gy1, gx2, gy2 = xywh_to_xyxy(gt_box)
        gt_rect = plt.Rectangle(
            (gx1, gy1),
            gx2 - gx1,
            gy2 - gy1,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label="Ground Truth",
        )
        ax.add_patch(gt_rect)

    if gt_box is not None:
        iou = compute_iou_xywh(pred_box, gt_box)
        ax.set_title(f"{title}\nIoU: {iou:.4f}")
    else:
        ax.set_title(title)

    ax.axis("off")
    plt.show()


def run_bbox_examples(model, dataset, device, num_samples=10):
    model.eval()

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        gt_box = sample["bbox"].cpu().numpy()

        with torch.no_grad():
            pred_box = model(image)[0].cpu().numpy()

        visualize_bbox(sample["image"], pred_box, gt_box, title=f"Sample {i}")



# SEGMENTATION

def evaluate_segmentation(model, loader, device):
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

    acc = correct_pixels / total_pixels
    print(f"Segmentation Pixel Accuracy: {acc:.4f}")


def visualize_segmentation(image_tensor, pred_mask, gt_mask, title="Segmentation Example"):
    image = tensor_to_display_image(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask.cpu().numpy(), cmap="viridis")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask.cpu().numpy(), cmap="viridis")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.show()


def run_segmentation_examples(model, dataset, device, num_samples=5):
    model.eval()

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["mask"]

        with torch.no_grad():
            pred_logits = model(image)
            pred_mask = pred_logits.argmax(dim=1)[0].cpu()

        visualize_segmentation(sample["image"], pred_mask, gt_mask, title=f"Sample {i}")



# MULTITASK

def evaluate_multitask(model, loader, device):
    correct_cls = 0
    total_cls = 0
    total_loc_loss = 0.0
    total_iou = 0.0
    total_iou_batches = 0
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)

            cls_out = outputs["classification"]
            box_out = outputs["localization"]
            seg_out = outputs["segmentation"]

            preds = cls_out.argmax(dim=1)
            correct_cls += (preds == labels).sum().item()
            total_cls += labels.size(0)

            loc_loss = torch.mean((box_out - boxes) ** 2)
            total_loc_loss += loc_loss.item()

            batch_ious = []
            for i in range(box_out.size(0)):
                pred_box = box_out[i].cpu().numpy()
                gt_box = boxes[i].cpu().numpy()
                batch_ious.append(compute_iou_xywh(pred_box, gt_box))
            total_iou += float(np.mean(batch_ious))
            total_iou_batches += 1

            seg_preds = seg_out.argmax(dim=1)
            correct_pixels += (seg_preds == masks).sum().item()
            total_pixels += masks.numel()

    cls_acc = correct_cls / total_cls
    loc_loss = total_loc_loss / len(loader)
    loc_iou = total_iou / total_iou_batches
    seg_acc = correct_pixels / total_pixels

    print(f"Multitask Classification Acc: {cls_acc:.4f}")
    print(f"Multitask Localization MSE: {loc_loss:.4f}")
    print(f"Multitask Localization Mean IoU: {loc_iou:.4f}")
    print(f"Multitask Segmentation Acc: {seg_acc:.4f}")


def run_multitask_examples(model, dataset, device, num_samples=3):
    model.eval()

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image_tensor = sample["image"]
        image = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)

        pred_class = outputs["classification"].argmax(dim=1)[0].item()
        pred_box = outputs["localization"][0].cpu().numpy()
        pred_mask = outputs["segmentation"].argmax(dim=1)[0].cpu()

        gt_box = sample["bbox"].cpu().numpy()
        gt_mask = sample["mask"]

        print(f"Sample {i} | Predicted class: {pred_class}")

        visualize_bbox(image_tensor, pred_box, gt_box, title=f"Multitask BBox Sample {i}")
        visualize_segmentation(image_tensor, pred_mask, gt_mask, title=f"Multitask Segmentation Sample {i}")



# MAIN

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="test",
        task=args.task,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.task, args.model_path, device)

    if args.mode == "eval":
        if args.task == "classification":
            evaluate_classification(model, loader, device)

        elif args.task == "localization":
            evaluate_localization(model, loader, device)

        elif args.task == "segmentation":
            evaluate_segmentation(model, loader, device)

        elif args.task == "multitask":
            evaluate_multitask(model, loader, device)

    elif args.mode == "feature_maps":
        if args.task != "classification":
            raise ValueError("feature_maps mode is only for classification")
        visualize_feature_maps(model, dataset, device, sample_index=0, max_maps=8)

    elif args.mode == "bbox_examples":
        if args.task != "localization":
            raise ValueError("bbox_examples mode is only for localization")
        run_bbox_examples(model, dataset, device, num_samples=args.num_samples)

    elif args.mode == "seg_examples":
        if args.task != "segmentation":
            raise ValueError("seg_examples mode is only for segmentation")
        run_segmentation_examples(model, dataset, device, num_samples=args.num_samples)

    elif args.mode == "multitask_examples":
        if args.task != "multitask":
            raise ValueError("multitask_examples mode is only for multitask")
        run_multitask_examples(model, dataset, device, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
