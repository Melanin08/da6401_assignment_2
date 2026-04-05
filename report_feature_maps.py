"""Visualize early and late feature maps for classification model."""

import argparse
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="feature_maps")
    return parser.parse_args()


def denormalize(img_tensor):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img


def make_grid(feature_tensor, max_maps=16):
    feat = feature_tensor.detach().cpu()[0]
    n = min(max_maps, feat.shape[0])
    cols = 4
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(10, 2.5 * rows))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(feat[i].numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"ch {i}")
    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordIIITPetDataset(root=args.data_root, split="val", task="classification")
    sample = dataset[args.image_index]
    image = sample["image"].unsqueeze(0).to(device)

    model = VGG11Classifier(num_classes=37).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    captured = {}

    def hook_first(module, inp, out):
        captured["first"] = out

    def hook_last(module, inp, out):
        captured["last"] = out

    h1 = model.encoder.block1.block[0].register_forward_hook(hook_first)
    h2 = model.encoder.block5[-1].block[0].register_forward_hook(hook_last)

    with torch.no_grad():
        _ = model(image)

    h1.remove()
    h2.remove()

    fig_first = make_grid(captured["first"])
    fig_last = make_grid(captured["last"])

    wandb.init(project="da6401_assignment_2", name=args.run_name, config=vars(args))
    wandb.log({
        "input_image": wandb.Image(denormalize(sample["image"])),
        "first_conv_feature_maps": wandb.Image(fig_first),
        "last_conv_feature_maps": wandb.Image(fig_last),
    })
    wandb.finish()

    plt.close("all")


if __name__ == "__main__":
    main()