import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import wandb

from models.classification import VGG11Classifier


# INIT W&B

wandb.init(project="da6401_assignment_2", name="feature_maps")


# PATHS

ckpt_path = "checkpoints/classifier.pth"
img_path = "data/oxford-iiit-pet/images/beagle_1.jpg"


# CHECK FILES

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")



# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# LOAD MODEL

model = VGG11Classifier()

checkpoint = torch.load(ckpt_path, map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()


# LOAD IMAGE

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)


# HELPER FUNCTIONS

def get_first_conv(module):
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in module")


def get_last_conv(module):
    last_conv = None
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
    if last_conv is None:
        raise ValueError("No Conv2d layer found in module")
    return last_conv


# HOOK STORAGE

first_layer_output = []
last_layer_output = []


# HOOK FUNCTIONS

def hook_first(module, input, output):
    first_layer_output.clear()
    first_layer_output.append(output.detach().cpu())


def hook_last(module, input, output):
    last_layer_output.clear()
    last_layer_output.append(output.detach().cpu())


# ATTACH HOOKS

first_conv = get_first_conv(model.encoder.block1)
last_conv = get_last_conv(model.encoder.block5)

h1 = first_conv.register_forward_hook(hook_first)
h2 = last_conv.register_forward_hook(hook_last)


# FORWARD PASS

with torch.no_grad():
    _ = model(img_tensor)

h1.remove()
h2.remove()


# FUNCTION TO SAVE FEATURE MAP GRID

def save_feature_map_grid(feature_maps, title):
    if len(feature_maps) == 0:
        raise ValueError(f"No feature maps captured for {title}")

    maps = feature_maps[0][0]   # remove batch dimension
    num_maps = min(16, maps.shape[0])

    fig = plt.figure(figsize=(8, 8))
    for i in range(num_maps):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(maps[i], cmap="gray")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return filename


# SAVE FEATURE MAPS

first_map_file = save_feature_map_grid(first_layer_output, "First Layer Feature Maps")
last_map_file = save_feature_map_grid(last_layer_output, "Last Layer Feature Maps")


# PREPARE ORIGINAL IMAGE FOR W&B

img_display = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)


# LOG TO W&B
wandb.log({
    "Original_Image": wandb.Image(img_display, caption="Original Input Image (224x224)"),
    "First_Layer_Feature_Maps": wandb.Image(first_map_file, caption="First Layer Feature Maps"),
    "Last_Layer_Feature_Maps": wandb.Image(last_map_file, caption="Last Layer Feature Maps")
})

print("Original image and feature maps generated and logged to W&B successfully!")

wandb.finish()
