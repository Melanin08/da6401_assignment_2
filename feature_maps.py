import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from models.classification import VGG11Classifier


# -------- PATHS --------
ckpt_path = "/kaggle/input/datasets/melaninayman08/all-checkpoint/classifier.pth"
img_path = "data/oxford-iiit-pet/images/american_bulldog_71.jpg"


# -------- CHECK PATHS --------
print("Checkpoint exists:", os.path.exists(ckpt_path))
print("Image exists:", os.path.exists(img_path))


# -------- LOAD MODEL --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG11Classifier()
checkpoint = torch.load(ckpt_path, map_location=device)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()


# -------- LOAD IMAGE --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)


# -------- HOOKS --------
first_layer_output = []
last_layer_output = []

def hook_first(module, input, output):
    first_layer_output.append(output.detach().cpu())

def hook_last(module, input, output):
    last_layer_output.append(output.detach().cpu())


# -------- ATTACH HOOKS --------
h1 = model.encoder.block1.block[0].register_forward_hook(hook_first)
h2 = model.encoder.block5.block[0].register_forward_hook(hook_last)


# -------- FORWARD PASS --------
with torch.no_grad():
    _ = model(img)

h1.remove()
h2.remove()


# -------- PLOT FUNCTION --------
def plot_feature_maps(feature_maps, title, num_maps=16):
    maps = feature_maps[0][0]
    num_maps = min(num_maps, maps.shape[0])

    plt.figure(figsize=(10, 10))
    for i in range(num_maps):
        plt.subplot(4, 4, i + 1)
        plt.imshow(maps[i], cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# -------- SHOW --------
plot_feature_maps(first_layer_output, "First Layer Feature Maps")
plot_feature_maps(last_layer_output, "Last Layer Feature Maps")
