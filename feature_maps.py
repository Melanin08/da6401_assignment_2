import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from models.classification import VGG11Classifier


# CHECK PATHS FIRST
print("Checkpoint exists:", os.path.exists("/kaggle/input/checkpoint/classifier.pth"))
print("Image exists:", os.path.exists("/kaggle/input/data/oxford-iiit-pet/images/american_bulldog_71.jpg.jpeg"))


# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG11Classifier()
checkpoint = torch.load("/kaggle/input/datasets/melaninayman08/all-checkpoint/classifier.pth", map_location=device)

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

img = Image.open("/kaggle/input/datasets/melaninayman08/dog-file/american_bulldog_71.jpg.jpeg").convert("RGB")
img = transform(img).unsqueeze(0).to(device)


# HOOKS
first_layer_output = []
last_layer_output = []

def hook_first(module, input, output):
    first_layer_output.append(output.detach().cpu())

def hook_last(module, input, output):
    last_layer_output.append(output.detach().cpu())


# ATTACH HOOKS
h1 = model.encoder.block1.block[0].register_forward_hook(hook_first)
h2 = model.encoder.block5.block[0].register_forward_hook(hook_last)


# FORWARD PASS
with torch.no_grad():
    _ = model(img)

h1.remove()
h2.remove()


# PLOT FUNCTION
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


# SHOW
plot_feature_maps(first_layer_output, "First Layer Feature Maps")
plot_feature_maps(last_layer_output, "Last Layer Feature Maps")
