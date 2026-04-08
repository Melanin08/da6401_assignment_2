import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import wandb

from models.classification import VGG11Classifier


# -------- INIT W&B --------
wandb.init(project="da6401_assignment_2", name="feature_maps")


# -------- PATHS --------
ckpt_path = "checkpoints/classifier.pth"
img_path = "data/oxford-iiit-pet/images/Abyssinian_94.jpg"


# -------- CHECK --------
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")


# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------- LOAD MODEL --------
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
img_tensor = transform(img).unsqueeze(0).to(device)


# -------- HOOKS --------
first_layer_output = []
last_layer_output = []

def hook_first(module, input, output):
    first_layer_output.append(output.detach().cpu())

def hook_last(module, input, output):
    last_layer_output.append(output.detach().cpu())


# -------- ATTACH HOOKS --------
h1 = model.encoder.block1.block[0].register_forward_hook(hook_first)
h2 = model.encoder.block5[0].register_forward_hook(hook_last)


# -------- FORWARD --------
with torch.no_grad():
    _ = model(img_tensor)

h1.remove()
h2.remove()


# -------- PLOT + LOG --------
def plot_and_log(feature_maps, title):
    maps = feature_maps[0][0]
    num_maps = min(16, maps.shape[0])

    fig = plt.figure(figsize=(8, 8))

    for i in range(num_maps):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(maps[i], cmap="gray")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    # Save locally
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename)

    # Log to W&B
    wandb.log({title: wandb.Image(filename)})

    plt.close(fig)


# -------- RUN --------
plot_and_log(first_layer_output, "First Layer Feature Maps")
plot_and_log(last_layer_output, "Last Layer Feature Maps")

print("Logged to W&B successfully")
