import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from models.classification import VGG11Classifier


# LOAD MODEL 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG11Classifier()
checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()


# LOAD IMAGE 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open("dog.jpg").convert("RGB")
img = transform(img).unsqueeze(0).to(device)


# HOOKS 
first_layer_output = []
last_layer_output = []

def hook_first(module, input, output):
    first_layer_output.append(output.detach().cpu())

def hook_last(module, input, output):
    last_layer_output.append(output.detach().cpu())


# Attach hooks
model.encoder.block1[0].register_forward_hook(hook_first)
model.encoder.block5[-1].register_forward_hook(hook_last)


# Forward pass
with torch.no_grad():
    _ = model(img)


# PLOT FUNCTION
def plot_feature_maps(feature_maps, title):
    maps = feature_maps[0][0]  # batch=1
    plt.figure(figsize=(10, 10))

    for i in range(16):  # show 16 maps
        plt.subplot(4, 4, i+1)
        plt.imshow(maps[i], cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.show()


# SHOW 
plot_feature_maps(first_layer_output, "First Layer Feature Maps")
plot_feature_maps(last_layer_output, "Last Layer Feature Maps")
