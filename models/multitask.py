"""Unified multi-task model."""

import os
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11
from .layers import CustomDropout


class DoubleConv(nn.Module):
    """Two convolution blocks used in the segmentation decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model with:
    - classification head
    - localization head
    - segmentation head
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        load_pretrained: bool = True,
    ):
        super().__init__()

        # Download checkpoints from Google Drive if they are not present locally.
        if not os.path.exists(classifier_path):
            gdown.download(
                id="https://drive.google.com/file/d/1OgDtzbogA_IkqjcH8vxyGZ1gFo9AJjvQ/view?usp=sharing",
                output=classifier_path,
                quiet=False,
            )

        if not os.path.exists(localizer_path):
            gdown.download(
                id="https://drive.google.com/file/d/1riCK94wNSMtMqfmFFHkT41s_sL3TZEZC/view?usp=sharing",
                output=localizer_path,
                quiet=False,
            )

        if not os.path.exists(unet_path):
            gdown.download(
                id="https://drive.google.com/file/d/1xTwaK0NxVMrhNoO_Th7IWRG1TuaWjEyl/view?usp=sharing",
                output=unet_path,
                quiet=False,
            )

        # Shared backbone
        self.backbone = VGG11(in_channels, use_batchnorm=use_batchnorm)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # Localization head
        self.localization_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 4),
        )

        # Segmentation decoder
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_seg = nn.Conv2d(64, seg_classes, kernel_size=1)

        if load_pretrained:
            self._load_pretrained_weights(classifier_path, localizer_path, unet_path)

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _load_matching_prefix(self, source_state, target_state, source_prefix, target_prefix):
        for key, value in source_state.items():
            if key.startswith(source_prefix):
                new_key = target_prefix + key[len(source_prefix):]
                if new_key in target_state and target_state[new_key].shape == value.shape:
                    target_state[new_key] = value
        return target_state

    def _load_pretrained_weights(self, classifier_path, localizer_path, unet_path):
        state = self.state_dict()

        # Classifier checkpoint
        if classifier_path and os.path.exists(classifier_path):
            ckpt = torch.load(classifier_path, map_location="cpu")
            src = self._extract_state_dict(ckpt)

            state = self._load_matching_prefix(src, state, "encoder.", "backbone.")
            state = self._load_matching_prefix(src, state, "classifier.", "classification_head.")

        # Localizer checkpoint
        if localizer_path and os.path.exists(localizer_path):
            ckpt = torch.load(localizer_path, map_location="cpu")
            src = self._extract_state_dict(ckpt)

            state = self._load_matching_prefix(src, state, "encoder.", "backbone.")
            state = self._load_matching_prefix(src, state, "regressor.", "localization_head.")

        # U-Net checkpoint
        if unet_path and os.path.exists(unet_path):
            ckpt = torch.load(unet_path, map_location="cpu")
            src = self._extract_state_dict(ckpt)

            state = self._load_matching_prefix(src, state, "encoder.", "backbone.")

            for name in [
                "up5", "dec5",
                "up4", "dec4",
                "up3", "dec3",
                "up2", "dec2",
                "up1", "dec1",
                "final_seg",
            ]:
                state = self._load_matching_prefix(src, state, f"{name}.", f"{name}.")

        self.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor):
        bottleneck, features = self.backbone(x, return_features=True)

        f1 = features["f1"]
        f2 = features["f2"]
        f3 = features["f3"]
        f4 = features["f4"]
        f5 = features["f5"]

        # Classification
        cls_out = self.classification_head(bottleneck)

        # Localization
        loc = self.localization_head(bottleneck)
        xc = torch.sigmoid(loc[:, 0]) * 224.0
        yc = torch.sigmoid(loc[:, 1]) * 224.0
        w = torch.relu(loc[:, 2])
        h = torch.relu(loc[:, 3])
        loc_out = torch.stack([xc, yc, w, h], dim=1)

        # Segmentation
        seg = self.up5(bottleneck)
        if seg.shape[-2:] != f5.shape[-2:]:
            seg = F.interpolate(seg, size=f5.shape[-2:], mode="bilinear", align_corners=False)
        seg = torch.cat([seg, f5], dim=1)
        seg = self.dec5(seg)

        seg = self.up4(seg)
        if seg.shape[-2:] != f4.shape[-2:]:
            seg = F.interpolate(seg, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        seg = torch.cat([seg, f4], dim=1)
        seg = self.dec4(seg)

        seg = self.up3(seg)
        if seg.shape[-2:] != f3.shape[-2:]:
            seg = F.interpolate(seg, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        seg = torch.cat([seg, f3], dim=1)
        seg = self.dec3(seg)

        seg = self.up2(seg)
        if seg.shape[-2:] != f2.shape[-2:]:
            seg = F.interpolate(seg, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        seg = torch.cat([seg, f2], dim=1)
        seg = self.dec2(seg)

        seg = self.up1(seg)
        if seg.shape[-2:] != f1.shape[-2:]:
            seg = F.interpolate(seg, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        seg = torch.cat([seg, f1], dim=1)
        seg = self.dec1(seg)

        seg_out = self.final_seg(seg)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
