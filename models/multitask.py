"""Unified multi-task model."""

import os
import gdown
import torch

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(torch.nn.Module):
    def __init__(
        self,
        num_breeds=37,
        seg_classes=3,
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth",
        load_pretrained=True,
    ):
        super().__init__()

        os.makedirs("checkpoints", exist_ok=True)

        if not os.path.exists(classifier_path):
            gdown.download(
                id="1OgDtzbogA_IkqjcH8vxyGZ1gFo9AJjvQ",
                output=classifier_path,
                quiet=False
            )

        if not os.path.exists(localizer_path):
            gdown.download(
                id="1riCK94wNSMtMqfmFFHkT41s_sL3TZEZC",
                output=localizer_path,
                quiet=False
            )

        if not os.path.exists(unet_path):
            gdown.download(
                id="1xTwaK0NxVMrhNoO_Th7IWRG1TuaWjEyl",
                output=unet_path,
                quiet=False
            )

        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer = VGG11Localizer()
        self.segmenter = VGG11UNet(num_classes=seg_classes)

        if load_pretrained:
            self._load_model(self.classifier, classifier_path, task="classifier")
            self._load_model(self.localizer, localizer_path, task="localizer")
            self._load_model(self.segmenter, unet_path, task="segmenter")

        self.classifier.eval()
        self.localizer.eval()
        self.segmenter.eval()

    def _extract_state_dict(self, ckpt):
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
            if "model_state_dict" in ckpt:
                return ckpt["model_state_dict"]
        return ckpt

    def _clean_prefixes(self, key):
        key = key.replace("module.", "")
        key = key.replace("model.", "")
        return key

    def _remap_key(self, key, task):
        key = self._clean_prefixes(key)

        if task == "classifier":
            key = key.replace("backbone.", "encoder.")
            key = key.replace("features.", "encoder.")
            key = key.replace("classification_head.", "classifier.")
            key = key.replace("classifier_head.", "classifier.")
            key = key.replace("head.", "classifier.")

        elif task == "localizer":
            key = key.replace("backbone.", "encoder.")
            key = key.replace("features.", "encoder.")
            key = key.replace("localization_head.", "regressor.")
            key = key.replace("regression_head.", "regressor.")
            key = key.replace("bbox_head.", "regressor.")
            key = key.replace("head.", "regressor.")

        elif task == "segmenter":
            key = key.replace("backbone.", "encoder.")
            key = key.replace("features.", "encoder.")
            key = key.replace("final_seg.", "final.")

        return key

    def _load_model(self, model, path, task="model"):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = self._extract_state_dict(ckpt)

        model_dict = model.state_dict()
        filtered = {}
        matched = 0

        for k, v in state_dict.items():
            new_k = self._remap_key(k, task)
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                filtered[new_k] = v
                matched += 1

        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)

        print(f"[DEBUG] {task}: matched {matched} tensors from {path}")

    def forward(self, x):
        cls_out = self.classifier(x)
        loc_out = self.localizer(x)
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }