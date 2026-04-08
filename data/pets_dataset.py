"""Oxford-IIIT Pet dataset loader."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """
    Multi-task Oxford-IIIT Pet dataset loader.

    Supports:
    - classification (predict breed)
    - localization (predict bounding box)
    - segmentation (predict pixel mask)
    - multitask (all tasks together)
    """

    def __init__(
        self,
        root="data/oxford-iiit-pet",   
        split="train",                
        image_size=224,              
        val_ratio=0.1,              
        seed=42,                      
        task="multitask",             
        download=False,               
    ):
        # Store parameters
        self.root = root
        self.split = split
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.task = task
        self.download = download

        # Define important paths
        self.images_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.trimaps_dir = os.path.join(self.ann_dir, "trimaps")  
        self.xml_dir = os.path.join(self.ann_dir, "xmls")         

        # Split definition files
        self.trainval_file = os.path.join(self.ann_dir, "trainval.txt")
        self.test_file = os.path.join(self.ann_dir, "test.txt")

        # Check dataset structure
        self._check_paths()

        # Build sample list
        self.samples = self._build_samples()

    def _check_paths(self):
        """
        Ensure dataset structure is correct.
        """
        required_paths = [
            self.images_dir,
            self.ann_dir,
            self.trimaps_dir,
            self.xml_dir,
            self.trainval_file,
            self.test_file,
        ]

        # Check all required files/folders exist
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset component: {path}")

        # Validate split
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        # Validate task
        if self.task not in {"classification", "localization", "segmentation", "multitask"}:
            raise ValueError(
                "task must be one of: 'classification', 'localization', 'segmentation', 'multitask'"
            )

    def _read_split_file(self, filepath):
        """
        Read train/test split file.
        Each line contains:
        image_id label
        """
        entries = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip empty or comment lines
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                img_id = parts[0].strip()
                label = int(parts[1]) - 1  # convert labels 1–37 to 0–36

                entries.append({
                    "img_id": img_id,
                    "label": label,
                })

        return entries

    def _split_train_val(self, entries):
        """
        Split train into train + validation.
        """
        n = len(entries)
        indices = np.arange(n)

        # Shuffle indices
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        # Select validation indices
        val_size = int(n * self.val_ratio)
        val_indices = set(indices[:val_size].tolist())

        train_entries = []
        val_entries = []

        # Split into train and val
        for i, entry in enumerate(entries):
            if i in val_indices:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

        return train_entries, val_entries

    def _parse_bbox_from_mask(self, img_id):
        """
        Fallback: derive bounding box from segmentation mask.

        Trimap labels:
        1 = pet
        2 = background
        3 = border

        We consider everything != 2 as foreground.
        """
        mask_path = os.path.join(self.trimaps_dir, f"{img_id}.png")

        if not os.path.exists(mask_path):
            return None

        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)

        # Get coordinates where pet exists
        ys, xs = np.where(mask != 2)

        if len(xs) == 0 or len(ys) == 0:
            return None

        # Compute bounding box
        xmin = float(xs.min())
        xmax = float(xs.max())
        ymin = float(ys.min())
        ymax = float(ys.max())

        # Convert to center format
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        return [x_center, y_center, width, height]

    def _parse_bbox_from_xml(self, img_id):
        """
        Read bounding box from XML.
        If missing → fallback to mask.
        """
        xml_path = os.path.join(self.xml_dir, f"{img_id}.xml")

        # Try XML first
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            bndbox = root.find(".//bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin

                return [x_center, y_center, width, height]

        # If XML missing → fallback to mask
        return self._parse_bbox_from_mask(img_id)

    def _build_samples(self):
        """
        Build dataset samples list.
        """
        # Load correct split
        if self.split == "test":
            entries = self._read_split_file(self.test_file)
        else:
            all_entries = self._read_split_file(self.trainval_file)
            train_entries, val_entries = self._split_train_val(all_entries)
            entries = train_entries if self.split == "train" else val_entries

        samples = []

        for entry in entries:
            img_id = entry["img_id"]
            label = entry["label"]

            # Get bounding box
            bbox = self._parse_bbox_from_xml(img_id)

            # For localization tasks → skip if no bbox
            if self.task in {"localization", "multitask"} and bbox is None:
                continue

            # For classification → allow empty bbox
            if bbox is None:
                bbox = [0.0, 0.0, 0.0, 0.0]

            samples.append({
                "img_id": img_id,
                "label": label,
                "bbox": bbox,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one sample.
        """
        sample = self.samples[idx]

        img_id = sample["img_id"]
        label = sample["label"]
        bbox = sample["bbox"]

        # Paths
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.trimaps_dir, f"{img_id}.png")

        # Safety checks
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask file: {mask_path}")

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Store original size (for bbox scaling)
        orig_w, orig_h = image.size

        # Resize image and mask
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        # Normalize image
        image = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Convert mask labels from {1,2,3} → {0,1,2}
        mask = np.array(mask, dtype=np.int64) - 1
        mask = torch.from_numpy(mask)

        # Scale bounding box to resized image
        x_center, y_center, width, height = bbox

        scale_x = self.image_size / float(orig_w)
        scale_y = self.image_size / float(orig_h)

        x_center *= scale_x
        y_center *= scale_y
        width *= scale_x
        height *= scale_y

        bbox = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "mask": mask,
        }
