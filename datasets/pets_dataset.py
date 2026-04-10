"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(
        self,
        root,
        split="train",
        image_size=224,
        transform=None
    ):

        self.root = root
        self.split = split
        self.image_size = image_size

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.bbox_file = os.path.join(root, "annotations", "list.txt")

        self.transform = transform

        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        self.samples = self._load_metadata()

    def _load_metadata(self):

        samples = []

        with open(self.bbox_file, "r") as f:
            lines = f.readlines()[6:]

        for line in lines:
            parts = line.strip().split()

            image_name = parts[0]
            label = int(parts[1]) - 1

            img_path = os.path.join(self.images_dir, image_name + ".jpg")
            mask_path = os.path.join(self.masks_dir, image_name + ".png")

            samples.append({
                "image": img_path,
                "mask": mask_path,
                "label": label
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def _extract_bbox(self, mask):

        mask = np.array(mask)

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        width = x_max - x_min
        height = y_max - y_min

        return np.array([x_center, y_center, width, height], dtype=np.float32)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"])

        # Transform image
        image = self.transform(image)

        # Resize mask first
        mask = T.Resize((self.image_size, self.image_size))(mask)

        # Extract bbox AFTER resizing
        bbox = self._extract_bbox(mask)

        # Convert mask to tensor and fix label range (0,1,2)
        mask = torch.from_numpy(np.array(mask)).long() - 1
        mask = mask.clamp(min=0)

        label = torch.tensor(sample["label"]).long()
        bbox = torch.tensor(bbox).float()

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "mask": mask
        }