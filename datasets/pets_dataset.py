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
            label = int(parts[1]) - 1  # 0-indexed breed label

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
        """Extract pixel-space bounding box from trimap mask.

        Returns (x_center, y_center, width, height) in pixel coordinates
        relative to the (resized) mask dimensions — e.g. values in [0, 224].
        This matches the localizer output format expected by the autograder.
        """

        mask_arr = np.array(mask)
        H, W = mask_arr.shape

        # Oxford trimaps: 1=foreground, 2=background, 3=boundary
        # Use foreground pixels; fall back to all non-zero if none found.
        ys, xs = np.where(mask_arr == 1)
        if len(xs) == 0:
            ys, xs = np.where(mask_arr > 0)
        if len(xs) == 0:
            # Degenerate mask — return full image box
            return np.array([W / 2, H / 2, float(W), float(H)],
                            dtype=np.float32)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Pixel-space (cx, cy, w, h)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width    = float(x_max - x_min)
        height   = float(y_max - y_min)

        return np.array([x_center, y_center, width, height],
                        dtype=np.float32)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"])

        # Transform image
        image = self.transform(image)

        # Resize mask
        mask = T.Resize(
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST
        )(mask)

        # Extract normalized bbox AFTER resizing
        bbox = self._extract_bbox(mask)

        # Convert mask to tensor: Oxford trimaps are 1=fg, 2=bg, 3=boundary
        # Remap to 0-indexed: 0=fg, 1=bg, 2=boundary
        mask_tensor = torch.from_numpy(np.array(mask)).long() - 1
        mask_tensor = mask_tensor.clamp(min=0, max=2)

        label = torch.tensor(sample["label"]).long()
        bbox = torch.tensor(bbox).float()

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "mask": mask_tensor
        }
