import os
from collections.abc import Callable
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class ClothesDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Callable | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Any:
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            augmented = self.transform(image, mask)
            image, mask = augmented

        return image, mask
