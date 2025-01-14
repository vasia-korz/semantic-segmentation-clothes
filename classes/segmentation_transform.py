import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

if TYPE_CHECKING:
    import numpy as np


class SegmentationTransform:
    def __init__(
        self,
        size: tuple[int, int] = (256, 256),
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        self.size = size
        self.normalize = normalize
        self.augment = augment

        self.resize = transforms.Resize(
            size, interpolation=transforms.InterpolationMode.NEAREST
        )

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            if normalize
            else transforms.Lambda(lambda x: x),
        ])

        self.mask_transform = transforms.Lambda(
            lambda x: torch.tensor(np.array(x), dtype=torch.long)
        )

    def __call__(
        self, image: Any, mask: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.resize(image)
        mask = self.resize(mask)

        if self.augment and random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze(0)

        return image, mask
