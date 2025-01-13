from torchvision import transforms
from PIL import Image
import random
import torch
import numpy as np

class SegmentationTransform:
    def __init__(self, size=(256, 256), normalize=True, augment=False):
        self.size = size
        self.normalize = normalize
        self.augment = augment

        self.resize = transforms.Resize(size, interpolation=Image.NEAREST)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else transforms.Lambda(lambda x: x)
        ])

        self.mask_transform = transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))


    def __call__(self, image, mask):
        image = self.resize(image)
        mask = self.resize(mask)

        if self.augment and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze(0)

        return image, mask
