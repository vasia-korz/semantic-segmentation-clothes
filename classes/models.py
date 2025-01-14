from abc import ABC, abstractmethod

import segmentation_models_pytorch as smp
from torch._prims_common import DeviceLikeType

from classes.clothes_model import ClothesModel, LossType, OptimType


class SegmentationModel(ClothesModel, ABC):
    def __init__(
        self,
        loss_fn: LossType,
        optimizer: OptimType,
        lr: float,
        device: DeviceLikeType,
        encoder: str = "resnet34",
    ) -> None:
        model = self.architecture()(
            encoder_name=encoder,  # ResNet34 as the backbone
            encoder_weights="imagenet",  # Pre-trained on ImageNet
            in_channels=3,  # RGB input
            classes=59,  # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, lr, device)

    @abstractmethod
    @staticmethod
    def architecture() -> type:
        ...


class UNetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.Unet


class DeepLabModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.DeepLabV3Plus


class FPNModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.FPN


class LinkNetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.Linknet


class PSPNetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.PSPNet
