from abc import ABC, abstractmethod

import segmentation_models_pytorch as smp
from torch._prims_common import DeviceLikeType

from classes.clothes_model import ClothesModel, LossType, OptimType


class SegmentationModel(ClothesModel, ABC):
    def __init__(
        self,
        loss_fn: LossType,
        optimizer: OptimType,
        device: DeviceLikeType,
        lr: float = 1e-4,
        encoder: str = "resnet34",
    ) -> None:
        model = self.architecture()(
            encoder_name=encoder,  # ResNet34 as the backbone
            encoder_weights="imagenet",  # Pre-trained on ImageNet
            in_channels=3,  # RGB input
            classes=59,  # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, device, lr)

    @staticmethod
    @abstractmethod
    def architecture() -> type:
        ...


class UnetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.Unet
    

class UnetPlusPlusModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.UnetPlusPlus


class FPNModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.FPN


class PSPNetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.PSPNet


class DeepLabV3Model(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.DeepLabV3
    

class DeepLabV3PlusModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.DeepLabV3Plus


class LinknetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.Linknet


class MAnetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.MAnet


class PANModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.PAN


class UPerNetModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.UPerNet


class SegformerModel(SegmentationModel):
    @staticmethod
    def architecture() -> type:
        return smp.Segformer
