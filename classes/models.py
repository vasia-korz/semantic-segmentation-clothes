import segmentation_models_pytorch as smp
from classes.clothes_model import ClothesModel

class UNetModel(ClothesModel):
    def __init__(self, loss_fn, optimizer, lr, device, encoder='resnet34'):
        model = smp.Unet(
            encoder_name=encoder,          # ResNet34 as the backbone
            encoder_weights="imagenet",    # Pre-trained on ImageNet
            in_channels=3,                 # RGB input
            classes=59                     # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, lr, device)


class DeepLabModel(ClothesModel):
    def __init__(self, loss_fn, optimizer, lr, device, encoder='resnet34'):
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,         # ResNet50 backbone
            encoder_weights="imagenet",   # Pre-trained weights
            in_channels=3,                # RGB input
            classes=59                    # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, lr, device)


class FPNModel(ClothesModel):
    def __init__(self, loss_fn, optimizer, lr, device, encoder='resnet34'):
        model = smp.FPN(
            encoder_name=encoder,           # EfficientNet backbone
            encoder_weights="imagenet",     # Pre-trained weights
            in_channels=3,                  # RGB input
            classes=59                      # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, lr, device)


class LinkNetModel(ClothesModel):
    def __init__(self, loss_fn, optimizer, lr, device, encoder='resnet34'):
        model = smp.Linknet(
            encoder_name=encoder,        # DenseNet backbone
            encoder_weights="imagenet",  # Pre-trained weights
            in_channels=3,               # RGB input
            classes=59                   # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, lr, device)


class PSPNetModel(ClothesModel):
    def __init__(self, loss_fn, optimizer, lr, device, encoder='resnet34'):
        model = smp.PSPNet(
            encoder_name=encoder,        # ResNet50 backbone
            encoder_weights="imagenet",  # Pre-trained weights
            in_channels=3,               # RGB input
            classes=59                   # Number of classes
        )
        super().__init__(model, loss_fn, optimizer, lr, device)
