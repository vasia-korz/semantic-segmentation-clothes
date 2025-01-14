from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, hsv_to_rgb
from torch._prims_common import DeviceLikeType


def visualize(pair: tuple, num_classes: int = 59) -> None:
    image, mask = pair

    colormap = _get_custom_colormap(num_classes)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=colormap, vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.show()


def visualize_prediction(
    model: Any,
    dataset_sample: tuple,
    device: DeviceLikeType,
    class_labels: dict | None = None,
    num_classes: int = 59,
) -> None:
    image, mask = dataset_sample

    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    colormap = _get_custom_colormap(num_classes)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().numpy(), cmap=colormap, vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap=colormap, vmin=0, vmax=num_classes - 1)
    plt.title("Predicted")
    plt.axis("off")

    plt.show()

    if class_labels:
        print("\nClass Labels:")
        for idx, label in class_labels.items():
            print(f"{idx}: {label}")


def _get_custom_colormap(num_classes: int) -> ListedColormap:
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    saturation, value = 0.65, 0.85
    colors = hsv_to_rgb(
        np.stack(
            [hues, np.full_like(hues, saturation), np.full_like(hues, value)],
            axis=1,
        )
    )
    colors[0] = hsv_to_rgb([0, 0.1, 0.99])
    colormap = ListedColormap(colors)

    return colormap
