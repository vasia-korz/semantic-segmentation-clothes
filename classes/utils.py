import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, ListedColormap
import numpy as np
import torch

def visualize(pair):
    image, mask = pair

    image = image.permute(1, 2, 0)

    _, ax = plt.subplots(1, 2, figsize=(7, 5))

    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask)
    ax[1].set_title("Mask")
    ax[1].axis('off')
    
    plt.show()


def visualize_prediction(model, dataset_sample, device, class_labels=None, num_classes=59):
    image, mask = dataset_sample

    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    hues = np.linspace(0, 1, num_classes, endpoint=False)
    saturation, value = 0.65, 0.85
    colors = hsv_to_rgb(np.stack([hues, np.full_like(hues, saturation), np.full_like(hues, value)], axis=1))
    colormap = ListedColormap(colors)

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