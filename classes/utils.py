import matplotlib.pyplot as plt

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
