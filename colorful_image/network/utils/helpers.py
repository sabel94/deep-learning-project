from matplotlib.pyplot import imsave
import numpy as np
from skimage import color


def save_images(images, path, group_index):
    """Helper to save images to path.

    Input:
        images:         Array with shape (batch_size, H, W, 3)
        path:           Location of storing images
        group_index:    The filenames first character(s) to group all provided
                        images together, e.g. group_index = 10 results in
                        path/10_1.png, path/10_2.png, ...
    """
    index = 0
    for image in images:
        # Convert image to RGB
        image_rgb = np.clip(color.lab2rgb(image), 0, 1) # range [0, 1]
        image_rgb = (255*image_rgb).astype('uint8') # range [0, 255]

        # Save image to path
        fname = path + "/" + str(group_index) + "_" + str(index) + ".png"
        imsave(fname, image_rgb)

        # Increment index
        index = index + 1