from typing import Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt


def visualize_outputs(*args: Tuple[Iterable], titles: Iterable = ()) -> None:
    r"""Helper function for visualizing arrays of related images.  Each input argument is expected to be an Iterable of
    images -- shape:  (batch, nchan, nrow, ncol).  Will handle both RGB and grayscale images. The i-th elements from all
    input arrays are displayed along a single row, with shared x- and y-axes for visualization.

    :param args: Iterables of related images to display.
    :param titles: Titles to display above each column.
    :return: None (plots the images with Matplotlib)
    """
    nrow, ncol = len(args[0]), len(args)
    fig, ax = plt.subplots(nrow, ncol, sharex='row', sharey='row', squeeze=False)

    for j, title in enumerate(titles[:ncol]):
        ax[0, j].set_title(title)

    for i, images in enumerate(zip(*args)):
        for j, image in enumerate(images):
            if len(image.shape) < 3:
                ax[i, j].imshow(image / image.max())
            else:
                ax[i, j].imshow(np.moveaxis(image, 0, -1) / image.max())

    plt.show()
