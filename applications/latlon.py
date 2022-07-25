import numpy as np
from matplotlib import pyplot as plt


def latlon_to_index(latimage, lonimage, lat, lon):

    tomin = np.abs(latimage - lat) * np.abs(lonimage - lon)
    # plt.imshow(tomin)
    # plt.show()

    return np.unravel_index(
        np.argmin(tomin), (latimage.shape[0], latimage.shape[1]))
