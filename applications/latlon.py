import numpy as np
from matplotlib import pyplot as plt


def latlon_to_index(latimage, lonimage, lat, lon):

    tomin = np.log10(np.abs(latimage - lat)**2 + np.abs(lonimage - lon)**2)
    x, y = np.unravel_index(
        np.argmin(tomin), (latimage.shape[0], latimage.shape[1]))
    # plt.imshow(tomin)
    # plt.scatter(y, x, color='black')
    # plt.show()
    return x, y
