import numpy as np
import skimage.util as skiutil
from matplotlib import pyplot as plt


def filter(interferogram, landcover, size, stat="mean", real=True):
    shape = interferogram.shape
    combined = np.zeros((2, shape[0], shape[1]))
    combined[0] = interferogram
    combined[1] = landcover
    padding = [[0, 0], [size, size], [size, size]]
    combined = np.pad(combined, pad_width=padding, mode='reflect')
    windows = skiutil.view_as_windows(combined, window_shape=(2, size, size))
    outshape = windows[0].shape
    windows = windows[0].reshape(
        (outshape[0] * outshape[1], 2, size, size))

    filtered = np.zeros((windows.shape[0]))
    count = np.zeros((windows.shape[0]))

    for i in range(len(windows)):
        pixel = windows[i]
        center_i = int(np.floor(size / 2))
        lc = pixel[1, center_i, center_i]

        matches = pixel[1] == lc

        if i > 1000 and not real:
            plt.hist(pixel[0].flatten(), bins=100)
            # plt.hist(pixel[0][matches].flatten(), bins=100)
            plt.show()

        if stat == 'mean':
            filtered[i] = np.mean(pixel[0][matches])
        elif stat == 'median':
            filtered[i] = np.median(pixel[0][matches])
        elif stat == 'variance':
            filtered[i] = np.var(pixel[0][matches])

        count[i] = len(pixel[0][pixel[1] == lc])

    filtered_padded = filtered.reshape((outshape[0], outshape[1]))
    count_padded = count.reshape((outshape[0], outshape[1]))

    depad = int(np.floor(size/2)) + 1
    filtered = filtered_padded[depad:-depad, depad:-depad]
    count = count_padded[depad:-depad, depad:-depad]

    return filtered, count
