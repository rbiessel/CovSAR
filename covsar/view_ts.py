import rasterio
from matplotlib import pyplot as plt
import gdal
import numpy as np
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage
import library as sarlab
from datetime import datetime as dt
import argparse
import glob
import os
import shutil


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the interferogram.')
    args = parser.parse_args()

    return args


def main():
    inputs = readInputs()
    stack_path = inputs.path
    stack_path = os.path.expanduser(stack_path)
    files = glob.glob(stack_path)
    files = sorted(files)

    stack = None
    for i in range(len(files)):
        print('Loading Unwrapped Interferogram...')
        im = createImage()
        im.load(files[i] + '.xml')
        mm = im.memMap()
        print(mm.shape)
        if stack is None:
            stack = np.zeros(
                (len(files), mm.shape[0], mm.shape[2]), dtype=np.float64)

        stack[i, :, :] = np.abs(mm[:, 1, :])

    stack = -1 * stack * 5.6 / (np.pi * 4)

    for i in range(stack.shape[0]):
        stack[i] -= stack[i, 697, 125]
        plt.imshow(stack[i])
        plt.show()

    for i in range(100):
        plt.plot(stack[:, i, i])
        plt.show()


main()
