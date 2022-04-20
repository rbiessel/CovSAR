from matplotlib import pyplot as plt
import isceio as io
import argparse
import numpy as np


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the SLC.')
    args = parser.parse_args()

    return args


def main():
    args = readInputs()
    print(args.path)
    slc = io.load_stack([args.path])[0]

    plt.imshow(10 * np.log(np.abs(slc)**2),
               origin='lower', cmap=plt.cm.Greys_r, vmin=60, vmax=100)
    plt.show()


main()
