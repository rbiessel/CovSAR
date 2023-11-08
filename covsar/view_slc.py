from matplotlib import pyplot as plt
import isceio as io
import argparse
import numpy as np
from library import multilook, non_local_complex
import colorcet as cc
from matplotlib.cm import get_cmap

cmap = get_cmap("cet_CET_C7")


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the SLC.')
    parser.add_argument('-uav', '--uavsar', action='store_true',
                        help='Load a UAVSAR SLC instead of ISCE XML')
    parser.add_argument('-a', '--angle', action='store_true',
                        help='Show angle instead')
    args = parser.parse_args()

    return args


def main():
    args = readInputs()
    print(args.path)
    if not args.uavsar:
        slc = io.load_stack([args.path])[0]
    else:
        # 26631
        # 9900
        slc = io.load_stack_uavsar([args.path], cols=26631, rows=3300)[0]
    ml = True

    if not args.angle:
        slc = slc * slc.conj()
    mled = multilook(slc, ml=(1, 1), thin=(1, 1))
    # nl = non_local_complex(slc, sig=1000)

    if not args.angle:
        image = np.log10(np.abs(mled * mled.conj()))

        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0

        mean = np.nanmean(image)
        std = 1.5 * np.nanstd(image)

        print(mean, std)

        plt.imshow(image, cmap=plt.cm.Greys_r, vmin=mean - std, vmax=mean+std)
        plt.show()
    else:

        # 10800
        # 11050

        # 550
        # 800

        plt.imshow(np.angle(mled * np.exp(1j * 2)),
                   cmap=cmap, interpolation='None')
        c = plt.colorbar()
        c.ax.set_title('[rad]')
        plt.show()


main()
