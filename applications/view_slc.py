from matplotlib import pyplot as plt
import isceio as io
import argparse
import numpy as np
from library import multilook, non_local_complex


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the SLC.')
    parser.add_argument('-uav', '--uavsar', action='store_true',
                        help='Load a UAVSAR SLC instead of ISCE XML')
    args = parser.parse_args()

    return args


def main():
    args = readInputs()
    print(args.path)
    if not args.uavsar:
        slc = io.load_stack([args.path])[0]
    else:
        slc = io.load_stack_uavsar([args.path], cols=11699, rows=4822)[0]
    ml = True

    slc = slc * slc.conj()
    mled = multilook(slc, ml=(4, 1), thin=(1, 1))
    # nl = non_local_complex(slc, sig=1000)

    image = np.log10(np.abs(mled))

    image[np.isnan(image)] = 0
    image[np.isinf(image)] = 0

    mean = np.nanmean(image)
    std = 1.5 * np.nanstd(image)

    print(mean, std)

    plt.imshow(image, cmap=plt.cm.Greys_r, vmin=mean - std, vmax=mean+std)
    plt.show()


main()
