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
        'path', help='Path to the folder.')

    parser.add_argument('-p', '--pols', nargs='+',
                        help='Polarizations', required=True)

    args = parser.parse_args()
    return args


def main():

    args = readInputs()
    print(args.path)
    print(args.pols)

    pols = args.pols

    cols = 26631
    rows = 3300

    stack = None
    for i in range(len(pols)):
        path = args.path.replace('VV', pols[i])
        if stack is None:
            ifgrm = io.load_stack_uavsar([path], cols=cols, rows=rows)[0]
            stack = np.zeros(
                (len(pols), ifgrm.shape[0], ifgrm.shape[1]), dtype=np.complex64)
            stack[i] = ifgrm
        else:
            stack[i] = io.load_stack_uavsar(
                [path], cols=cols, rows=rows)[0]

    fig, axes = plt.subplots(nrows=len(pols), ncols=len(
        pols), sharex=True, sharey=True)
    print(stack.shape)
    for i in range(stack.shape[0]):
        for j in range(stack.shape[0]):
            if len(pols) == 1:
                ax = axes
            else:
                ax = axes[i, j]

            # Plot Interferograms on diagonal
            if i == j:
                mled = multilook(stack[i], ml=(1, 1), thin=(1, 1))
                ax.imshow(np.angle(mled),
                          cmap=cmap, interpolation='None')
            # Plot double interferograms off diagonal
            elif j > i:
                rng = np.pi/2
                mled = multilook(
                    stack[i] * stack[j].conj(), ml=(2, 2), thin=(1, 1))
                ax.imshow(np.angle(mled),
                          cmap=plt.cm.seismic, interpolation='None', vmin=-rng, vmax=rng)

            elif i > j:
                ax.remove()

    for i in range(stack.shape[0]):
        for j in range(stack.shape[0]):
            if len(pols) == 1:
                ax = axes
            else:
                ax = axes[i, j]
            if i == 0:
                ax.set_title(pols[j])

            if i == j:
                ax.set_ylabel(pols[i])

    plt.tight_layout()
    plt.show()


main()
