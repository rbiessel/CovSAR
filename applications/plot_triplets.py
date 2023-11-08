import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib import colors
import glob
import figStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable

divColor = plt.cm.RdBu_r
divColor = plt.cm.seismic
# divColor = plt.cm.PuOr


def main():
    import isceio as io

    stack = 'vegas'

    if stack == 'vegas':
        folder = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures/triplets'
        nrows = 2
        ncols = 1
    elif stack == 'dalton':
        folder = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/closures/triplets'
        ncols = 2
        nrows = 1

    cbar_kwargs = {
        'orientation': 'vertical',
        'location': 'right',
        'pad': 0.05,
        'aspect': 12,
        'fraction': 0.025,
    }
    tickSize = 8

    triplets = glob.glob(folder + '/phase_*.vrt')

    for triplet in triplets:
        print(triplet)
        indices = triplet.replace('.vrt', '').split('_')
        i1 = indices[-3]
        i2 = indices[-2]
        i3 = indices[-1]

        phase_path = triplet.replace('.vrt', '')
        intensity_path = phase_path.replace('phase', 'intensity')

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 4))
        phase = np.flip(io.load_file(phase_path), axis=1)
        intensity = np.flip(io.load_file(intensity_path), axis=1)

        # Phase
        phase_trip_im = ax[0].imshow(phase, cmap=divColor, norm=colors.CenteredNorm(
            halfrange=1.5), interpolation='none')

        divider = make_axes_locatable(ax[0])
        caxls = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(phase_trip_im, cax=caxls, ticks=[
            -1, -0.5, 0, 0.5, 1])

        print(i1, i2, i3)
        clb.ax.tick_params(labelsize=tickSize)
        clb.ax.set_title('[rad]', fontsize=tickSize)
        ax[0].set_title(r'(a) Phase Triplet: $\Xi_{{ {0}, {1}, {2}}}$'.format(
            i1, i2, i3), loc='left', fontsize=10)

        # Phase
        intensity_triplet_im = ax[1].imshow(intensity, cmap=divColor, norm=colors.CenteredNorm(
            halfrange=1.5), interpolation='none')

        divider = make_axes_locatable(ax[1])
        caxls = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(intensity_triplet_im, cax=caxls, ticks=[
            -1, -0.5, 0, 0.5, 1])

        clb.ax.tick_params(labelsize=tickSize)
        clb.ax.set_title('[dB]', fontsize=tickSize)
        ax[1].set_title(r'(b) Intensity Triplet: $\mathfrak{{S}}_{{ {0}, {1}, {2}}}$'.format(
            i1, i2, i3), loc='left', fontsize=10)

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        plt.tight_layout()
        plt.savefig(
            f'/Users/rbiessel/Documents/triplets/{stack}/triplets_{stack}_{i1}_{i2}_{i3}.jpg', dpi=300)
        # plt.show()


main()
