import logging
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import matplotlib.patches as mpl_patches
import scipy.stats as stats
import closures
import figStyle
import matplotlib as mpl
# import colorcet
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

data_root = '/Users/rbiessel/Documents/InSAR/plotData'

colors = ['steelblue', 'tomato', '#646484']


def main():

    pixel_paths = glob.glob(data_root + '/**/p_*/')
    pixel_paths = [
        path for path in pixel_paths if 'imnav' in path or 'DNWR' in path]

    # print(pixel_paths)
    keep = [4, 2, 9, 12]

    # pixel_paths = ['/Users/rbiessel/Documents/InSAR/plotData/DNWR/p_116_54/', '/Users/rbiessel/Documents/InSAR/plotData/DNWR/p_159_91/',
    #                '/Users/rbiessel/Documents/InSAR/plotData/imnav/p_75_62/', '/Users/rbiessel/Documents/InSAR/plotData/imnav/p_75_73/']

    from pub_pixels import pixel_paths

    plen = len(pixel_paths)
    plotlen = 1
    fig, axes = plt.subplots(nrows=plotlen, ncols=plen,
                             figsize=(13, 2.5))

    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

    for p in range(len(pixel_paths)):
        # ax = axes[1, p]
        path = pixel_paths[p]
        label = path.split('/')[-3]

        kappas = np.load(os.path.join(
            path[:-1] + '_kappa', 'kappas.npy'))

        R2 = np.load(os.path.join(
            path[:-1] + '_kappa', 'R2_kappas.npy'))

        axes[p].plot(kappas * 10, R2, color=colors[2], linewidth=3)
        axes[p].set_ylim([0, 0.8])
        if p > 0:
            axes[p].set_yticks([0, 0.2, 0.4, 0.6, 0.8], color='w')

        axes[p].grid(alpha=0.1)
        axes[p].set_xscale('log')
        axes[p].axvline(x=10, alpha=0.6, color='black', linewidth=1)

        axes[p].set_xlabel(r'$\kappa$')
        axes[p].set_title(f'({subplot_labels[p]})', loc='left')

        xticks_manual = [1e-1, 1e1, 1e3, 1e5]
        axes[p].set_xticks(xticks_manual, labels=xticks_manual)
        axes[p].get_xaxis().set_major_formatter(
            mpl.ticker.LogFormatterSciNotation())

    axes[0].set_ylabel(r'$R^2$')

    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=1.0)
    plt.savefig(
        '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/kappaOpt.png', dpi=300)
    plt.show()


main()
