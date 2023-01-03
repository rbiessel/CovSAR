import logging
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import matplotlib.patches as mpl_patches
import scipy.stats as stats
import closures
# import colorcet
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

data_root = '/Users/rbiessel/Documents/InSAR/plotData'
globfigparams = {
    'fontsize': 12, 'family': 'serif', 'usetex': True,
    'preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{times} \usepackage{mathtools}',
    'column_inch': 229.8775 / 72.27, 'markersize': 24, 'markercolour': '#AA00AA',
    'fontcolour': 'black', 'tickdirection': 'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1}

# colsbg = ['#19192a', '#626282', '#aaaabe', '#cbcbd7']
plt.rc('font', **
       {'size': globfigparams['fontsize']})
plt.rcParams['text.usetex'] = globfigparams['usetex']
plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
plt.rcParams['font.size'] = globfigparams['fontsize']
plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
plt.rcParams['xtick.color'] = globfigparams['fontcolour']
plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
plt.rcParams['ytick.color'] = globfigparams['fontcolour']
plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']
plt.rcParams['axes.linewidth'] = 0.5


def main():

    pixel_paths = glob.glob(data_root + '/**/p_*/')
    pixel_paths = [
        path for path in pixel_paths if 'imnav' in path or 'DNWR' in path]

    # print(pixel_paths)
    keep = [4, 2, 9, 12]

    pixel_paths = [pixel_paths[i] for i in keep]

    plen = len(pixel_paths)
    plotlen = 2
    fig, axes = plt.subplots(nrows=plotlen, ncols=plen,
                             figsize=(12, 4))

    # PLOT SCATTER
    for p in range(len(pixel_paths)):
        # ax = axes[0, p]
        path = pixel_paths[p]
        label = pixel_paths[p].split('/')[-3]
        ampTriplets = np.load(os.path.join(
            path, 'ampTriplets.np.npy'))
        phaseClosures = np.load(os.path.join(path, 'closures.np.npy'))
        coeff = np.load(os.path.join(path, 'coeff.np.npy'))

        r, pval = stats.pearsonr(ampTriplets, phaseClosures)
        axes[0, p].scatter(ampTriplets, phaseClosures,
                           s=10, color='black', alpha=0.3)

        x = np.linspace(ampTriplets.min() - 0.1 * np.abs(ampTriplets.min()),
                        ampTriplets.max() + 0.1 * np.abs(ampTriplets.max()), 100)

        axes[0, p].plot(x, closures.eval_sytstematic_closure(
            x, coeff, form='linear'), linewidth=2.5, alpha=0.8, color='tomato', label='Fit: mx')

        axes[0, p].plot(x, closures.eval_sytstematic_closure(
            x, coeff, form='lineari'), '--', linewidth=2, color='steelblue', label='Fit: mx+b')

        axes[0, p].set_title(label)
        axes[0, p].set_xlabel(r'$\mathfrak{S} [$-$]  $')
        axes[0, p].set_ylabel(r'$\Xi$ [$rad$]')
        axes[0, p].axhline(y=0, color='k', alpha=0.15)
        axes[0, p].axvline(x=0, color='k', alpha=0.15)
        axes[0, p].grid(alpha=0.2)

        labels = []
        labels.append(f'R$^{{2}} = {{{np.round(r**2, 2)}}}$')

        handles = [mpl_patches.Rectangle((0, 0), 2, 2, fc="white", ec="white",
                                         lw=0, alpha=0)] * 2
        # create the legend, supressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        axes[0, p].legend(handles, labels, loc='best', fontsize='large',
                          fancybox=True, framealpha=0.7,
                          handlelength=0, handletextpad=0)

        # axes[0, p].spines['top'].set_visible(False)
        # axes[0, p].spines['right'].set_visible(False)
        # axes[0, p].spines['bottom'].set_visible(False)
        # axes[0, p].spines['left'].set_visible(False)

    # PLOT Difference
    for p in range(len(pixel_paths)):
        # ax = axes[1, p]
        path = pixel_paths[p]
        label = path.split('/')[-3]

        differences = np.load(os.path.join(
            path, 'dispDiff.np.npy'))

        tbaseline = 12
        if 'vegas' in label:
            tbaseline = 6

        days = np.arange(0, len(differences), 1) * tbaseline
        axes[1, p].plot(days, differences, linewidth=2, color='#646484')
        axes[1, p].set_xlabel(r'$\Delta t$ [days]')
        axes[1, p].set_ylabel(r'$\Delta \Theta$ [$mm$]')
        axes[1, p].grid(alpha=0.2)
        # axes[1, p].spines['top'].set_visible(False)
        # axes[1, p].spines['right'].set_visible(False)

    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=1.0)
    plt.savefig(
        '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/scatter.png', dpi=300)
    plt.show()


main()
