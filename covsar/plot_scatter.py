import logging
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import matplotlib.patches as mpl_patches
import scipy.stats as stats
import closures
import figStyle
# import colorcet
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

data_root = '/Users/rbiessel/Documents/InSAR/plotData'


def main():
    from pub_pixels import pixel_paths

    plen = len(pixel_paths)
    plotlen = 2
    fig, axes = plt.subplots(nrows=plotlen, ncols=plen,
                             figsize=(13, 4))

    subplot_labels = [['a', 'b', 'c', 'd', 'e', 'f'],
                      ['g', 'h', 'i', 'j', 'k', 'l']]

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

        sig = np.load(os.path.join(path, 'R_sigma_p.npy'))

        axes[0, p].scatter(ampTriplets, phaseClosures,
                           s=10, color='black', alpha=0.3)

        xrange = np.max(np.abs(ampTriplets)) + np.std(ampTriplets)

        x = np.linspace(-xrange, xrange, 100)

        axes[0, p].plot(x, closures.eval_sytstematic_closure(
            x, coeff, form='linear'), linewidth=2.5, alpha=0.9, color='tomato', label='Fit: mx')

        axes[0, p].plot(x, closures.eval_sytstematic_closure(
            x, coeff, form='lineari'), '--', linewidth=2, color='steelblue', label='Fit: mx+b')

        # axes[0, p].set_title(label)
        axes[0, p].set_xlabel(r'$\mathfrak{S}$ [$\mathrm{-}$]')
        axes[0, 0].set_ylabel(r'$\Xi$ [$\mathrm{rad}$]')
        axes[0, p].axhline(y=0, color='k', alpha=0.15)
        axes[0, p].axvline(x=0, color='k', alpha=0.15)
        axes[0, p].grid(alpha=0.2)

        max_closure = np.max(np.abs(phaseClosures))
        buf = np.std(phaseClosures) * 3
        yticks = [-np.round(max_closure, 1), 0, np.round(max_closure, 1)]
        axes[0, p].set_yticks(yticks, labels=yticks)
        axes[0, p].set_ylim(
            [-max_closure - 0.05, max_closure + buf])

        xticks = [-np.round(xrange/1.5, 1), 0, np.round(xrange/1.5, 1)]
        print(xticks)
        axes[0, p].set_xticks(xticks, labels=xticks)

        labels = []

        if sig[2] > 0.5:
            sig[2] = 1-sig[2]
        labels.append(f'$R^{{2}} = {{{np.round(r**2, 2)}}}$')
        if sig[2] <= 1e-4:
            labels.append(r'$p \le 10^{-4}$')
        else:
            labels.append(r'$p=' + str(np.round(sig[2], 10)) + '$')
        axes[0, p].set_title(f'({subplot_labels[0][p]})', loc='left')

        handles = [mpl_patches.Rectangle((0, 0), 2, 2, fc=None, ec="white",
                                         lw=0, alpha=0)] * 2
        # create the legend, supressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        axes[0, p].legend(handles, labels, loc='best', fontsize=12,
                          fancybox=True, framealpha=0,
                          handlelength=0, handletextpad=0, borderpad=0)

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

        differences_kappa = np.load(os.path.join(
            path[0:-1] + '_kappa',  'dispDiff.np.npy'))

        kappas = np.load(os.path.join(
            path[0:-1] + '_kappa', 'kappas.npy'))

        R2s = np.load(os.path.join(
            path[0:-1] + '_kappa', 'R2_kappas.npy'))

        kappa_opt = np.round(10 * kappas[np.argmax(R2s)], decimals=1)

        tbaseline = 12
        if 'vegas' in label or 'DNWR' in label:
            tbaseline = 6

        days = np.arange(0, len(differences), 1) * tbaseline

        axes[1, p].plot(days, differences_kappa, '--o', linewidth=2,
                        markersize=3.5, color='#646484', label=r'$\kappa=' + str(kappa_opt) + '$')
        axes[1, p].plot(days, differences, '-o', linewidth=2,
                        markersize=3.5, color='black')
        axes[1, p].set_xlabel(r'$\Delta t$ [$\mathrm{days}$]')
        axes[1, 0].set_ylabel(r'$\Delta s$ [$\mathrm{mm}$]')
        axes[1, p].grid(alpha=0.2)
        leg = axes[1, p].legend(loc='best',
                                labelcolor='linecolor', facecolor='None', frameon=False, fontsize=14, borderpad=-0.1, labelspacing=0, handlelength=0, borderaxespad=0.24)

        leg._legend_box.align = "right"

        for item in leg.legendHandles:
            item.set_visible(False)

        axes[1, p].set_title(f'({subplot_labels[1][p]})', loc='left')

        yticks = [np.round(np.min(differences_kappa), 1), 0,
                  np.round(np.max(differences_kappa), 1)]
        if np.max(np.abs(differences_kappa)) < 0.1:
            yticks = [np.round(np.min(differences_kappa), 2), 0,
                      np.round(np.max(differences_kappa), 2)]

        if np.max(yticks) <= 0:
            yticks = [0, np.round(np.min(differences_kappa)/2, 1),
                      np.round(np.min(differences_kappa), 1)]

        xticks = np.array([0, 1, 2, 3, 4]) * 24
        axes[1, p].set_yticks(yticks, labels=yticks)
        axes[1, p].set_xticks(xticks, labels=xticks)

        difrange = np.abs(np.max(differences_kappa) -
                          np.min(differences_kappa)) / 5

        axes[1, p].set_ylim([np.min(differences_kappa) - difrange,
                            np.max(differences_kappa) + difrange])

        # axes[1, p].tick_params(axis='both', which='major',
        #                        labelsize=12, labelcolor='black')

        # axes[1, p].spines['top'].set_visible(False)
        # axes[1, p].spines['right'].set_visible(False)

    yticks_manual = [-0.4, 0, 0.4, 0.8]
    axes[1, 0].set_yticks(yticks_manual, labels=yticks_manual)

    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=1.0)
    plt.savefig(
        '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/scatter.png', dpi=300)
    plt.show()


main()
