import logging
from this import d
from matplotlib import dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import matplotlib.patches as mpl_patches
import scipy.stats as stats
from triplets import eval_triplets
import figStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from datetime import datetime
import colorcet as cc
import closures
from covariance import CovarianceMatrix
from matplotlib.cm import get_cmap
import library as sarlab
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

data_root = '/Users/rbiessel/Documents/InSAR/plotData'

# colsbg = ['#19192a', '#626282', '#aaaabe', '#cbcbd7']

cmap_div = get_cmap('cet_diverging_bwr_20_95_c54')
cmap_cont = get_cmap('cet_linear_grey_0_100_c0')
cmap_cont = cc.cm.fire
cmap_cont = cc.cm.dimgray
# cmap_cont = get_cmap('cet_linear_protanopic_deuteranopic_kbw_5_98_c40')


def get_optimum_kappa(intensities, cphases, triplets):
    ''''
    returns the optimum kappa for the given intensities and closure phases
    '''
    kappas = np.linspace(-4, 4, 100)
    kappas = 10**kappas
    R2s = np.zeros(kappas.shape)

    itriplets = np.zeros(len(triplets))

    for k in range(len(kappas)):
        for i in range(len(triplets)):
            triplet = triplets[i]

            itriplets[i] = sarlab.intensity_closure(
                intensities[triplet[0]], intensities[triplet[1]], intensities[triplet[2]], norm=False, cubic=False, filter=1, kappa=kappas[k], function='tanh')

        r, pval = stats.pearsonr(
            itriplets.flatten(), np.angle(cphases).flatten())
        R2s[k] = r**2

    kappa = kappas[np.argmax(R2s)]
    print(f'Found Optimum kappa: {kappa}')
    return kappas, R2s


def main():

    from pub_pixels import pixel_paths

    for pixel in pixel_paths:
        print(pixel)
        try:
            C = np.load(os.path.join(pixel, 'C_raw.np.npy'))
        except:
            continue
        kappas = np.linspace(0.001, 20, 70)
        kappas = np.linspace(-2, 3, 100)
        kappas = 10**kappas
        # kappas = np.array([0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 10, 20])
        R2 = np.zeros((len(kappas)))
        ms = np.zeros((len(kappas)))
        bs = np.zeros((len(kappas)))
        residuals = np.zeros((len(kappas)))
        rmse_c = np.zeros((len(kappas)))

        variances = np.zeros((len(kappas)))
        for kappa, k in zip(kappas, range(len(R2))):
            # C = C[np.newaxis, np.newaxis, :, :]
            triplets = closures.get_triplets(C.shape[0])

            closure_stack = np.zeros((
                len(triplets), C.shape[0], C.shape[1]), dtype=np.complex64)

            amp_triplet_stack = closure_stack.copy()
            amp_triplet_stack = amp_triplet_stack.astype(np.float64)

            intensity = np.load(os.path.join(
                pixel_paths[0], 'Intensities.np.npy'))

            for i in range(len(triplets)):
                triplet = triplets[i]

                try:
                    closure = C[triplet[0], triplet[1]] * C[triplet[1],
                                                            triplet[2]] * C[triplet[0], triplet[2]].conj()

                    amp_triplet = sarlab.intensity_closure(
                        intensity[triplet[0]], intensity[triplet[1]], intensity[triplet[2]], norm=False, cubic=False, filter=1, kappa=kappa, function='arctan')
                except:
                    continue

                closure_stack[i] = closure
                amp_triplet_stack[i] = amp_triplet

            closure_stack[np.isnan(closure_stack)] = 0
            amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

            r, pval = stats.pearsonr(
                amp_triplet_stack.flatten(), np.angle(closure_stack).flatten())

            results = np.polyfit(amp_triplet_stack.flatten(),
                                 np.angle(closure_stack).flatten(), deg=1, full=True)
            poly = results[0]
            residual = np.sqrt(results[1] / len(kappas))

            def get_rsme(predicted, observed):
                loss = 2 * (1 - np.cos(predicted - observed))
                return np.sqrt(np.sum(loss**2) / len(predicted))

            rmse = get_rsme(np.polyval(poly, amp_triplet_stack.flatten()), np.angle(
                closure_stack).flatten())

            R2[k] = r**2
            ms[k] = poly[0]
            bs[k] = poly[1]
            rmse_c[k] = rmse
            residuals[k] = residual

            # plt.scatter(amp_triplet_stack, np.angle(closure_stack))
            # plt.title(f'kappa={kappa}')
            # plt.show()
        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].set_xscale('log')
        ax[0].plot(kappas * 10, R2, '--o')
        ax[0].axvline(x=10)

        ax[0].set_xlabel('kappa')
        ax[0].set_ylabel('R2')

        ax[1].set_xscale('log')
        # ax[1].set_yscale('symlog')

        ax[1].plot(kappas * 10, ms, '--o')
        ax[1].set_ylim((-1, 1))
        ax[1].set_xlabel('kappa')
        ax[1].set_ylabel('m')
        ax[1].axvline(x=10)

        ax[2].set_xscale('log')
        ax[2].plot(kappas * 10, bs, '--o')
        ax[2].set_xlabel('kappa')
        ax[2].set_ylabel('b')
        ax[2].axvline(x=10)

        ax[3].set_xscale('log')
        # ax[3].plot(kappas * 10, residuals, '--o')
        ax[3].plot(kappas * 10, rmse_c, '--o')

        ax[3].set_xlabel('kappa')
        ax[3].set_ylabel('residuals')
        ax[3].axvline(x=10)

        plt.show()

        # np.save(os.path.join(pixel, 'R2_kappa_i'), R2)
        # np.save(os.path.join(pixel, 'kappa_i'), kappas)


if __name__ == '__main__':
    main()
