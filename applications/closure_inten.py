from latlon import latlon_to_index
from cgitb import small
from distutils.log import error
from email.errors import FirstHeaderLineIsContinuationDefect
import logging
from operator import index
from random import sample
from turtle import width
from numpy.core.arrayprint import _leading_trailing
from numpy.lib.polynomial import polyval
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from scipy import special
from bootstrapCov import bootstrap_correlation
from evd import write_timeseries, write_geometry
import library as sarlab
from datetime import datetime as dt
import argparse
import glob
import os
import shutil
from covariance import CovarianceMatrix
import isceio as io
import closures
import scipy.stats as stats
from pl.nn import nearest_neighbor
from interpolate_phase import interpolate_phase_intensity
from scipy.stats import gaussian_kde
from greg import linking as MLE
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpl_patches
from greg import simulation as greg_sim
from triplets import eval_triplets
from plot_bootstrapHist import plot_hist
import figStyle
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the interferogram.')
    parser.add_argument('-l', '--layndcover', type=str, dest='landcover',
                        required=False, help='Landcover file path')
    parser.add_argument('-o', '--output', type=str,
                        dest='output', required=True, help='Output folder to save timeseries to')
    parser.add_argument('-L', '--Looks', type=int,
                        dest='lf', required=True, help='Number of Look Factors')

    parser.add_argument('-P', '--PL', type=str,
                        dest='phaselinking', required=False, default='EIG', help='Phase linking algorithm: "EIG", "MLE", or "NN"')
    parser.add_argument('-label', '--label', type=str,
                        dest='label', required=False, default=None, help='Number of Look Factors')
    parser.add_argument('-s', '--save', dest='saveData', required=False,
                        action='store_true', help='Save data at specific pixels for plotting later')
    parser.add_argument('-plat', '--platform', type=str,
                        dest='platform', required=False, default='S1', help='platform')
    args = parser.parse_args()
    return args


def main():
    inputs = readInputs()
    stack_path = inputs.path
    stack_path = os.path.expanduser(stack_path)
    outputs = inputs.output
    files = glob.glob(os.path.join(stack_path, './**/*.slc.full'), recursive=True) + \
        glob.glob(os.path.join(stack_path, './**/*.slc'), recursive=True)
    files = sorted(files)
    print(os.path.join(inputs.path, '/**/*.slc'))
    assert len(files) >= 1
    # files = files[0:5]

    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    if inputs.landcover:
        landcover = np.squeeze(rasterio.open(inputs.landcover).read())
        print(landcover.shape)

    geom_path = os.path.join(os.path.dirname(
        files[0]), '../../geom_reference/')

    assert os.path.exists(geom_path)

    SLCs = io.load_stack_vrt(files)

    clip = None

    clip = [0, -1, 0, -1]
    # clip = [4000, 8000, 0, -1]
    # clip = [0, -1, 1000, 2000]

    # clip = [0, 1000, 0, 1000]
    SLCs = SLCs[:, clip[0]:clip[1], clip[2]:clip[3]]
    lf = inputs.lf
    N = SLCs.shape[0]

    if inputs.platform == 'S1':
        ml_size = (7*lf, 19*lf)
        sample_size = (2 * lf, 5*lf)

    elif inputs.platform == 'UAVSAR':
        ml_size = (5 * lf, 5 * lf)
        sample_size = (2 * lf, 2 * lf)

    n = ml_size[0] * ml_size[1]  # Number of looks
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=sample_size)

    SLCs = None

    coherence = cov.get_coherence()
    uncorrected = coherence.copy()
    intensity = cov.get_intensity()

    # points = np.indices(intensity[:, :, 0].shape)

    # points = points[:, ::100, ::100]
    # points = points.reshape((2, (points.shape[1] * points.shape[2]))).T
    # points = points[1:]
    # print(points)

    lats = io.load_geom_from_slc(files[0], file='lat')[
        ::sample_size[0], ::sample_size[1]]
    lons = io.load_geom_from_slc(files[0], file='lon')[
        ::sample_size[0], ::sample_size[1]]

    if 'imnav' in inputs.label:
        lat_weather = 68.62
        lon_weather = -149.3
        weather_point = latlon_to_index(lats, lons, lat_weather, lon_weather)
        print(weather_point)
        points = np.array([[weather_point[0], weather_point[1]], [
                          75, 62], [85, 43], [75, 73]])
    if 'dalton' in inputs.label:
        points = np.array(
            [[24, 73], [150, 115], [729, 122], [777, 125], [400, 202], [482, 141]])

    # points = np.array([[153, 81], [116, 54], [75, 171]])

    if 'delta' in inputs.label:
        points = np.array([[10, 10], [50, 50]])

    if 'DNWR' in inputs.label or 'dnwr' in inputs.label:
        lat_weather = 36.4381
        lon_weather = -115.3601
        weather_point = latlon_to_index(lats, lons, lat_weather, lon_weather)
        print(weather_point)
        points = np.array([[156, 92], [159, 91], [163, 83], [116, 54], [75, 171], [
            weather_point[0], weather_point[1]]])
        # points = np.array([[10, 100], [25, 75]])

    # DALTON POINTS
    # points = np.array([[34, 69], [201, 69], [189, 338], [123, 402], [91, 316]])

    # points = np.array([[10, 55], [15, 55], [20, 55]])
    for i in range(intensity.shape[2]):
        intensity[:, :, i] = sarlab.multilook(intensity[:, :, i], ml=(2, 2))

    plt.imshow(np.median(intensity[:, :, :], axis=2), cmap=plt.cm.Greys)
    plt.scatter(points[:, 0], points[:, 1], s=30,
                facecolors='none', edgecolors='r')
    plt.show()
    # m = -2
    # coherence_m = sarlab.reduce_cov(coherence, keep_diag=m)
    # print(np.abs(coherence_m[:, :, 10, 10]))
    # return
    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0], all=False)
    cumulative_mask = closures.cumulative_mask(triplets)

    baslines_a = np.floor(np.array([(triplet[2] - triplet[0])
                                    for triplet in triplets]) * 12)

    baslines_b = np.array([triplet[2] - triplet[1]
                           for triplet in triplets]) * 12

    A, rank = closures.build_A(
        triplets, coherence)

    U, S, Vh = np.linalg.svd(A)
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]

    AdagA = A_dagger @ A
    print(np.round(AdagA, 2))

    if not inputs.landcover:
        landcover = np.zeros((coherence.shape[2], coherence.shape[3]))

    closure_stack = np.zeros((
        len(triplets), coherence.shape[2], coherence.shape[3]), dtype=np.complex64)

    landcover_types = np.unique(landcover)

    amp_triplet_stack = closure_stack.copy()
    amp_triplet_stack = amp_triplet_stack.astype(np.float64)
    amp_triplet_stack_legacy = amp_triplet_stack.copy()

    amp_difference_stack = np.zeros(
        (len(triplets), 3, coherence.shape[2], coherence.shape[3]))
    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        filter_strength = 4
        closure = sarlab.multilook(closure, ml=(
            filter_strength, filter_strength))

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(np.median(intensity[:, :, :], axis=2), cmap=plt.cm.Greys)
        ax[1].imshow(np.angle(closure), cmap=plt.cm.seismic)
        plt.show()

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, filter=1, inc=None)

        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet

    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    rs = np.zeros(landcover.shape)

    ps = np.zeros(landcover.shape)

    degree = 1
    power = 1

    print('Estimating relationship')
    poly = np.zeros((landcover.shape[0], landcover.shape[1], degree + 1))

    for j in range(amp_triplet_stack.shape[1]):
        for i in range(amp_triplet_stack.shape[2]):
            l = landcover[j, i]
            if l == 2 or l != 2:

                intensities = intensity[j, i, :]
                raw_intensities = intensities
                intensities = np.tile(intensities, (len(intensities), 1))
                intensities = 10 * (intensities.T - intensities)
                ws = 0
                window_closure = closure_stack[:, j-ws:j+ws+1, i-ws:i+ws+1]
                window_amps = amp_triplet_stack[:,
                                                j-ws:j+ws+1, i-ws:i+ws+1]  # [mask]

                if len(window_amps.flatten()) > 2 and len(window_closure.flatten()) > 2:
                    try:
                        r, p = stats.pearsonr(
                            window_amps.flatten(), np.angle(window_closure).flatten())

                        r = r
                        fitform = 'linear'

                        coeff, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                        ), W=None, function=fitform)
                    except:
                        print(window_amps.flatten())
                        print('')
                        print(np.angle(window_closure).flatten())
                        r = 0
                        p = 0
                        coeff = [0, 0]

                    do_huber = False

                    poly[j, i, :] = coeff

                    rs[j, i] = r
                    ps[j, i] = p

                    # modeled systematic closures
                    if np.abs(r) >= 0:

                        est_closures = closures.eval_sytstematic_closure(
                            amp_triplet_stack[:, j, i], model=coeff, form='linear')

                        est_closures_int = closures.eval_sytstematic_closure(
                            amp_triplet_stack[:, j, i], model=coeff, form='lineari')

                        systematic_phi_errors = closures.least_norm(
                            A, est_closures, pinv=False, pseudo_inv=A_dagger)

                        systematic_phi_errors_int = closures.least_norm(
                            A, est_closures_int, pinv=False, pseudo_inv=A_dagger)

                        uncorrected_phi_errors = closures.least_norm(
                            A, np.random.normal(loc=-0.02, scale=0.05, size=len(
                                closure_stack[:, j, i].flatten())), pinv=False, pseudo_inv=A_dagger)

                        uncorrected_phi_errors = closures.least_norm(
                            A, closure_stack[:, j, i].flatten(), pinv=False, pseudo_inv=A_dagger)

                        error_coh = closures.phivec_to_coherence(
                            systematic_phi_errors, coherence[:, :, j, i].shape[0])

                        error_coh_int = closures.phivec_to_coherence(
                            systematic_phi_errors_int, coherence[:, :, j, i].shape[0])
                        error_coh_unc = closures.phivec_to_coherence(
                            uncorrected_phi_errors, coherence[:, :, j, i].shape[0])

                        # gradient = interpolate_phase_intensity(
                        #     raw_intensities, error_coh)

                        # gradient = 0
                        # linear_phase = np.exp(
                        #     1j * (-1 * intensities * gradient))

                        # error_coh = error_coh * linear_phase

                        coherence[:, :, j, i] = coherence[:,
                                                          :, j, i] * error_coh.conj()

                        # coherence[:, :, j, i] = error_coh

                    # o
                    if np.abs(r) > 1 or ((points == np.array([i, j])).all(1).any() and inputs.saveData):

                        # cum_closures = np.cumsum(
                        #     np.angle(window_closure).flatten()[cumulative_mask])
                        # print(len(cum_closures))

                        # cum_closures_slope = np.cumsum(
                        #     est_closures[cumulative_mask])

                        # cum_closures_int = np.cumsum(
                        #     est_closures_int[cumulative_mask])

                        # inten_timeseries = raw_intensities - raw_intensities[0]

                        # plt.plot(cum_closures, label='Observed')
                        # plt.plot(cum_closures_slope, label='Just slope')
                        # plt.plot(cum_closures_int, label='Slope and Intercept')
                        # plt.plot(inten_timeseries[1:-1], label='Intensities')
                        # plt.legend(loc='best')
                        # plt.title('Cumulative Closure Phase')
                        # plt.show()
                        if inputs.saveData:
                            pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}/'

                            if os.path.exists(pixel_data_folder_path):
                                print('Output folder already exists, clearing it')
                                shutil.rmtree(pixel_data_folder_path)
                            os.mkdir(pixel_data_folder_path)

                        print(f'point: ({i}, {j})')
                        # gradient = interpolate_phase_intensity(
                        #     raw_intensities, error_coh_unc, plot=True)
                        # gradient = interpolate_phase_intensity(
                        #     raw_intensities, error_coh, plot=True)
                        print('COVARIANCE')
                        if False:
                            l = int(
                                np.floor(np.sqrt(ml_size[0] * ml_size[1]))/2)

                            decay = np.array(
                                [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
                            samples = 1000
                            rs_decays = np.zeros((decay.shape[0], samples))
                            for i in range(len(decay)):
                                C_decay = greg_sim.decay_model(
                                    R=1, L=l, P=cov.cov.shape[0], coh_decay=decay[i], coh_infty=0.05, returnC=True)
                                print(C_decay.shape)
                                rs_decay = bootstrap_correlation(
                                    C_decay, l, triplets, nsample=samples, fitLine=False)
                                rs_decays[i] = rs_decay

                            rs_sim, coeffs_sim = bootstrap_correlation(
                                cov.cov[:, :, j, i], l, triplets, nsample=1000, fitLine=True, zeroPhi=True)

                            plot_hist(rs_sim, r, rs_decays, decay)
                            fig, ax = plt.subplots(ncols=3, nrows=1)
                            bins = 100
                            ax[0].hist(rs_sim.flatten(), bins=bins)
                            ax[0].axvline(r, 0, 1, color='red')
                            ax[0].set_title('Rs')

                            ax[1].hist(coeffs_sim[0].flatten(), bins=bins)
                            ax[1].axvline(coeff[0], 0, 1, color='red')
                            ax[1].set_title('Slope')

                            ax[2].hist(coeffs_sim[1].flatten(), bins=bins)
                            ax[2].axvline(coeff[1], 0, 1, color='red')
                            ax[2].set_title('Mean Residual Phase')

                            plt.show()

                        # fig, ax = plt.subplots(nrows=1, ncols=2)
                        # n, bins, p = ax[0].hist(r2, bins=60)
                        # ax[0].axvline(r, 0, 1, color='red')
                        # ax[0].set_title('Phi = observed')
                        # ax[0].set_xlabel('Correlation Coefficient')

                        # n, bins, p = ax[1].hist(r2_phizero, bins=60)
                        # ax[1].axvline(r, 0, 1, color='red')

                        # ax[1].set_title('Phi = zero')
                        # ax[1].set_xlabel('Correlation Coefficient')

                        # plt.show()

                        x = np.linspace(window_amps.min() - 0.1 * np.abs(window_amps.min()),
                                        window_amps.max() + 0.1 * np.abs(window_amps.max()), 100)

                        fig, ax = plt.subplots(figsize=(5, 2.5))

                        # xy = np.vstack(
                        #     [window_amps.flatten(), np.angle(window_closure).flatten()])
                        # z = gaussian_kde(xy)(xy)
                        ax.scatter(window_amps.flatten(), np.angle(
                            window_closure).flatten(), s=10)

                        ax.plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form=fitform), '--', label='Fit: mx')

                        ax.plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form='lineari'), '--', label='Fit: mx+b')
                        ax.axhline(y=0, color='k', alpha=0.1)
                        ax.axvline(x=0, color='k', alpha=0.1)
                        ax.set_xlabel('Amplitude Triplet')
                        ax.set_ylabel('Closure Phase (rad)')

                        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                                         lw=0, alpha=0)] * 2

                        # create the corresponding number of labels (= the text you want to display)
                        labels = []
                        labels.append(f'R$^{{2}} = {{{np.round(r**2, 2)}}}$')

                        # create the legend, supressing the blank space of the empty line symbol and the
                        # padding between symbol and label by setting handlelenght and handletextpad
                        ax.legend(handles, labels, loc='best', fontsize='large',
                                  fancybox=True, framealpha=0.7,
                                  handlelength=0, handletextpad=0)

                        plt.tight_layout()
                        plt.savefig(os.path.join(
                            pixel_data_folder_path, 'scatter.png'), dpi=200)
                        np.save(os.path.join(pixel_data_folder_path,
                                             'ampTriplets.np'), window_amps.flatten())
                        np.save(os.path.join(pixel_data_folder_path, 'closures.np'), np.angle(
                            window_closure).flatten())
                        np.save(os.path.join(
                            pixel_data_folder_path, 'coeff.np'), coeff)
                        np.save(os.path.join(
                            pixel_data_folder_path, 'C_raw.np'), uncorrected[:, :, j, i])
                        np.save(os.path.join(
                            pixel_data_folder_path, 'C_ln_slope.np'), error_coh)
                        np.save(os.path.join(
                            pixel_data_folder_path, 'C_ln_unc.np'), error_coh_unc)
                        np.save(os.path.join(
                            pixel_data_folder_path, 'Intensities.np'), raw_intensities)

                        # plt.show()

                        fig, ax = plt.subplots(
                            nrows=3, ncols=1, figsize=(5, 5))
                        # slope_stderr = np.sqrt(covm[0][0])
                        # intercept_stderr = np.sqrt(covm[1][1])

                        xy = np.vstack(
                            [window_amps.flatten(), np.angle(window_closure).flatten()])
                        z = gaussian_kde(xy)(xy)

                        ax[0].scatter(window_amps.flatten(), np.angle(
                            window_closure).flatten(), c=z, s=10)  # alpha=(alphas)**(1))

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form=fitform), '--', label='Fit: mx')

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form='lineari'), '--', label='Fit: mx+b')
                        ax[0].set_title(f'r: {np.round(r, 3)}')
                        ax[0].set_ylabel(r'$\Xi$')
                        ax[0].set_xlabel(
                            'Intensity Triplet')
                        ax[0].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        ax[1].set_xlabel('Intensity Ratio (dB)')
                        ax[1].set_ylabel('Estimated Phase Error (rad)')

                        iratios = closures.coherence_to_phivec(intensities)

                        nl_phases_uncorrected = np.angle(
                            closures.coherence_to_phivec(error_coh_unc))

                        nlphases = np.angle(
                            closures.coherence_to_phivec(error_coh))

                        nlphases_int = np.angle(
                            closures.coherence_to_phivec(error_coh_int))

                        # print(m)
                        x = np.linspace(iratios.min(
                        ) - 0.1 * np.abs(iratios.min()), iratios.max() + 0.1 * np.abs(iratios.max()), 100)

                        max_range = np.max(np.abs(iratios))  # + 0.1
                        x = np.linspace(-max_range, max_range, 100)

                        # ax[1].scatter(iratios, nlphases, s=15,
                        #               marker='x', color='blue', label='/w a linear component to force monotonicity')

                        # ax[1].scatter(iratios, nl_phases_uncorrected, s=10,
                        #               marker='x', color='black', label='From Uncorrected Closures')

                        # ax[1].scatter(iratios, nlphases_int, s=15,
                        #               marker='x', color='orange', label='With Intercept')

                        ax[1].scatter(iratios, nlphases, s=20,
                                      marker='x', color='blue', label='mx')
                        ax[1].axhline(y=0, color='k', alpha=0.1)
                        ax[1].axvline(x=0, color='k', alpha=0.1)

                        ax[1].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        residual_closure = np.angle(window_closure).flatten() - closures.eval_sytstematic_closure(
                            window_amps.flatten(), coeff, form='linear')

                        indexsort = np.argsort(baslines_b)
                        residual_closure = residual_closure[indexsort]
                        baslines_b = baslines_b[indexsort]

                        u, s = np.unique(baslines_b, return_index=True)
                        split_residuals_b = np.split(residual_closure, s[1:])

                        u, s = np.unique(baslines_b, return_index=True)
                        split_residuals_b = np.split(residual_closure, s[1:])

                        # ax[2].scatter(
                        #     baslines_a, residual_closure, s=10, label='a')
                        ax[2].boxplot(split_residuals_b,
                                      positions=u, widths=9)
                        # ax[2].boxplot(split_residuals_b)

                        # ax[2].scatter(
                        #     baslines_b, residual_closure, s=10, label='b', alpha=0.5)
                        ax[2].set_xlabel('basline')
                        ax[2].set_ylabel('Closures')
                        ax[2].legend(loc='lower right')

                        plt.tight_layout()
                        plt.show()

                        fig, ax = plt.subplots(ncols=2, nrows=1)
                        ax[0].hist(np.angle(window_closure).flatten(), bins=50)
                        # ax[0].set_title()
                        ax[1].hist(window_amps.flatten(), bins=50)
                        ax[1].set_title('Amp Triplet')

                        plt.show()

                        fig, ax = plt.subplots(nrows=1, ncols=2)

                        residual = np.angle(window_closure.flatten() *
                                            np.exp(1j * -1 * est_closures))
                        ax[0].hist(
                            np.angle(np.exp(1j*est_closures)).flatten(), bins=60, density=True)
                        ax[0].set_title('Predicted')

                        ax[1].hist(residual, bins=60, density=True)
                        ax[1].set_title(
                            f'Residual Mean: {np.round(np.mean(residual), 2)} -- Intercept: {np.round(coeff[1], 2)}')
                        plt.show()

                        fig, ax = plt.subplots(nrows=1, ncols=3)
                        ax[0].set_title(
                            'Estimated Phase Error -- slope')

                        ax[0].imshow(np.angle(error_coh),
                                     vmin=-np.pi/15, vmax=np.pi/15, cmap=plt.cm.seismic)

                        ax[0].set_xlabel('Reference Image')
                        ax[0].set_ylabel('Secondary Image')
                        ax[1].set_title(
                            'Estimated Phase Error -- intercept')
                        im = ax[1].imshow(np.angle(error_coh_int * error_coh.conj()),
                                          vmin=-np.pi/15, vmax=np.pi/15, cmap=plt.cm.seismic)

                        ax[1].set_xlabel('Reference Image')
                        ax[1].set_ylabel('Secondary Image')

                        ax[2].set_title(
                            'Estimated Phase Error -- all')
                        im = ax[2].imshow(
                            np.angle(error_coh_unc), vmin=-np.pi/5, vmax=np.pi/5, cmap=plt.cm.seismic)

                        ax[2].set_xlabel('Reference Image')
                        ax[2].set_ylabel('Secondary Image')

                        fig.subplots_adjust(right=0.8)
                        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                        fig.colorbar(im, cax=cbar_ax,
                                     label='Estimated Nonlinear Phase Error (rad)')
                        plt.show()

                    else:
                        r = 0
                    # except:
                    #     # print('robust regression failed :(')
                    #     poly[j, i, :] = np.zeros((2))
                    #     rs[j, i] = 0
            if (j % 1) == 0 and i == 0:
                print(
                    f'Phase Closure Correction Progress: {(j/amp_triplet_stack.shape[1]* 100)}%')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(poly[:, :, 1], cmap=plt.cm.seismic, vmin=-1, vmax=1)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.colorbar(im, ax=ax, orientation='horizontal')
    plt.savefig('/Users/rbiessel/Documents/dalton_intercept.png',
                transparent=True, dpi=300)
    plt.show()

    ax = plt.subplot()
    im = ax.imshow(rs**2, vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.scatter(points[:, 0], points[:, 1], s=30,
               facecolors='none', edgecolors='black')
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/dalton_rsquared.png',
                transparent=True, dpi=300)
    plt.show()

    TS_method = inputs.phaselinking

    for m in range(N, N+1):
        coherence_m = sarlab.reduce_cov(coherence, keep_diag=m)
        uncorrected_m = sarlab.reduce_cov(uncorrected, keep_diag=m)

        if TS_method == 'MLE':
            timeseries = MLE.EMI_py_stack(coherence_m)
            uncorrected_ts = MLE.EMI_py_stack(uncorrected_m)
            sm_ts = nearest_neighbor(coherence_m * uncorrected_m.conj())
        elif TS_method == 'EIG':
            timeseries = sarlab.eig_decomp(coherence_m)
            uncorrected_ts = sarlab.eig_decomp(uncorrected_m)
            sm_ts = nearest_neighbor(coherence_m * uncorrected_m.conj())
        elif TS_method == 'NN':
            timeseries = nearest_neighbor(coherence_m)
            uncorrected_ts = nearest_neighbor(uncorrected_m)
            sm_ts = nearest_neighbor(coherence_m * uncorrected_m.conj())

        normed_dif = (np.angle((timeseries * uncorrected_ts.conj()))
                      ) * 55.465763 / (4 * np.pi)

        avg_coherence = sarlab.mean_coh(uncorrected)

        plt.imshow(avg_coherence)
        plt.show()

        if inputs.saveData:
            for pixel in points:
                pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{pixel[0]}_{pixel[1]}/'
                fig, ax = plt.subplots(figsize=(5, 2.5))

                plt.plot(normed_dif[pixel[1], pixel[0]])
                plt.ylabel('mm')
                plt.xlabel('Time (days)')
                plt.tight_layout()

                plt.savefig(os.path.join(
                    pixel_data_folder_path, f'displacementDiff_{m}.png'), dpi=200)
                # plt.show()
                np.save(os.path.join(pixel_data_folder_path, f'dispDiff_{m}.np'),
                        normed_dif[pixel[1], pixel[0]])

                # Plot residual
                print(timeseries.shape)
                ts_pred = uncorrected_ts[pixel[1], pixel[0]]
                ts_pred = np.squeeze(ts_pred)[:, np.newaxis]
                print(ts_pred.shape)
                C_pred = ts_pred @ ts_pred.T.conj()
                C_unc = uncorrected[:, :, pixel[1], pixel[0]]
                print(C_unc.shape, C_pred.shape)

                # fig, ax = plt.subplots(ncols=2, nrows=1)
                # ax[0].imshow(np.angle(C_pred), cmap=plt.cm.seismic)
                # ax[1].imshow(np.angle(C_unc * C_pred.conj()),
                #              cmap=plt.cm.seismic)
                # plt.show()

    normed_dif_sum = np.sum(normed_dif, axis=2)
    plt.imshow(normed_dif_sum, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    plt.show()

    cov = None

    if os.path.exists(os.path.join(os.getcwd(), outputs)):
        print('Output folder already exists, clearing it')
        shutil.rmtree(os.path.join(os.getcwd(), outputs))

    print('creating output folder')
    os.mkdir(os.path.join(os.getcwd(), outputs))

    # Write new geometry and update path
    out_geom = os.path.join(os.getcwd(), outputs,
                            './geom_subset/')

    write_geometry(geom_path, out_geom, clip=clip, sample_size=sample_size)
    geom_path = out_geom

    for i in range(poly.shape[2]):
        path = os.path.join(os.getcwd(), outputs,
                            f'./degree_{i}.fit')
        io.write_image(path, poly[:, :, i].astype(
            np.float32), geocode=geom_path)

    r_path = os.path.join(os.getcwd(), outputs,
                          './correlation.fit')

    dif_path = os.path.join(os.getcwd(), outputs,
                            './sum_difference.fit')
    coherence_path = os.path.join(os.getcwd(), outputs,
                                  './average_coherence.fit')

    cumu_dif_path = os.path.join(os.getcwd(), outputs,
                                 './cumulative_difference.fit')
    io.write_image(r_path, rs.astype(np.float32), geocode=geom_path)
    io.write_image(dif_path, normed_dif_sum.astype(
        np.float32), geocode=geom_path)
    io.write_image(coherence_path, avg_coherence.astype(
        np.float32), geocode=geom_path)
    io.write_image(cumu_dif_path, normed_dif[:, :, -1].astype(
        np.float32), geocode=geom_path)

    poly = None
    rs = None

    do_write_timeseries = True

    if do_write_timeseries:

        corrected_timeseries = timeseries.conj()
        uncorrected_ts = uncorrected_ts.conj()

        kappa_uncorrected = sarlab.compute_tc(coherence, uncorrected_ts)
        kappa_corrected = sarlab.compute_tc(coherence, corrected_timeseries)

        corrected_path = os.path.join(os.getcwd(), outputs,
                                      './corrected_timeseries/')

        uncorrected_path = os.path.join(os.getcwd(), outputs,
                                        './raw_timeseries/')

        sm_path = os.path.join(os.getcwd(), outputs,
                               './sm_ts/')

        write_timeseries(corrected_timeseries, dates,
                         kappa_corrected, corrected_path, geocode=geom_path)
        write_timeseries(uncorrected_ts, dates,
                         kappa_uncorrected, uncorrected_path, geocode=geom_path)
        write_timeseries(sm_ts, dates,
                         kappa_uncorrected, sm_path, geocode=geom_path)


main()
