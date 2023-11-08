from latlon import latlon_to_index
from cgitb import small
from distutils.log import error
import logging
from operator import index
from random import sample
from turtle import width
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
from scipy.stats import gaussian_kde
from greg import linking as MLE
import matplotlib.patches as mpl_patches
from greg import simulation as greg_sim
from triplets import eval_triplets
import seaborn as sns
import optimize_kappa as ok
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
    parser.add_argument('-k', '--kappa', dest='kappaOpt', required=False,
                        action='store_true', help='optimize kappa for select pixels')
    parser.add_argument('-tf', '--tripletform', type=str,
                        dest='tripletform', required=False, default='tanh', help='Function used to generate intensity triplets')
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
    files = files[:20]

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

    points = np.array([[0, 0], [1, 1]])

    SLCs = SLCs[:, clip[0]:clip[1], clip[2]:clip[3]]
    lf = inputs.lf
    N = SLCs.shape[0]

    if inputs.platform == 'S1':
        ml_size = (7*lf, 19*lf)
        ml_size = (1*lf, 7*lf)
        lf = 4
        sample_size = (2 * (lf-0), 5*(lf-0))
        # sample_size = (2, 5)

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

    lats = io.load_geom_from_slc(files[0], file='lat')[
        ::sample_size[0], ::sample_size[1]]
    lons = io.load_geom_from_slc(files[0], file='lon')[
        ::sample_size[0], ::sample_size[1]]

    if 'imnav' in inputs.label:
        lat_weather = 68.62
        lon_weather = -149.3
        weather_point = latlon_to_index(lats, lons, lat_weather, lon_weather)
        print(weather_point)
        points = np.array(
            [[weather_point[1], weather_point[0]], [44, 113], [42, 115], [41, 115], [42, 114]])
        points = np.array(
            [[weather_point[1], weather_point[0]], [45, 91], [27, 88], [49, 106], [34, 87], [22, 14]])
        points = np.array(
            [[weather_point[1], weather_point[0]], [90, 27], [33, 83], [23, 89], [27, 90], [28, 87], [28, 89], [28, 90], [27, 89], [26, 89], [29, 87], [30, 90], [29, 88], [29, 90]])

    if 'vegas_east' in inputs.label:
        points = np.array([[10, 10], [50, 50], [34, 17], [17, 34], [563, 233]])

    if 'dalton' in inputs.label:
        points = np.array(
            [[24, 73], [777, 125], [400, 202], [482, 141]])

        points = np.array([[153, 81], [116, 54], [75, 171], [
                          36, 424], [171, 366], [147, 368], [177, 361], [185, 339], [207, 373], [33, 411], [49, 418], [117, 103], [182, 218], [218, 182]])

    # if 'delta' or 'kaktovik' in inputs.label:
    #     points = np.array([[10, 10], [50, 50]])
    #     print()

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
        intensity[:, :, i] = sarlab.multilook(intensity[:, :, i], ml=(1, 1))

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

    amp_triplet_stack = closure_stack.copy()
    amp_triplet_stack = amp_triplet_stack.astype(np.float64)

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        filter_strength = 2
        closure = sarlab.multilook(closure, ml=(
            filter_strength, filter_strength))

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, filter=1,  function=inputs.tripletform, kappa=1)

        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet

    intensity_triplet_variance = np.var(amp_triplet_stack, 0)
    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    rs = np.zeros(landcover.shape)
    rsme_linear = np.zeros(landcover.shape)

    degree = 1

    print('Estimating relationship')
    poly = np.zeros((landcover.shape[0], landcover.shape[1], degree + 1))

    def get_rsme(predicted, observed):
        return np.sqrt(np.sum(np.angle(np.exp(1j * (predicted - observed))) ** 2) / len(predicted))

    for j in range(amp_triplet_stack.shape[1]):
        for i in range(amp_triplet_stack.shape[2]):
            l = landcover[j, i]
            if l == 2 or l != 2:

                intensities = intensity[j, i, :]
                raw_intensities = intensities
                intensities = np.tile(intensities, (len(intensities), 1))
                intensities = 10 * (intensities.T - intensities)
                ws = 0
                window_closure = closure_stack[:, j, i]
                window_amps = amp_triplet_stack[:, j, i]

                if len(window_amps.flatten()) > 2 and len(window_closure.flatten()) > 2:
                    # try:
                    r, p = stats.pearsonr(
                        window_amps.flatten(), np.angle(window_closure).flatten())

                    fitform = 'linear'

                    clin, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    ), W=None, function=fitform)

                    if (points == np.array([i, j])).all(1).any():
                        if inputs.saveData:
                            pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}/'
                            if inputs.kappaOpt:
                                pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}_kappa/'
                            if os.path.exists(pixel_data_folder_path):
                                print('Output folder already exists, clearing it')
                                shutil.rmtree(pixel_data_folder_path)
                            os.mkdir(pixel_data_folder_path)

                    if (points == np.array([i, j])).all(1).any() and inputs.kappaOpt:
                        print('COMPUTING RESULTS FOR OPTIMUM KAPPA')
                        print('first: ', (points ==
                              np.array([i, j])).all(1).any())

                        kappas, R2s = ok.get_optimum_kappa(
                            raw_intensities, window_closure, triplets)

                        pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}/'
                        if inputs.kappaOpt:
                            pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}_kappa/'

                        np.save(os.path.join(
                            pixel_data_folder_path, 'R2_kappas'), R2s)
                        np.save(os.path.join(
                            pixel_data_folder_path, 'kappas'), kappas)
                        kappa_opt = kappas[np.argmax(R2s)]

                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        ax.set_xscale('log')
                        ax.plot(kappas * 10, R2s, '--o')

                        ax.set_xlabel('kappa')
                        ax.set_ylabel('residuals')
                        ax.axvline(x=10)
                        ax.axvline(x=kappa_opt * 10)

                        plt.show()

                        print(f'Optimum kappa = {kappa_opt}')

                        for t in range(len(triplets)):
                            triplet = triplets[t]

                            window_amps[t] = sarlab.intensity_closure(
                                raw_intensities[triplet[0]], raw_intensities[triplet[1]], raw_intensities[triplet[2]], norm=False, cubic=False, filter=1, kappa=kappa_opt, function='tanh')

                        r, pval = stats.pearsonr(
                            window_amps.flatten(), np.angle(window_closure).flatten())

                        clin, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                        ), W=None, function=fitform)
                        # coeff = clin
                        print(clin, r)

                    poly[j, i, :] = clin
                    rs[j, i] = r
                    coeff = clin

                    # except:
                    #     r = 0
                    #     p = 0
                    #     coeff = [0, 0]

                    est_closures_lin = closures.eval_sytstematic_closure(
                        amp_triplet_stack[:, j, i], model=clin, form='linear')

                    rsme_linear[j, i] = get_rsme(
                        est_closures_lin, window_closure.flatten())

                    systematic_phi_errors = closures.least_norm(
                        A, est_closures_lin, pinv=False, pseudo_inv=A_dagger)

                    # uncorrected_phi_errors = closures.least_norm(
                    #     A, np.random.normal(loc=-0.02, scale=0.05, size=len(
                    #         closure_stack[:, j, i].flatten())), pinv=False, pseudo_inv=A_dagger)

                    # uncorrected_phi_errors = closures.least_norm(
                    #     A, closure_stack[:, j, i].flatten(), pinv=False, pseudo_inv=A_dagger)

                    error_coh = closures.phivec_to_coherence(
                        systematic_phi_errors, coherence[:, :, j, i].shape[0])

                    # error_coh_unc = closures.phivec_to_coherence(
                    #     uncorrected_phi_errors, coherence[:, :, j, i].shape[0])

                    coherence[:, :, j, i] = coherence[:,
                                                      :, j, i] * error_coh.conj()

                    # print((points == np.array([i, j])).all(1).any())
                    if (points == np.array([i, j])).all(1).any():
                        # print('is working?????????????')
                        # if inputs.saveData:
                        #     pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}/'
                        #     if inputs.kappaOpt:
                        #         pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}_kappa/'
                        #     if os.path.exists(pixel_data_folder_path):
                        #         print('Output folder already exists, clearing it')
                        #         shutil.rmtree(pixel_data_folder_path)
                        #     os.mkdir(pixel_data_folder_path)

                        print(f'point: ({i}, {j})')

                        x = np.linspace(window_amps.min() - 0.1 * np.abs(window_amps.min()),
                                        window_amps.max() + 0.1 * np.abs(window_amps.max()), 100)

                        fig, ax = plt.subplots(figsize=(5, 2.5))

                        ax.scatter(window_amps.flatten(), np.angle(
                            window_closure).flatten(), s=10)

                        ax.plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form=fitform), '--', label='Fit: mx')

                        ax.plot(x, closures.eval_sytstematic_closure(
                            x, clin, form='lineari'), '--', label='mx+b')
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
                            pixel_data_folder_path, 'C_raw.np'), cov.cov[:, :, j, i])
                        np.save(os.path.join(
                            pixel_data_folder_path, 'C_ln_slope.np'), error_coh)
                        # np.save(os.path.join(
                        #     pixel_data_folder_path, 'C_ln_unc.np'), error_coh_unc)
                        np.save(os.path.join(
                            pixel_data_folder_path, 'Intensities.np'), raw_intensities)

                        lat = lats[j, i]
                        lon = lons[j, i]

                        np.save(os.path.join(
                            pixel_data_folder_path, 'latlon.np'), np.array([lat, lon]))
                        fig, ax = plt.subplots(
                            nrows=3, ncols=1, figsize=(5, 5))

                        xy = np.vstack(
                            [window_amps.flatten(), np.angle(window_closure).flatten()])
                        z = gaussian_kde(xy)(xy)

                        ax[0].scatter(window_amps.flatten(), np.angle(
                            window_closure).flatten(), c=z, s=10)  # alpha=(alphas)**(1))

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form=fitform), '--', label='Fit: nl')
                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, clin, form=fitform), '--', label='Fit: lin')

                        # ax[0].plot(x, closures.eval_sytstematic_closure(
                        #     x, coeff, form='lineari'), '--', label='Fit: mx+b')
                        ax[0].set_title(f'r: {np.round(r, 3)}')
                        ax[0].set_ylabel(r'$\Xi$')
                        ax[0].set_xlabel(
                            'Intensity Triplet')
                        ax[0].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        ax[1].set_xlabel('Intensity Ratio (dB)')
                        ax[1].set_ylabel('Estimated Phase Error (rad)')

                        iratios = closures.coherence_to_phivec(intensities)

                        # nl_phases_uncorrected = np.angle(
                        #     closures.coherence_to_phivec(error_coh_unc))

                        nlphases = np.angle(
                            closures.coherence_to_phivec(error_coh))

                        x = np.linspace(iratios.min(
                        ) - 0.1 * np.abs(iratios.min()), iratios.max() + 0.1 * np.abs(iratios.max()), 100)

                        max_range = np.max(np.abs(iratios))  # + 0.1
                        x = np.linspace(-max_range, max_range, 100)

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

                        ax[2].boxplot(split_residuals_b,
                                      positions=u, widths=9)
                        ax[2].set_xlabel('basline')
                        ax[2].set_ylabel('Closures')
                        ax[2].legend(loc='lower right')

                        plt.tight_layout()
                        plt.show()

                        # fig, ax = plt.subplots(ncols=2, nrows=1)
                        # sns.kdeplot(np.angle(window_closure).flatten(),
                        #             bw_adjust=0.5, ax=ax[0])
                        # ax[0].set_title('Closure Phases')
                        # sns.kdeplot(window_amps.flatten(),
                        #             bw_adjust=0.5, ax=ax[1])
                        # ax[1].set_title('Amp Triplet')

                        # plt.show()

                    else:
                        r = 0
            if (j % 1) == 0 and i == 0:
                print(
                    f'Phase Closure Correction Progress: {(j/amp_triplet_stack.shape[1]* 100)}%')

    TS_method = inputs.phaselinking

    for m in range(N, N+1):
        # m = 2
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

        for i in range(normed_dif.shape[2]):
            plt.imshow(normed_dif[:, :, i],
                       cmap=plt.cm.seismic, vmin=-3, vmax=3)
            plt.title(f'Difference, {i}')
            plt.show()
        # Compute phase bias and intensit rate

        x = np.arange(0, normed_dif.shape[2]) * 12
        print(x.shape)
        difference_flattened = normed_dif.reshape(
            normed_dif.shape[0] * normed_dif.shape[1], normed_dif.shape[2])

        print(intensity.shape)
        print(normed_dif.shape)
        intensity_flattened = intensity.reshape(
            normed_dif.shape[0] * normed_dif.shape[1], normed_dif.shape[2])

        intensity_rate = np.polyfit(x, intensity_flattened.T, 1)

        bias = np.polyfit(x, difference_flattened.T, 1)
        bias = bias[0, :].reshape(
            normed_dif.shape[0], normed_dif.shape[1])

        intensity_rate = intensity_rate[0, :].reshape(
            normed_dif.shape[0], normed_dif.shape[1])

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(bias * 365, vmin=-10, vmax=10, cmap=plt.cm.seismic)
        ax[1].imshow(intensity_rate * 365, vmin=-10,
                     vmax=10, cmap=plt.cm.seismic)

        ax[0].set_title('Velocity Bias')
        ax[1].set_title('Intensity Rate')
        plt.show()

        plt.scatter(intensity_rate.flatten() * 365, (bias * np.sign(poly[:, :, 0])).flatten()
                    * 365, alpha=0.5, color='black')
        plt.xlabel('Intensity Rate [dB/yr]')
        plt.ylabel('Velocity Bias [mm/yr]')
        plt.show()

        if inputs.saveData:
            for pixel in points:
                pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{pixel[0]}_{pixel[1]}/'
                if inputs.kappaOpt:
                    pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{pixel[0]}_{pixel[1]}_kappa/'
                fig, ax = plt.subplots(figsize=(5, 2.5))

                x = np.arange(0, normed_dif.shape[2])
                slope = np.polyfit(x, normed_dif[pixel[1], pixel[0]], 1)[0]
                plt.plot(x, normed_dif[pixel[1], pixel[0]])
                plt.plot(x, x * slope, '--')
                plt.ylabel('mm')
                plt.xlabel('Time (days)')
                plt.tight_layout()

                plt.savefig(os.path.join(
                    pixel_data_folder_path, f'displacementDiff.png'), dpi=200)
                # plt.show()
                np.save(os.path.join(pixel_data_folder_path, f'dispDiff.np'),
                        normed_dif[pixel[1], pixel[0]])

                # Plot residual
                print(timeseries.shape)
                ts_pred = uncorrected_ts[pixel[1], pixel[0]]
                ts_pred = np.squeeze(ts_pred)[:, np.newaxis]
                print(ts_pred.shape)
                C_pred = ts_pred @ ts_pred.T.conj()
                C_unc = uncorrected[:, :, pixel[1], pixel[0]]
                print(C_unc.shape, C_pred.shape)

                plt.show()
    max_dif = np.max(normed_dif, axis=2)
    min_dif = np.min(normed_dif, axis=2)
    max_difshape = max_dif.shape

    max_dif[np.where(np.abs(min_dif) > max_dif)
            ] = min_dif[np.where(np.abs(min_dif) > max_dif)]

    plt.imshow(max_dif, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    plt.title('Max Difference')
    plt.scatter(points[:, 0], points[:, 1], s=30,
                facecolors='none', edgecolors='b')
    plt.show()

    plt.imshow(rs, cmap=plt.cm.seismic, vmin=-1, vmax=1)
    plt.scatter(points[:, 0], points[:, 1], s=30,
                facecolors='none', edgecolors='b')
    plt.title('R')
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
                            './max_difference.fit')
    coherence_path = os.path.join(os.getcwd(), outputs,
                                  './average_coherence.fit')

    cumu_dif_path = os.path.join(os.getcwd(), outputs,
                                 './cumulative_difference.fit')
    bias_path = os.path.join(os.getcwd(), outputs,
                             './bias.fit')
    irate_path = os.path.join(os.getcwd(), outputs,
                              './irate.fit')

    rsme_linear_path = os.path.join(os.getcwd(), outputs,
                                    './rsme_linear.fit')

    io.write_image(r_path, rs.astype(np.float32), geocode=geom_path)
    io.write_image(dif_path, max_dif.astype(
        np.float32), geocode=geom_path)
    io.write_image(coherence_path, avg_coherence.astype(
        np.float32), geocode=geom_path)
    io.write_image(cumu_dif_path, normed_dif[:, :, -1].astype(
        np.float32), geocode=geom_path)
    io.write_image(bias_path, bias.astype(
        np.float32), geocode=geom_path)
    io.write_image(irate_path, intensity_rate.astype(
        np.float32), geocode=geom_path)

    io.write_image(rsme_linear_path, rsme_linear.astype(
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
