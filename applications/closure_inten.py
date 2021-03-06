from cgitb import small
from distutils.log import error
from email.errors import FirstHeaderLineIsContinuationDefect
from operator import index
from turtle import width
from numpy.core.arrayprint import _leading_trailing
from numpy.lib.polynomial import polyval
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from scipy import special
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
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge
from pl.nn import nearest_neighbor
from interpolate_phase import interpolate_phase_intensity
from scipy.stats import gaussian_kde
from scipy import optimize
from greg import linking as MLE


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the interferogram.')
    parser.add_argument('-l', '--landcover', type=str, dest='landcover',
                        required=False, help='Landcover file path')
    parser.add_argument('-o', '--output', type=str,
                        dest='output', required=True, help='Output folder to save timeseries to')
    args = parser.parse_args()
    return args


def main():
    inputs = readInputs()
    stack_path = inputs.path
    stack_path = os.path.expanduser(stack_path)
    outputs = inputs.output
    files = glob.glob(stack_path)
    files = sorted(files)
    files = files[1:7]
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

    n = 19
    lf = 4
    ml_size = (7*lf, 19*lf)
    n = ml_size[0] * ml_size[1]
    sample_size = (2 * lf, 5*lf)
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=sample_size)

    SLCs = None

    coherence = cov.get_coherence()
    uncorrected = coherence.copy()
    intensity = cov.get_intensity()

    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0], all=False)
    # triplets = triplets[0:int(k)]

    baslines_a = np.floor(np.array([(triplet[2] - triplet[0])
                                    for triplet in triplets]) * 12)

    baslines_b = np.array([triplet[2] - triplet[1]
                           for triplet in triplets]) * 12

    A, rank = closures.build_A(triplets, coherence)
    U, S, Vh = np.linalg.svd(A)
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]

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

        clml = (8, 8)
        closure = sarlab.multilook(closure, ml=clml)

        # mean_coherence = (np.abs(coherence[triplet[0], triplet[1]]) + np.abs(
        #     coherence[triplet[1], triplet[2]]) + np.abs(coherence[triplet[0], triplet[2]].conj())) / 3

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=True, cubic=False, filter=2)

        amp_triplet_legacy = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, legacy=True)

        deltaI1 = sarlab.logistic(
            intensity[:, :, triplet[1]] - intensity[:, :, triplet[0]])
        deltaI2 = sarlab.logistic(
            intensity[:, :, triplet[2]] - intensity[:, :, triplet[1]])
        deltaI3 = sarlab.logistic(
            intensity[:, :, triplet[0]] - intensity[:, :, triplet[2]])

        amp_difference_stack[i, :, :, :] = np.array(
            [deltaI1, deltaI2, deltaI3])

        # fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
        # ax[0].imshow(np.angle(closure), vmin=-np.pi/2,
        #              vmax=np.pi/2, cmap=plt.cm.seismic)
        # ax[1].imshow(np.sign(amp_triplet) *
        #              np.log10(np.abs(amp_triplet)), cmap=plt.cm.seismic)
        # plt.show()
        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet
        amp_triplet_stack_legacy[i] = amp_triplet_legacy

    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    rs = np.zeros(landcover.shape)
    rs_legacy = np.zeros(landcover.shape)

    ps = np.zeros(landcover.shape)
    ps_legacy = np.zeros(landcover.shape)

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
                intensities = intensities.T - intensities

                ws = 0
                window_closure = closure_stack[:, j-ws:j+ws+1, i-ws:i+ws+1]
                # mask = np.abs(np.angle(window_closure)) < np.pi/2

                # window_closure = window_closure  # [mask]
                window_amps = amp_triplet_stack[:,
                                                j-ws:j+ws+1, i-ws:i+ws+1]  # [mask]

                window_amps_legacy = amp_triplet_stack_legacy[:,
                                                              j-ws:j+ws+1, i-ws:i+ws+1]

                if len(window_amps.flatten()) > 2 and len(window_closure.flatten()) > 2:
                    r, p = stats.pearsonr(
                        np.angle(window_closure).flatten(), window_amps.flatten())

                    r = r**2
                    n = len(np.angle(window_closure).flatten())
                    r = 1 - (1 - r) * ((n - 1)/(n-2))

                    r_legacy, p_legacy = stats.pearsonr(
                        np.angle(window_closure).flatten(), window_amps_legacy.flatten())

                    r_legacy = r_legacy**2

                    fitform = 'linear'

                    coeff, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    ), W=None, function=fitform)

                    coeff_i, covm2 = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    ), C=None, W=None, function='lineari')

                    do_huber = False

                    poly[j, i, :] = coeff

                    rs[j, i] = r
                    ps[j, i] = p

                    rs_legacy[j, i] = r_legacy
                    ps_legacy[j, i] = p_legacy

                    # modeled systematic closures
                    if np.abs(r) >= 0:

                        est_closures = closures.eval_sytstematic_closure(
                            amp_triplet_stack[:, j, i], model=coeff, form=fitform)

                        est_closures_int = closures.eval_sytstematic_closure(
                            amp_triplet_stack[:, j, i], model=coeff_i, form='lineari')

                        systematic_phi_errors = closures.least_norm(
                            A, est_closures, pinv=False, pseudo_inv=A_dagger)

                        systematic_phi_errors_int = closures.least_norm(
                            A, est_closures_int, pinv=False, pseudo_inv=A_dagger)

                        uncorrected_phi_errors = closures.least_norm(
                            A, closure_stack[:, j, i], pinv=False, pseudo_inv=A_dagger)

                        error_coh = closures.phivec_to_coherence(
                            systematic_phi_errors, coherence[:, :, j, i].shape[0])

                        error_coh_int = closures.phivec_to_coherence(
                            systematic_phi_errors_int, coherence[:, :, j, i].shape[0])
                        error_coh_unc = closures.phivec_to_coherence(
                            uncorrected_phi_errors, coherence[:, :, j, i].shape[0])

                        # gradient = interpolate_phase_intensity(
                        #     raw_intensities, error_coh)

                        gradient = 0
                        linear_phase = np.exp(
                            1j * (-2 * intensities * gradient))

                        error_coh = error_coh * linear_phase

                        coherence[:, :, j, i] = coherence[:,
                                                          :, j, i] * error_coh.conj()

                        # coherence[:, :, j, i] = error_coh

                    if np.abs(r) > 0.99:
                        # gradient = interpolate_phase_intensity(
                        #     raw_intensities, error_coh, plot=True)
                        # gradient = interpolate_phase_intensity(
                        #     raw_intensities, error_coh_int, plot=True)

                        # fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')

                        # # print(triplets.shape)
                        # # print(sm[triplets[:, 0]])

                        # x = amp_difference_stack[:, 0, j, i]
                        # y = amp_difference_stack[:, 1, j, i]
                        # z = amp_difference_stack[:, 2, j, i]
                        # # c = closures.eval_sytstematic_closure(
                        # #     amp_triplets.flatten(), coeff, form='linear')
                        # c2 = np.angle(window_closure).flatten()

                        # maxc = np.abs(np.max(c2))

                        # # img = ax.scatter(x, y, z, c=c, cmap=plt.cm.seismic, vmin=-maxc, vmax=maxc)
                        # img = ax.scatter(
                        #     x, y, z, c=c2, cmap=plt.cm.seismic, vmin=-maxc, vmax=maxc)

                        # fig.colorbar(img)
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

                        x = np.linspace(
                            window_amps.min() - 0.1 * np.abs(window_amps.min()), window_amps.max() + 0.1 * np.abs(window_amps.max()), 100)

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form=fitform), '--', label='Closure Fit')

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff_i, form='lineari'), '--', label='Closure Fit w/ intercept')

                        # ax[0].plot(x, sarlab.gen_logistic(x, logistic_params[0], logistic_params[1],
                        #                                   logistic_params[2]), '--', label='Closure Fit w/ intercept')

                        # conf95_slope = 1.96 * slope_stderr
                        # conf95_intercept = 1.96 * intercept_stderr

                        # print(
                        #     f'Confidence interval of the intercept: {coeff[1]} +/- {conf95_intercept}')
                        # print(
                        #     f'Confidence interval of the Slope: {coeff[0]} +/- {conf95_slope}')
                        # # ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

                        # ax[0].fill_between(x, closures.eval_sytstematic_closure(
                        #     x, coeff - conf95_slope, form=fitform), closures.eval_sytstematic_closure(
                        #     x, coeff + conf95_slope, form=fitform), alpha=0.2, color='orange', label='95% Confidence Region')

                        ax[0].set_title(f'r: {np.round(r, 3)}')
                        ax[0].set_ylabel('Closure Phase (rad)')
                        ax[0].set_xlabel(
                            'Intensity Ratio Triple Product (dB^3)')
                        ax[0].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        ax[1].set_xlabel('Intensity Ratio (dB)')
                        ax[1].set_ylabel('Estimated Phase Error (rad)')

                        iratios = closures.coherence_to_phivec(intensities)

                        # nlphases_int = np.angle(
                        #     systematic_phi_errors_int).flatten()

                        nl_phases_uncorrected = np.angle(
                            closures.coherence_to_phivec(error_coh_unc))
                        # # m = sarlab.fit_phase_ratio(
                        #     iratios, nlphases, degree=4)

                        # polycoeff = np.polyfit(
                        #     iratios, nlphases_int, 3)

                        # polydir = np.polyder(polycoeff)
                        # slope_to_add = polydir[-1]

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

                        ax[1].scatter(iratios, nl_phases_uncorrected, s=10,
                                      marker='x', color='black', label='From Uncorrected Closures')

                        ax[1].scatter(iratios, nlphases_int, s=15,
                                      marker='x', color='orange', label='With Intercept')

                        ax[1].scatter(iratios, nlphases, s=20,
                                      marker='x', color='blue', label='Intercept removed')

                        # ax[1].plot(x, np.polyval(m, x), '--',
                        #            label='nonlinear poly fit')
                        # ax[1].plot(x, np.polyval(polycoeff, x),
                        #            '--', label='Full poly fit')

                        # ax[1].plot(x, np.polyval(polycoeff, x) - (2 * slope_to_add * x),
                        #            '--', label='Monotonic poly fit')

                        ax[1].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        residual_closure = np.angle(window_closure).flatten() - closures.eval_sytstematic_closure(
                            window_amps.flatten(), coeff_i, form='lineari')

                        indexsort = np.argsort(baslines_a)
                        residual_closure = residual_closure[indexsort]
                        baslines_a = baslines_a[indexsort]

                        u, s = np.unique(baslines_a, return_index=True)
                        split_residuals_a = np.split(residual_closure, s[1:])

                        print(u)

                        u, s = np.unique(baslines_b, return_index=True)
                        split_residuals_b = np.split(residual_closure, s[1:])

                        # ax[2].scatter(
                        #     baslines_a, residual_closure, s=10, label='a')
                        ax[2].boxplot(split_residuals_a,
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
                        ax[0].set_title(
                            'Estimated Phase Error (w/o intercept)')

                        ax[0].imshow(np.angle(error_coh),
                                     vmin=-np.pi/5, vmax=np.pi/5, cmap=plt.cm.seismic)

                        ax[0].set_xlabel('Reference Image')
                        ax[0].set_ylabel('Secondary Image')
                        ax[1].set_title(
                            'Estimated Phase Error (w/ intercept)')
                        im = ax[1].imshow(np.angle(error_coh_int),
                                          vmin=-np.pi/5, vmax=np.pi/5, cmap=plt.cm.seismic)

                        ax[1].set_xlabel('Reference Image')
                        ax[1].set_ylabel('Secondary Image')

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

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].hist(ps_legacy.flatten(), alpha=0.5, bins=100,
    #            label='P Values - Triple product')
    # ax[1].hist(np.abs(rs_legacy).flatten(), alpha=0.5, bins=100,
    #            label='|Correlations| - triple product')

    # ax[0].hist(ps.flatten(), alpha=0.5, bins=100,
    #            label='P Values - Linear')
    # ax[1].hist(np.abs(rs).flatten(), alpha=0.5, bins=100,
    #            label='|Correlations| - Linear')

    # ax[0].legend(loc='upper right')
    # ax[1].legend(loc='upper right')

    # plt.ylabel('Pixels')
    # plt.show()

    TS_method = 'NN'

    if TS_method == 'MLE':
        timeseries = MLE.EMI_py_stack(coherence)
        uncorrected_ts = MLE.EMI_py_stack(uncorrected)

        plt.imshow(np.angle(timeseries[2, :, :]))
        plt.show()
        plt.imshow(np.angle(uncorrected_ts[2, :, :]))
        plt.show()

        print(timeseries.shape)
        print(uncorrected_ts.shape)

        sm_ts = nearest_neighbor(coherence * uncorrected.conj())
    elif TS_method == 'EIG':
        timeseries = sarlab.eig_decomp(coherence)
        uncorrected_ts = sarlab.eig_decomp(uncorrected)
        print(timeseries.shape)
        sm_ts = nearest_neighbor(coherence * uncorrected.conj())
    elif TS_method == 'NN':
        timeseries = nearest_neighbor(coherence)
        uncorrected_ts = nearest_neighbor(uncorrected)
        sm_ts = nearest_neighbor(coherence * uncorrected.conj())

    normed_dif = np.linalg.norm(
        np.angle((timeseries * uncorrected_ts.conj())), 2, axis=2)
    plt.imshow(normed_dif)
    plt.show()

    cov = None

    # Check if output folder exists already
    if os.path.exists(os.path.join(os.getcwd(), outputs)):
        print('Output folder already exists, clearing it')
        shutil.rmtree(os.path.join(os.getcwd(), outputs))

    print('creating output folder')
    os.mkdir(os.path.join(os.getcwd(), outputs))

    for i in range(poly.shape[2]):
        path = os.path.join(os.getcwd(), outputs,
                            f'./degree_{i}.fit')
        io.write_image(path, poly[:, :, i].astype(
            np.float32), geocode=geom_path)

    r_path = os.path.join(os.getcwd(), outputs,
                          './correlation.fit')

    io.write_image(r_path, rs.astype(np.float32), geocode=geom_path)
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

        out_geom = os.path.join(os.getcwd(), outputs,
                                './geom_subset/')

        write_geometry(geom_path, out_geom, sample_size=sample_size)

        write_timeseries(corrected_timeseries, dates,
                         kappa_corrected, corrected_path, geocode=geom_path)
        write_timeseries(uncorrected_ts, dates,
                         kappa_uncorrected, uncorrected_path, geocode=geom_path)
        write_timeseries(sm_ts, dates,
                         kappa_uncorrected, sm_path, geocode=geom_path)


main()
