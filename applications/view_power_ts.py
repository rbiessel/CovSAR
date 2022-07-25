from cgitb import small
from distutils.log import error
from numpy.core.arrayprint import _leading_trailing
from numpy.lib.polynomial import polyval
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from scipy import special
from evd import write_timeseries
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
    # files = files[0:12]
    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    # clone = None
    if inputs.landcover:
        landcover = np.squeeze(rasterio.open(inputs.landcover).read())
        print('landcover:')
        print(landcover.shape)

    SLCs = io.load_stack_vrt(files)

    n = 19
    lf = 1
    ml_size = (7*lf, 19*lf)
    # ml_size = (1)
    n = ml_size[0] * ml_size[1]
    cov = CovarianceMatrix(SLCs, ml_size=ml_size, sample=(2 * lf, 5*lf))
    # cov = CovarianceMatrix(SLCs, ml_size=ml_size, sample=(2, 5))

    SLCs = None

    coherence = cov.get_coherence()
    uncorrected = coherence.copy()
    intensity = cov.get_intensity()

    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0], all=False)

    return

    triplets_permuted_1 = [[triplet[0], triplet[2], triplet[1]]
                           for triplet in triplets]

    # triplets = np.concatenate(
    #     (triplets, triplets_permuted_1))

    # k *= 2

    print('Triplets: ', triplets)

    A, rank = closures.build_A(triplets, coherence)

    # closure_cov = closures.get_triplet_covariance(
    #     cov.cov, A, n)[0]

    # print('Closure covariance shape: ', closure_cov.shape)
    U, S, Vh = np.linalg.svd(A)
    print(S)
    print(A)
    print(np.diag(1/S[:rank]))
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]

    if not inputs.landcover:
        landcover = np.zeros((coherence.shape[2], coherence.shape[3]))

    closure_stack = np.zeros((
        len(triplets), coherence.shape[2], coherence.shape[3]), dtype=np.complex64)

    landcover_types = np.unique(landcover)

    amp_triplet_stack = closure_stack.copy()
    amp_triplet_stack = amp_triplet_stack.astype(np.float64)

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        closure = sarlab.multilook(closure, ml=ml_size)
        # plt.imshow(np.angle(closure))
        # plt.show()

        # mean_coherence = (np.abs(coherence[triplet[0], triplet[1]]) + np.abs(
        #     coherence[triplet[1], triplet[2]]) + np.abs(coherence[triplet[0], triplet[2]].conj())) / 3

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]])
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
                intensities = intensities.T - intensities

                ws = 0
                window_closure = closure_stack[:, j-ws:j+ws+1, i-ws:i+ws+1]
                # mask = np.abs(np.angle(window_closure)) < np.pi/2

                # window_closure = window_closure  # [mask]
                window_amps = amp_triplet_stack[:,
                                                j-ws:j+ws+1, i-ws:i+ws+1]  # [mask]

                if len(window_amps.flatten()) > 2 and len(window_closure.flatten()) > 2:
                    r, p = stats.pearsonr(
                        np.angle(window_closure).flatten(), window_amps.flatten())
                    # other_C = closure_cov[j, i, :, :]
                    # # print(other_C.shape)
                    # diag = np.diag(other_C)
                    # W = (1/np.sqrt(diag))
                    # alphas = np.abs(np.log10(W))
                    # alphas = alphas/alphas.max()
                    # W = np.diag(W)

                    # other_C = np.diag(diag)
                    fitform = 'linear'

                    coeff, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    ), W=None, function=fitform)

                    # coeff_i, covm2 = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    # ), C=None, W=None, function='lineari')
                    # nt = window_amps.shape[0]

                    do_huber = False

                    poly[j, i, :] = coeff

                    rs[j, i] = r
                    ps[j, i] = p

                    # modeled systematic closures
                    if np.abs(r) >= 0:

                        est_closures = closures.eval_sytstematic_closure(
                            amp_triplet_stack[:, j, i], model=coeff, form=fitform)

                        # est_closures_int = closures.eval_sytstematic_closure(
                        #     amp_triplet_stack[:, j, i], model=coeff_i, form='lineari')

                        systematic_phi_errors = closures.least_norm(
                            A, est_closures, pinv=False, pseudo_inv=A_dagger)

                        # systematic_phi_errors_int = closures.least_norm(
                        #     A, est_closures_int, pinv=False, pseudo_inv=A_dagger)

                        # uncorrected_phi_errors = closures.least_norm(
                        #     A, closure_stack[:, j, i], pinv=False, pseudo_inv=A_dagger)

                        error_coh = closures.phivec_to_coherence(
                            systematic_phi_errors, coherence[:, :, j, i].shape[0])

                        error_coh_nl = error_coh
                        # error_coh_unc = closures.phivec_to_coherence(
                        #     uncorrected_phi_errors, coherence[:, :, j, i].shape[0])

                        gradient = interpolate_phase_intensity(
                            raw_intensities, error_coh)

                        linear_phase = np.exp(
                            1j * (-2 * intensities * gradient))

                        error_coh = error_coh * linear_phase

                        coherence[:, :, j, i] = coherence[:,
                                                          :, j, i] * error_coh.conj()

                        # coherence[:, :, j, i] = error_coh

                    if p < 0.05 and np.abs(r) > 0.9:
                        gradient = interpolate_phase_intensity(
                            raw_intensities, error_coh_nl, plot=True)

                        fig, ax = plt.subplots(nrows=2, ncols=1)
                        # slope_stderr = np.sqrt(covm[0][0])
                        # intercept_stderr = np.sqrt(covm[1][1])

                        # print('Alphas: ', alphas)
                        ax[0].scatter(window_amps.flatten(), np.angle(
                            window_closure).flatten(), s=10)  # alpha=(alphas)**(1))

                        x = np.linspace(
                            window_amps.min() - 0.1 * np.abs(window_amps.min()), window_amps.max() + 0.1 * np.abs(window_amps.max()), 100)

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff, form=fitform), '--', label='Closure Fit')

                        # ax[0].plot(x, closures.eval_sytstematic_closure(
                        #     x, coeff_i, form='lineari'), '--', label='Closure Fit w/ intercept')

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

                        ax[0].set_title(
                            f'Type: {l}, corr: {r}')
                        ax[0].set_ylabel('Closures (rad)')
                        ax[0].set_xlabel('Intensity Ratio Triple Product')
                        ax[0].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        ax[1].set_xlabel('Intensity Ratio')
                        ax[1].set_ylabel('Estimated Phase Error (rad)')

                        iratios = closures.coherence_to_phivec(intensities)

                        # nlphases_int = np.angle(
                        #     systematic_phi_errors_int).flatten()

                        # nl_phases_uncorrected = np.angle(
                        #     uncorrected_phi_errors)
                        # # m = sarlab.fit_phase_ratio(
                        #     iratios, nlphases, degree=4)

                        # polycoeff = np.polyfit(
                        #     iratios, nlphases_int, 3)

                        # polydir = np.polyder(polycoeff)
                        # slope_to_add = polydir[-1]

                        nlphases = np.angle(
                            closures.coherence_to_phivec(error_coh))

                        nlphases_raw = np.angle(
                            closures.coherence_to_phivec(error_coh_nl))

                        # print(m)
                        x = np.linspace(iratios.min(
                        ) - 0.1 * np.abs(iratios.min()), iratios.max() + 0.1 * np.abs(iratios.max()), 100)

                        max_range = np.max(np.abs(iratios))  # + 0.1
                        x = np.linspace(-max_range, max_range, 100)

                        ax[1].scatter(iratios, nlphases, s=15,
                                      marker='x', color='blue', label='/w a linear component to force monotonicity')

                        ax[1].scatter(iratios, nlphases_raw, s=15,
                                      marker='x', color='orange', label='raw inverted phase')

                        # ax[1].scatter(iratios, nlphases_int, s=15,
                        #               marker='x', color='orange', label='/w Intercept')

                        # ax[1].scatter(iratios, nl_phases_uncorrected, s=15,
                        #               marker='x', label='From uncorrected Closures')
                        # ax[1].plot(x, np.polyval(m, x), '--',
                        #            label='nonlinear poly fit')
                        # ax[1].plot(x, np.polyval(polycoeff, x),
                        #            '--', label='Full poly fit')

                        # ax[1].plot(x, np.polyval(polycoeff, x) - (2 * slope_to_add * x),
                        #            '--', label='Monotonic poly fit')

                        ax[1].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        plt.tight_layout()
                        plt.show()

                        plt.hist(np.angle(window_closure).flatten(), bins=50)
                        plt.show()

                        fig, ax = plt.subplots(nrows=1, ncols=2)
                        ax[0].set_title(
                            'Estimated Phase Error (w/o intercept)')

                        ax[0].imshow(np.angle(error_coh),
                                     vmin=-np.pi/2, vmax=np.pi/2, cmap=plt.cm.seismic)

                        ax[1].set_title(
                            'Estimated Phase Error (w/ intercept)')
                        # im = ax[1].imshow(np.angle(error_coh_unc),
                        #                   vmin=-np.pi/2, vmax=np.pi/2, cmap=plt.cm.seismic)
                        # fig.colorbar(im, label='Estimated Phase Error (rad)')

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

    plt.hist(ps.flatten(), bins=100)
    plt.show()

    print('Poly shape:')
    fig, ax = plt.subplots(ncols=(4), nrows=1,
                           sharex=True, sharey=True)
    ax[1].imshow(rs, cmap=plt.cm.seismic, vmin=-1, vmax=1)
    # ax[0].imshow(10 * np.log10(intensity[:, :, 0]))
    ax[0].imshow(ps)

    ax[2].imshow(poly[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
    ax[2].set_title(f'Degree 0')
    ax[3].imshow(poly[:, :, 1], cmap=plt.cm.seismic, vmin=-1, vmax=1)
    ax[3].set_title(f'Degree 1')
    plt.show()

    # for i in range(coherence.shape[0]):
    #     for j in range(coherence.shape[1]):
    #         plt.imshow(np.angle(coherence[j, i, :, :]))
    #         plt.show()

    timeseries = sarlab.eig_decomp(coherence)
    uncorrected_ts = sarlab.eig_decomp(uncorrected)
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

    slope_path = os.path.join(os.getcwd(), outputs,
                              './slope.fit')

    intercept_path = os.path.join(os.getcwd(), outputs,
                                  './intercept.fit')

    r_path = os.path.join(os.getcwd(), outputs,
                          './correlation.fit')

    slopes = poly[:, :, 1]
    io.write_image(slope_path, slopes.astype(np.float32))
    io.write_image(intercept_path, poly[:, :, 0].astype(np.float32))
    io.write_image(r_path, rs.astype(np.float32))

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
                         kappa_corrected, corrected_path)
        write_timeseries(uncorrected_ts, dates,
                         kappa_uncorrected, uncorrected_path)

        write_timeseries(sm_ts, dates,
                         kappa_uncorrected, sm_path)


main()
