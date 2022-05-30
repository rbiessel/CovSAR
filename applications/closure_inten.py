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
from scipy.interpolate import griddata


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
    # files = files[1:10]
    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    # clone = None
    if inputs.landcover:
        landcover = np.squeeze(rasterio.open(inputs.landcover).read())
        print('landcover:')
        print(landcover.shape)

    # SLCs = io.load_stack(files)
    SLCs = io.load_stack_vrt(files)
    # inc_map = io.load_inc_from_slc(files[0])

    SLCs = SLCs[:, 1100:2100, 1100:2100]
    n = 19
    # 7 x 19
    # cov = CovarianceMatrix(SLCs, ml_size=(7, 19))
    # cov = CovarianceMatrix(SLCs, ml_size=(7, 19), sample=(2, 5))
    lf = 2
    ml_size = (7*lf, 19*lf)
    n = ml_size[0] * ml_size[1]
    cov = CovarianceMatrix(SLCs, ml_size=ml_size, sample=(2*lf, 5*lf))
    # cov = CovarianceMatrix(SLCs, ml_size=ml_size, sample=(2, 5))

    SLCs = None

    coherence = cov.get_coherence()
    uncorrected = coherence.copy()
    intensity = cov.get_intensity()

    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0], all=False)
    # variances = np.var(triplets, axis=1)
    # print('variances: ', variances)
    # triplets = triplets[np.argsort(np.var(triplets, axis=1))]
    # triplets = triplets[0:int(k)]

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
        print(triplet)
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

                # # try:
                # slice_closure = np.angle(
                #     closure_stack[:, j, i])
                # slice_amps = amp_triplet_stack[:, j, i]

                # phases = coherence[:, :,  j, i]
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

                    coeff_i, covm2 = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    ), C=None, W=None, function='lineari')
                    nt = window_amps.shape[0]

                    do_huber = False

                    print('Model Covariance:')

                    print(covm)
                    print('coeff unweighted:')
                    print(coeff_i)
                    poly[j, i, :] = coeff

                    rs[j, i] = r
                    ps[j, i] = p

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

                        error_coh_unc = closures.phivec_to_coherence(
                            uncorrected_phi_errors, coherence[:, :, j, i].shape[0])

                        coherence[:, :, j, i] = coherence[:,
                                                          :, j, i] * error_coh.conj()

                        # coherence[:, :, j, i] = error_coh

                    if p < 0.05 and np.abs(r) > 0.2:
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

                        ax[0].plot(x, closures.eval_sytstematic_closure(
                            x, coeff_i, form='lineari'), '--', label='Closure Fit w/ intercept')

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
                        nlphases = np.angle(
                            systematic_phi_errors).flatten()

                        nlphases_int = np.angle(
                            systematic_phi_errors_int).flatten()

                        nl_phases_uncorrected = np.angle(
                            uncorrected_phi_errors)
                        m = sarlab.fit_phase_ratio(
                            iratios, nlphases, degree=4)

                        polycoeff = np.polyfit(
                            iratios, nlphases_int, 3)

                        polydir = np.polyder(polycoeff)
                        slope_to_add = polydir[-1]

                        # print(m)
                        x = np.linspace(iratios.min(
                        ) - 0.1 * np.abs(iratios.min()), iratios.max() + 0.1 * np.abs(iratios.max()), 100)

                        max_range = np.max(np.abs(iratios))  # + 0.1
                        x = np.linspace(-max_range, max_range, 100)

                        ax[1].scatter(iratios, nlphases, s=15,
                                      marker='x', color='blue', label='/wo Intercept')

                        ax[1].scatter(iratios, nlphases_int, s=15,
                                      marker='x', color='orange', label='/w Intercept')

                        # ax[1].scatter(iratios, nl_phases_uncorrected, s=15,
                        #               marker='x', label='From uncorrected Closures')

                        print('model: ', m)
                        # ax[1].plot(x, np.polyval(m, x), '--',
                        #            label='nonlinear poly fit')
                        # ax[1].plot(x, np.polyval(polycoeff, x),
                        #            '--', label='Full poly fit')

                        # ax[1].plot(x, np.polyval(polycoeff, x) - (2 * slope_to_add * x),
                        #            '--', label='Monotonic poly fit')

                        ax[1].legend(bbox_to_anchor=(1.05, 0.5),
                                     loc='center left', borderaxespad=0.)

                        # m[-2] = 1/2
                        # ax[3].plot(x, np.polyval(m, x), '--')

                        plt.tight_layout()
                        plt.show()

                        imax = np.max(np.abs(raw_intensities))
                        imin = np.min(np.abs(raw_intensities))

                        ngrid = 100j
                        grid_x, grid_y = np.mgrid[imin:imax:ngrid,
                                                  imin:imax:ngrid]
                        points = []
                        values = []
                        for i in range(error_coh.shape[0]):
                            for j in range(error_coh.shape[1]):
                                points.append(
                                    (raw_intensities[i], raw_intensities[j]))
                                values.append(np.angle(error_coh_unc[i, j]))

                        points = np.array(points)
                        values = np.array(values)
                        # print(points.shape)
                        # print(values.shape)
                        # print(grid_x.shape)
                        grid_z2 = griddata(
                            points, values, (grid_x, grid_y),  method='linear')

                        fig, ax = plt.subplots(nrows=1, ncols=2)
                        im = ax[0].imshow(grid_z2.T, extent=(
                            imin, imax, imin, imax), origin='lower', cmap=plt.cm.hsv)

                        ax[0].scatter(points[:, 0], points[:, 1],
                                      marker='o', s=20, facecolors='none', edgecolors='black')

                        ax[0].contour(grid_z2.T, colors='k', origin='lower', extent=(
                            imin, imax, imin, imax))
                        ax[0].set_xlabel('Backscatter Intensity (Reference)')
                        ax[0].set_ylabel('Backscatter Intensity (Secondary)')
                        plt.colorbar(
                            im, label='Nonlinear Phase Error (rad)')

                        grid_z2 = np.flip(grid_z2, axis=0)

                        dif_interpolated = np.tile(
                            grid_x[:, 0], (len(grid_x[:, 0]), 1))
                        dif_interpolated = dif_interpolated.T - dif_interpolated
                        dif_interpolated = np.flip(dif_interpolated, axis=0)

                        #  print(grid_z2)
                        print(dif_interpolated)

                        gradient = (grid_z2[-10, -10] - grid_z2[-1, -1]) / (
                            (dif_interpolated[-10, -10] - dif_interpolated[-1, -1]))
                        # print(grid_z2[-10, -10])
                        # print(grid_z2[-1, -1])
                        # print(grid_x[-1, -1])
                        # print(grid_y[-1, -1])

                        lin_phase = dif_interpolated * gradient
                        lin_phase = np.flip(lin_phase, axis=0)
                        grid_z2 = np.flip(grid_z2, axis=0)

                        with_linear_phase = grid_z2 + (2 * lin_phase.T)
                        ax[1].imshow(with_linear_phase, extent=(
                            imin, imax, imin, imax), origin='lower', cmap=plt.cm.hsv)
                        ax[1].contour(with_linear_phase, colors='k', origin='lower', extent=(
                            imin, imax, imin, imax))
                        ax[1].set_xlabel('Backscatter Intensity (Reference)')
                        ax[1].set_ylabel('Backscatter Intensity (Secondary)')
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
                        im = ax[1].imshow(np.angle(error_coh_unc),
                                          vmin=-np.pi/2, vmax=np.pi/2, cmap=plt.cm.seismic)
                        fig.colorbar(im, label='Estimated Phase Error (rad)')

                        plt.show()

                    else:
                        r = 0
                    # except:
                    #     # print('robust regression failed :(')
                    #     poly[j, i, :] = np.zeros((2))
                    #     rs[j, i] = 0
            if (j % 100) == 0 and i == 0:
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
