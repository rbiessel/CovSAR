from cgitb import small
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
    files = files[:]
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

    # SLCs = SLCs[:, 1000:2000, 1000:3000]
    n = 19
    # 7 x 19
    n = np.sqrt(7 * 19)
    # cov = CovarianceMatrix(SLCs, ml_size=(7, 19))
    cov = CovarianceMatrix(SLCs, ml_size=(7, 19), sample=(2, 5))

    SLCs = None
    k = special.comb(cov.cov.shape[0] - 1, 2)

    triplets = closures.get_triplets(cov.cov.shape[0])
    triplets = triplets[0:int(k)]

    coherence = cov.get_coherence()
    uncorrected = coherence.copy()
    intensity = cov.get_intensity()

    # plt.show()

    # triplets = closures.get_adjacent_triplets(int(k))
    # Get A
    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0])
    variances = np.var(triplets, axis=1)
    print('variances: ', variances)
    triplets = triplets[np.argsort(np.var(triplets, axis=1))]
    k = 15
    triplets = triplets[0:int(k)]
    A, rank = closures.build_A(triplets, coherence)

    # closure_cov = closures.get_triplet_covariance(cov.cov, A, n)[0]

    # print('Closure covariance shape: ', closure_cov.shape)
    U, S, Vh = np.linalg.svd(A)
    print(S)
    print(A)
    print(np.diag(1/S[:rank]))
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]
    # print('Singular values of A')
    # print(S)

    # print(A)
    # print('Rank: ', np.linalg.matrix_rank(A))

    # triplets = np.delete(triplets, 1, axis=0)
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

    degree = 2
    power = 1

    print('Estimating relationship')
    poly = np.zeros((landcover.shape[0], landcover.shape[1], degree + 1))

    for j in range(amp_triplet_stack.shape[1]):
        for i in range(amp_triplet_stack.shape[2]):
            l = landcover[j, i]
            if l == 2 or l != 2:

                # try:
                slice_closure = np.angle(
                    closure_stack[:, j, i])
                slice_amps = amp_triplet_stack[:, j, i]

                phases = coherence[:, :,  j, i]
                intensities = intensity[j, i, :]
                intensities = np.tile(intensities, (len(intensities), 1))
                intensities = intensities.T - intensities

                ws = 0
                window_closure = closure_stack[:, j-ws:j+ws+1, i-ws:i+ws+1]
                mask = np.abs(np.angle(window_closure)) < np.pi/2

                window_closure = window_closure  # [mask]
                window_amps = amp_triplet_stack[:,
                                                j-ws:j+ws+1, i-ws:i+ws+1]  # [mask]

                if len(window_amps.flatten()) > 2 and len(window_closure.flatten()) > 2:
                    # r, p = stats.spearmanr(
                    #     np.angle(window_closure).flatten(), window_amps.flatten())

                    # coeff, res, a, b, c = np.polyfit(np.angle(window_closure).flatten(
                    # ), window_amps.flatten(), deg=degree, full=True, w=np.abs(window_closure).flatten(
                    # )**(1/3))

                    # other_C = closure_cov[j, i, :, :]
                    # print(other_C.shape)
                    other_C = np.diag(
                        np.ones(window_closure.flatten().shape[0]))
                    # coeff = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    # ), C=np.diag(np.diag(closure_cov[j, i])))
                    # print(closure_cov[j, i].shape)
                    fitform = 'root3'
                    coeff = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
                    ), C=other_C, function=fitform)

                    do_huber = False
                    if do_huber:
                        huber = HuberRegressor(alpha=0.0, epsilon=1)
                        huber.fit(
                            np.array([window_amps.flatten()]).T,  np.angle(window_closure).flatten(), np.diag(closure_cov[j, i]))

                        coef_huber = np.array(
                            [huber.coef_, huber.intercept_])
                        poly[j, i, :] = coeff

                    else:
                        poly[j, i, :] = coeff

                    rs[j, i] = 0  # r
                    ps[j, i] = 0  # p
                    r = 0

                    # modeled systematic closures
                    if np.abs(r) >= 0:

                        est_closures = coeff[0] * np.sign(amp_triplet_stack[:, j, i]) * np.abs(
                            amp_triplet_stack[:, j, i].flatten())**(1/3) + coeff[1] * amp_triplet_stack[:, j, i].flatten() + coeff[2]

                        systematic_phi_errors = closures.least_norm(
                            A, est_closures, pinv=False, pseudo_inv=A_dagger)

                        error_coh = closures.phivec_to_coherence(
                            systematic_phi_errors, coherence[:, :, j, i].shape[0])

                        coherence[:, :, j, i] = coherence[:,
                                                          :, j, i] * error_coh.conj()

                        # coherence[:, :, j, i] = error_coh

                    if np.abs(r) > 1:
                        plt.plot(slice_closure, label='Closures')
                        plt.plot(
                            slice_amps, label='Intensity Non-linearity')
                        plt.title(
                            f'Power: {power} Type: {l}')
                        plt.xlabel('Triplet')
                        plt.ylabel('Closure/Intensity Closure')
                        plt.legend(loc='lower left')
                        plt.show()

                        for k in range(window_closure.shape[0]):
                            plt.scatter(window_amps[k].flatten(
                            ), np.angle(window_closure)[k].flatten())

                        x = np.linspace(
                            window_amps.min() - 0.1 * np.abs(window_amps.min()), window_amps.max() + 0.1 * np.abs(window_amps.max()), 100)
                        if fitform is 'linear':
                            plt.plot(x, polyval(coeff, x))
                        elif fitform is 'root3':
                            plt.plot(x, coeff[0] * np.sign(x) * np.abs(x)**(1/3) +
                                     coeff[1] * x + coeff[2])

                        if do_huber:
                            coef_ = huber.coef_ * x + huber.intercept_
                            plt.plot(x, coef_, label="huber loss, 1")

                        plt.title(
                            f'Type: {l}, corr: {rs[j, i]}')
                        plt.ylabel('Closures (rad)')
                        plt.xlabel('Intensity Non-linearity')
                        plt.legend(loc='lower right')
                        plt.show()

                        plt.xlabel('Intensity Ratios')
                        plt.ylabel('Estimated Phase Error')
                        print('number of phases: ', len(systematic_phi_errors))
                        plt.scatter((closures.coherence_to_phivec(intensities)), np.angle(
                            systematic_phi_errors).flatten())
                        plt.show()

                    else:
                        r = 0
                    # except:
                    #     # print('robust regression failed :(')
                    #     poly[j, i, :] = np.zeros((2))
                    #     rs[j, i] = 0

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

    normed_dif = np.linalg.norm((timeseries - uncorrected_ts), 2, axis=2)
    plt.imshow(normed_dif)
    plt.show()

    plt.scatter(normed_dif.flatten(), poly[:, :, 1].flatten())
    plt.xlabel('normed difference between corrected and uncorrected timeseries')
    plt.ylabel('Phase Closure backscatter scaling')
    plt.show()
    # for i in np.random.randint(0, 140, size=140):
    #     for j in np.random.randint(0, 140, size=140):
    #         if normed_dif[j, i] > 1.5:
    #             plt.plot(np.angle(timeseries[i, j]), label='corrected')
    #             plt.plot(np.angle(uncorrected_ts[i, j]),
    #                      '--', label='uncorrected_ts')
    #             plt.legend(loc='upper left')
    #             plt.show()

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

        write_timeseries(corrected_timeseries, dates,
                         kappa_corrected, corrected_path)
        write_timeseries(uncorrected_ts, dates,
                         kappa_uncorrected, uncorrected_path)


main()
