# Based on a workflow by Lohman & Burgi, 2023 (PREPRINT)
##
#
##
import colorcet as cc
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
import figStyle
import seaborn as sns
from matplotlib.cm import get_cmap
from regression.nl_phase import estimate_s

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


pc = get_cmap('cet_cyclic_ymcgy_60_90_c67_r')


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
    files = files[1:]

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

    clip = [0, 1000, 0, 2000]
    # clip = [0, -1, 0, -1]

    points = np.array([[0, 0], [1, 1]])

    SLCs = SLCs[:, clip[0]:clip[1], clip[2]:clip[3]]
    lf = inputs.lf
    N = SLCs.shape[0]

    if inputs.platform == 'S1':
        ml_size = (7*lf, 19*lf)
        sample_size = (2 * lf, 5*lf)

    elif inputs.platform == 'UAVSAR':
        ml_size = (5 * lf, 5 * lf)
        sample_size = (2 * lf, 2 * lf)

    sample_size = (2 * lf, 2 * lf)
    sample_i = (4, 4)
    sample_size = sample_i
    n = ml_size[0] * ml_size[1]  # Number of looks
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=sample_i)

    cov_full_res = CovarianceMatrix(SLCs, ml_size=(1, 1),
                                    sample=sample_i)

    SLCs = None

    coherence_ml = cov.cov
    coherence_full = cov_full_res.cov
    coherence_full_corrected = coherence_full.copy()

    intensity = cov.get_intensity()
    coherence = coherence_ml

    high_pass = coherence_full * coherence_ml.conj() / np.abs(coherence_ml)

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

        filter_strength = 2
        closure = sarlab.multilook(closure, ml=(
            filter_strength, filter_strength))

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, filter=1,  function=inputs.tripletform, kappa=1)

        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet

    Rs = np.zeros((high_pass.shape[2], high_pass.shape[3]))
    C1s = np.zeros((high_pass.shape[2], high_pass.shape[3]))
    C2s = np.zeros((high_pass.shape[2], high_pass.shape[3]))

    # TRAIN ON FIRST HALF OF STACK
    triui = np.triu_indices(19)
    triui = np.triu_indices(8)

    for i in range(high_pass.shape[2]):
        for j in range(high_pass.shape[3]):

            intensities = intensity[i, j, :]
            raw_intensities = intensities
            intensities = np.tile(intensities, (len(intensities), 1))
            intensities = (intensities.T - intensities)

            # r, p = stats.pearsonr(
            #     intensities[triui].flatten(), np.angle(high_pass[:, :, i, j][triui]).flatten())
            r = 1
            Rs[i, j] = r

            # Fit with only upper triangle of first half of stack
            # coeff1 = np.polyfit(
            #     intensities[triui], np.angle(high_pass[:, :, i, j][triui]), 1)

            s, iter = estimate_s(
                np.exp(1j * np.angle(high_pass[:, :, i, j])), intensities, gridN=20)

            # coeff2 = np.polyfit(
            #     amp_triplet_stack[:, i, j], np.angle(
            #         closure_stack[:, i, j]), 1)

            coeff2, iter = estimate_s(
                np.exp(1j * np.angle(closure_stack[:, i, j])), amp_triplet_stack[:, i, j], gridN=20)

            C1s[i, j] = s
            C2s[i, j] = coeff2

            # correction

            phi_corr = np.exp(1j * C1s[i, j] * intensities)
            # print(np.angle(phi_corr))

            if np.abs(r) > 0:
                coherence_full_corrected[:, :, i,
                                         j] = coherence_full[:, :, i, j] * phi_corr.conj()

            if False and (np.abs(r) > 0):

                theilslope = stats.theilslopes(np.angle(
                    high_pass[:, :, i, j][triui]), intensities[triui])

                fig, ax = plt.subplots(nrows=1, ncols=3)
                ax[0].scatter(intensities[triui],  np.angle(
                    high_pass[:, :, i, j][triui]), label='High Pass')
                ax[0].scatter(amp_triplet_stack[:, i, j],  np.angle(
                    closure_stack[:, i, j]), label='Triplets', alpha=0.5)

                x = np.linspace(np.min(intensities[triui]) - 0.1, np.max(
                    intensities[triui]) + 0.1)

                ax[0].plot(x, np.polyval(coeff1, x), label='fit phase')
                ax[0].plot(x, np.polyval(coeff2, x), label='fit triplet')
                ax[0].plot(x, x * theilslope[0], label='fit theilsen')

                ax[0].legend(loc='best')

                ax[1].scatter(intensities[triui],  np.exp(
                    1j * np.angle(high_pass[:, :, i, j][triui])).real)
                ax[1].set_title('Real')
                ax[2].scatter(intensities[triui], np.exp(
                    1j * np.angle(high_pass[:, :, i, j][triui])).imag)
                ax[2].set_title('Imaginary')

                plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=3)
    im = ax[0].imshow(Rs, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title('High Pass v. Intensity Change \nCorrelation')

    ax[1].set_title(r' High pass phase v. \Delta \sigma  Correlation')

    im = ax[1].imshow(C1s, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title('High-Phase Slopes')

    im = ax[2].imshow(C2s, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title('Triplet Slopes')
    plt.show()

    def get_coherence_improvement(i, j):

        ml = (10, 10)

        i1 = sarlab.multilook(coherence_full_corrected[i, j], ml=ml)
        i1 /= np.sqrt(np.abs(sarlab.multilook(coherence_full_corrected[i, i], ml=ml)) * np.abs(
            sarlab.multilook(coherence_full_corrected[j, j], ml=ml)))

        i2 = sarlab.multilook(coherence_full[i, j], ml=ml)
        i2 /= np.sqrt(np.abs(sarlab.multilook(coherence_full[i, i], ml=ml)) * np.abs(
            sarlab.multilook(coherence_full[j, j], ml=ml)))

        # Positive sign == improvement
        return (np.abs(i1) - np.abs(i2))

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True,
                           sharey=True, figsize=(10, 5))
    for axis in ax.flatten():
        axis.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left=False,
            right=False,         # ticks along the top edge are off
            labelbottom=False)
        axis.set_yticklabels([])
    # labels along the bottom edge are off
    # interferogram 1
    padding = 40

    i = 0
    j = 4
    # j = 1

    ax[0, 0].set_ylabel('Training\nInterferogram',
                        rotation=0, labelpad=padding)
    ax[0, 0].set_title('Unaltered Full-Res Interferogram')
    ax[0, 0].imshow(np.angle(coherence_full[i, j]),
                    vmin=-np.pi, vmax=np.pi, cmap=pc, interpolation=None)
    ax[0, 1].set_title('Corrected')
    ax[0, 1].imshow(np.angle(coherence_full_corrected[i, j]),
                    vmin=-np.pi, vmax=np.pi, cmap=pc, interpolation=None)

    improvement = get_coherence_improvement(i, j)
    ax[0, 2].set_title('Coherence Improvement')

    im = ax[0, 2].imshow(
        improvement, cmap=plt.cm.seismic, vmin=-0.2, vmax=0.2)

    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    i = 11
    j = 16

    i = 6
    j = 7

    ax[1, 0].imshow(np.angle(coherence_full[i, j]),
                    vmin=-np.pi, vmax=np.pi, cmap=pc, interpolation=None)
    ax[1, 0].set_ylabel('Test\nInterferogram', rotation=0, labelpad=padding)

    ax[1, 1].imshow(np.angle(coherence_full_corrected[i, j]),
                    vmin=-np.pi, vmax=np.pi, cmap=pc, interpolation=None)

    improvement = get_coherence_improvement(i, j)
    im = ax[1, 2].imshow(
        improvement, cmap=plt.cm.seismic, vmin=-0.2, vmax=0.2)

    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/high_pass_correction.png', dpi=300)
    plt.show()

    plt.hist(C1s.flatten(), bins=100)
    plt.show()

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
    r_path = os.path.join(os.getcwd(), outputs,
                          './correlation.fit')
    s_path = os.path.join(os.getcwd(), outputs,
                          './slope.fit')
    triplet_path = os.path.join(os.getcwd(), outputs,
                                './triplet_slope.fit')
    io.write_image(r_path, Rs.astype(np.float32), geocode=geom_path)
    io.write_image(s_path, C1s.astype(np.float32), geocode=geom_path)
    io.write_image(triplet_path, C2s.astype(np.float32), geocode=geom_path)

    corrected = CovarianceMatrix(None)
    corrected.set_cov(coherence_full_corrected)
    corrected.relook(ml_size, sample=sample_size)
    corrected = corrected.get_coherence()

    cov.relook((1, 1), sample=sample_size)
    original = cov.get_coherence()

    TS_method = inputs.phaselinking

    if TS_method == 'MLE':
        corrected_timeseries = MLE.EMI_py_stack(corrected)
        uncorrected_ts = MLE.EMI_py_stack(original)
    elif TS_method == 'EIG':
        corrected_timeseries = sarlab.eig_decomp(corrected)
        uncorrected_ts = sarlab.eig_decomp(original)

    if True:

        corrected_timeseries = corrected_timeseries.conj()
        uncorrected_ts = uncorrected_ts.conj()

        kappa_uncorrected = sarlab.compute_tc(original, uncorrected_ts)
        kappa_corrected = sarlab.compute_tc(corrected, corrected_timeseries)

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

    return

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

        points = np.array([[153, 81], [116, 54], [75, 171]])

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

        filter_strength = 2
        closure = sarlab.multilook(closure, ml=(
            filter_strength, filter_strength))

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, filter=1,  function=inputs.tripletform, kappa=1)

        if True:

            # out_triplets = os.path.join(os.getcwd(), outputs,
            #                             './triplets/')
            # if not os.path.exists(out_triplets):
            #     os.mkdir(out_triplets)

            # triplet_name = f'{triplet[0]}_{triplet[1]}_{triplet[2]}'

            # io.write_image(os.path.join(out_triplets, f'intensity_{triplet_name}'), amp_triplet.astype(
            #     np.float32), geocode=geom_path)

            # io.write_image(os.path.join(out_triplets, f'phase_{triplet_name}'), np.angle(closure).astype(
            #     np.float32), geocode=geom_path)
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(np.median(intensity[:, :, :], axis=2),
                         cmap=plt.cm.Greys)
            ax[1].imshow(np.angle(closure), cmap=plt.cm.seismic)
            ax[2].imshow(amp_triplet, cmap=plt.cm.seismic)

            plt.show()

        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet

    intensity_triplet_variance = np.var(amp_triplet_stack, 0)
    print(intensity_triplet_variance.shape)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(np.var(amp_triplet_stack, 0), vmax=0.3)
    ax[0].set_title('Intensity Variance')

    ax[1].imshow(np.var(np.angle(closure_stack), 0), vmin=0, vmax=np.pi)
    ax[1].set_title('Phase Variance')
    plt.show()

    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    rs = np.zeros(landcover.shape)

    ps = np.zeros(landcover.shape)

    degree = 1
    power = 1

    print('Estimating relationship')
    poly = np.zeros((landcover.shape[0], landcover.shape[1], degree + 1))


if __name__ == 'main':
    main()
