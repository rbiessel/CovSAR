from closig import expansion
from closig.plotting import triangle_plot
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
from matplotlib.colors import Normalize
from covariance import CovarianceMatrix
import isceio as io
import closures
from matplotlib import cm

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
from regression.nl_phase import estimate_s
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
    files = files[:30]

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
    NDVIs = io.load_stack_ndvi(files)

    print(NDVIs.shape)
    print(SLCs.shape)

    # return

    clip = None

    clip = [0, -1, 0, -1]

    points = np.array([[0, 0], [1, 1]])

    SLCs = SLCs[:, clip[0]:clip[1], clip[2]:clip[3]]
    NDVIs = NDVIs[:, clip[0]:clip[1], clip[2]:clip[3]]
    lf = inputs.lf
    N = SLCs.shape[0]

    if inputs.platform == 'S1':
        ml_size = (7*lf, 19*lf)
        sample_size = (2 * lf, 5*lf)
        # sample_size = (1, 1)

    elif inputs.platform == 'UAVSAR':
        ml_size = (5 * lf, 5 * lf)
        sample_size = (2 * lf, 2 * lf)

    n = ml_size[0] * ml_size[1]  # Number of looks
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=sample_size)

    print('COV SHAPE:', cov.cov.shape)
    print('SLC SHAPE:', SLCs.shape)
    SLCs = None

    coherence = cov.get_coherence()
    uncorrected = coherence.copy()
    intensity = cov.get_intensity()

    # points = np.indices(intensity[:, :, 0].shape)

    # points = points[:, ::100, ::100]
    # points = points.reshape((2, (points.shape[1] * points.shape[2]))).T
    # points = points[1:]
    # print(points)

    # lats = io.load_geom_from_slc(files[0], file='lat')[
    #     ::sample_size[0], ::sample_size[1]]
    # lons = io.load_geom_from_slc(files[0], file='lon')[
    #     ::sample_size[0], ::sample_size[1]]

    NDVIs[np.isnan(NDVIs)] = 0

    print(NDVIs.shape)
    # NDVIs = 10 * np.log10(NDVIs)

    for i in range(intensity.shape[2]):
        intensity[:, :, i] = sarlab.multilook(intensity[:, :, i], ml=(1, 1))
        NDVIs[i, :, :] = sarlab.multilook(NDVIs[i, :, :], ml=ml_size)

    NDVIs = NDVIs[:, ::sample_size[0], ::sample_size[1]]

    # plt.imshow(np.median(intensity[:, :, :], axis=2), cmap=plt.cm.Greys)
    # plt.scatter(points[:, 0], points[:, 1], s=30,
    #             facecolors='none', edgecolors='r')
    # plt.show()
    # m = -2
    # coherence_m = sarlab.reduce_cov(coherence, keep_diag=m)
    # print(np.abs(coherence_m[:, :, 10, 10]))
    # return
    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0], all=False)
    A, rank = closures.build_A(
        triplets, coherence)

    U, S, Vh = np.linalg.svd(A)
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]

    AdagA = A_dagger @ A

    if not inputs.landcover:
        landcover = np.zeros((coherence.shape[2], coherence.shape[3]))

    closure_stack = np.zeros((
        len(triplets), coherence.shape[2], coherence.shape[3]), dtype=np.complex64)

    amp_triplet_stack = closure_stack.copy()
    amp_triplet_stack = amp_triplet_stack.astype(np.float64)
    ndvi_triplet_stack = amp_triplet_stack.copy()

    print(amp_triplet_stack.shape)

    # for i in range(NDVIs.shape[0]):
    #     plt.imshow(NDVIs[i], vmin=0, vmax=0.4)
    #     plt.show()

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        filter_strength = 1
        closure = sarlab.multilook(closure, ml=(
            filter_strength, filter_strength))

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, filter=1,  function=inputs.tripletform, kappa=1)

        ndvi_triplet = sarlab.intensity_closure(
            NDVIs[triplet[0], :, :], NDVIs[triplet[1], :, :], NDVIs[triplet[2], :, :], norm=False, cubic=False, filter=1,  function='arctan', kappa=15, L=10e3)

        if False:

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
        ndvi_triplet_stack[i] = ndvi_triplet

    intensity_triplet_variance = np.var(amp_triplet_stack, 0)
    print(intensity_triplet_variance.shape)
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].imshow(np.var(amp_triplet_stack, 0), vmax=0.3)
    # ax[0].set_title('Intensity Variance')

    # ax[1].imshow(np.var(np.angle(closure_stack), 0), vmin=0, vmax=np.pi)
    # ax[1].set_title('Phase Variance')
    # plt.show()

    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    rs = np.zeros(landcover.shape)
    rsme_linear = np.zeros(landcover.shape)
    # rsme_nonlinear = np.zeros(landcover.shape)
    # coeff_lin = np.zeros(landcover.shape)

    p = 5

    def ml_eval(eval):
        # return np.mean(eval, axis=(1, 2))
        return np.mean(eval.real, axis=(1, 2)) + 1j * np.mean(eval.imag, axis=(1, 2))

    cmap, vabs = plt.cm.seismic, 45
    # p = 40

    def update_plot(covf, x, y, vits2):
        closure_slice = closure_stack[:, x, y]
        ndvi_triplet_slice = ndvi_triplet_stack[:, x, y]
        itriplet_slice = amp_triplet_stack[:, x, y]

        scatter_inten = ax['intensity'].scatter(
            itriplet_slice.flatten(), np.angle(closure_slice).flatten(), s=10, alpha=0.5, color='black')

        coef = np.polyfit(itriplet_slice.flatten(),
                          np.angle(closure_slice).flatten(), 1)

        syst_i_cphase = np.exp(1j * coef[0] * itriplet_slice.flatten())

        scatter_ndvi = ax['ndvi'].scatter(
            ndvi_triplet_slice.flatten(), np.angle(closure_slice.flatten() * syst_i_cphase.conj().flatten()), s=10, alpha=0.5, color='black')

        ax['intensity'].set_title('Intensity Triplet')
        ax['ndvi'].set_title('NDVI Triplet')
        ax['ndvi'].set_xlabel('NDVI Triplet')
        ax['ndvi'].set_ylabel('Closure Phase')

        ax['intensity'].set_ylabel('Closure Phase')
        ax['intensity'].set_xlabel('Intensity Triplet')

        coh_slice = covf[:, :, x-p:x+p, y-p:y+p]
        inds = np.tril_indices(coh_slice.shape[0])

        Cvec = coh_slice[inds[0], inds[1], :, :]
        Cvec = np.moveaxis(Cvec, 0, -1)

        vectorized = True

        basis = expansion.TwoHopBasis(covf.shape[0])
        ax['triangle'].set_title('Two Hop Basis')

        # eval = basis.evaluate_covariance(
        #     Cvec, compl=True, normalize=False, vectorized=vectorized)
        eval = basis.evaluate_covariance(
            covf[:, :, x, y], compl=True, normalize=False, vectorized=False)
        # print(eval.dtype)

        triangle_plot(basis, eval,
                      ax=ax['triangle'], cmap=cmap, vabs=vabs)

        basis = expansion.SmallStepBasis(covf.shape[0])
        # eval = basis.evaluate_covariance(
        #     Cvec, compl=True, normalize=False, vectorized=vectorized)
        eval = basis.evaluate_covariance(
            covf[:, :, x, y], compl=True, normalize=False, vectorized=False)
        # print(eval.dtype)

        triangle_plot(basis, eval,
                      ax=ax['triangle2'], cmap=cmap, vabs=vabs)
        ax['triangle2'].set_title('Small Step Basis')
        ax['triangle2'].set_xlabel('Scene')
        ax['triangle'].set_ylabel('Timescale')

        l1, = ax['vits'].plot(NDVIs[:, x, y], '-o',
                              label='NDVI', color='seagreen', alpha=0.8, markersize=4)
        ax['vits'].set_xlabel('Scene')
        ax['vits'].set_ylabel('NDVI')
        vits2.set_ylabel('Intensity [dB]')

        iseries = intensity[x, y, :] - intensity[x, y, 0]

        # iseries = iseries/np.max(np.abs(iseries))
        l2, = vits2.plot(iseries, '-o', label='intensity',
                         color='tomato', alpha=0.8, markersize=4)

        ll = [l1, l2]
        vits2.legend(ll, [ll_.get_label() for ll_ in ll],
                     loc='upper right', fontsize=10, ncol=2)

    fig, ax = plt.subplot_mosaic([['left', 'left', 'intensity', 'triangle'],
                                  ['left', 'left', 'intensity', 'triangle'],
                                  ['left', 'left', 'intensity', 'triangle2'],
                                  ['left', 'left', 'ndvi', 'triangle2'],
                                  ['left', 'left', 'ndvi', 'vits'],
                                  ['left', 'left', 'ndvi', 'vits']],

                                 figsize=(16, 9))

    x, y = 5, 5
    vits2 = ax['vits'].twinx()

    update_plot(coherence, x, y, vits2)

    ax['intensity'].set_title('Intensity Triplet')
    ax['ndvi'].set_title('NDVI Triplet')
    ax['left'].set_title('Median NDVI \& POI')
    ax['triangle'].set_xlabel('Scene')
    ax['triangle'].set_ylabel('Timescale')
    ndviim = ax['left'].imshow(np.median(NDVIs, axis=0), vmin=0, vmax=0.3)

    divider = make_axes_locatable(ax['left'])
    caxls = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(ndviim, cax=caxls, orientation="vertical")
    cbar.ax.set_title('[NDVI]', pad=10)

    scatter = ax['left'].scatter(y, x, s=30,
                                 facecolors='none', edgecolors='r')

    # reference = ax['left'].scatter(POIs['reference'][1], POIs['reference'][0], s=50,
    #                                facecolors='none', edgecolors='black', marker='*')
    # ax['left'].set_title('Median Intensity \& POI')

    divider = make_axes_locatable(ax['triangle'])
    caxls = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(cm.ScalarMappable(
        norm=Normalize(-vabs, vabs, clip=True), cmap=cmap), cax=caxls, orientation="vertical")
    cbar.set_ticks([-vabs, -vabs/2, 0, vabs/2, vabs])
    cbar.ax.set_title('[$^{\circ}$]')
    divider = make_axes_locatable(ax['triangle2'])
    caxls = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(cm.ScalarMappable(
        norm=Normalize(-vabs, vabs, clip=True), cmap=cmap), cax=caxls, orientation="vertical")
    cbar.set_ticks([-vabs, -vabs/2, 0, vabs/2, vabs])
    cbar.ax.set_title('[$^{\circ}$]')

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            # ax['left'].clear()
            ax['intensity'].clear()
            ax['ndvi'].clear()
            ax['triangle'].clear()
            ax['triangle2'].clear()

            ax['vits'].clear()
            vits2.clear()

            x, y = int(np.floor(event.xdata)), int(
                np.floor(event.ydata))
            update_plot(coherence, y, x, vits2)
            scatter.set_offsets((event.xdata, event.ydata))
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


main()
