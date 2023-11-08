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
from closig import expansion
from closig.plotting import triangle_plot
import colorcet as cc
from regression.nl_phase import estimate_s
from matplotlib import cm
from matplotlib.colors import Normalize
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


def high_pass_correction(high_pass, intensities, gridN=20):
    s, iter = estimate_s(
        np.exp(1j * np.angle(high_pass)), intensities, gridN=gridN)

    phi_corr = np.exp(1j * s * intensities)
    return phi_corr


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
    files = files[:]

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
    # clip = [0, 500, 0, 1000]

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

    n = ml_size[0] * ml_size[1]  # Number of looks
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=(1, 1))
    full_res = CovarianceMatrix(SLCs, ml_size=(1, 1),
                                sample=(1, 1))

    SLCs = None

    # coherence = cov.get_coherence()
    # uncorrected = coherence.copy()
    intensity = cov.get_intensity()
    cov = cov.cov
    full_res = full_res.cov
    coherence_full_corrected = full_res.copy()

    print('Computing high pass Correction')

    # Correct full res interferograms

    for i in range(high_pass.shape[2]):
        for j in range(high_pass.shape[3]):
            intensities = intensity[i, j, :]
            raw_intensities = intensities
            intensities = np.tile(intensities, (len(intensities), 1))
            intensities = (intensities.T - intensities)
            phi_corr = high_pass_correction(
                high_pass[:, :, i, j], intensities, gridN=20)
            coherence_full_corrected[:, :, i,
                                     j] = full_res[:, :, i, j] * phi_corr.conj()

            if (i % 10) == 0 and j == 0:
                print(
                    f'High Pass Correction Progress: {(i/high_pass.shape[2]* 100)}%', end='\r')
    print('Finished!')

    corrected = CovarianceMatrix(None)
    corrected.set_cov(coherence_full_corrected)
    corrected.relook(ml_size, sample=sample_size)
    corrected = corrected.cov

    original = CovarianceMatrix(None)
    original.set_cov(cov)
    original.relook(ml_size, sample=sample_size)
    intensity = original.get_intensity()

    cov = original.cov

    basis = expansion.TwoHopBasis(cov.shape[0])
    cmap, vabs = cc.cm['CET_C3'], 180
    cmap, vabs = cc.cm['CET_C3_r'], 180
    cmap, vabs = cc.cm['CET_D1A'], 180

    def update_triangles(coherence, corrected, x, y):
        coh_slice = coherence[:, :, x, y]
        basis = expansion.TwoHopBasis(coherence.shape[0])
        eval = basis.evaluate_covariance(
            coh_slice, compl=True)
        triangle_plot(basis, eval, ax=ax['um'], cmap=cmap)
        basis = expansion.SmallStepBasis(coherence.shape[0])
        eval = basis.evaluate_covariance(
            coh_slice, compl=True)
        triangle_plot(
            basis, eval, ax=ax['ur'], cmap=cmap)

        # Compute high pass phase correction

        coh_slice_corrected = corrected[:, :, x, y]
        basis = expansion.TwoHopBasis(coherence.shape[0])
        eval = basis.evaluate_covariance(
            coh_slice_corrected, compl=True)
        triangle_plot(basis, eval, ax=ax['ul'], cmap=cmap)
        basis = expansion.SmallStepBasis(coherence.shape[0])
        eval = basis.evaluate_covariance(
            coh_slice_corrected, compl=True)
        triangle_plot(
            basis, eval, ax=ax['lr'], cmap=cmap)

    # fig, ax = plt.subplots(nrows=2, ncols=3)

    fig, ax = plt.subplot_mosaic([['left', 'um', 'ur'],
                                  ['left', 'ul', 'lr'],
                                  ['left', 'll', 'llr']],
                                 figsize=(6, 3.5), layout="constrained")

    # gs = ax[1, 2].get_gridspec()
    # for ax in ax[0, 0:]:
    #     ax.remove()
    # axbig = fig.add_subplot(gs[0, 0:])
    # print(ax.shape)

    update_triangles(cov, corrected, 0, 0)
    ax['left'].set_title('Intensity \& POI')
    ax['left'].imshow(np.median(intensity[:, :, :], axis=2),
                      cmap=plt.cm.Greys_r)
    scatter = ax['left'].scatter(0, 0, s=30,
                                 facecolors='none', edgecolors='r')

    ax['left'].set_title('Two Hop')
    ax['um'].set_title('Small Step')

    cax = fig.add_axes((0.928, 0.300, 0.015, 0.400))
    # cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar = fig.colorbar(cm.ScalarMappable(
        norm=Normalize(-vabs, vabs, clip=True), cmap=cmap), cax=cax)
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbarlabel = 'phase [$^{\\circ}$]'
    cax.text(2.4, 1.2, cbarlabel, ha='center',
             va='baseline', transform=cax.transAxes)

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            ax['um'].clear()
            ax['ur'].clear()
            ax['ul'].clear()
            ax['lr'].clear()
            update_triangles(cov, corrected, int(
                event.ydata), int(event.xdata))
            scatter.set_offsets((event.xdata, event.ydata))
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

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


main()
