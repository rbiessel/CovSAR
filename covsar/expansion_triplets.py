import seaborn as sns
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

    print(s)
    phi_corr = np.exp(1j * s * intensities)
    return phi_corr


def get_closures(cov, triplets):
    xi = np.zeros(triplets.shape[0], dtype=np.complex64)
    itriplets = np.zeros(triplets.shape[0])

    for i in range(xi.shape[0]):
        triplet = triplets[i]
        print(cov.shape)
        xi[i] = cov[triplet[0], triplet[1]] * cov[triplet[1],
                                                  triplet[2]] * cov[triplet[0], triplet[2]].conj()

    return xi


def get_iphase(cov, intensity, triplets, Adag):
    xi = np.zeros(triplets.shape[0], dtype=np.complex64)
    itriplets = np.zeros(triplets.shape[0])

    for i in range(xi.shape[0]):
        triplet = triplets[i]
        print(cov.shape)
        xi[i] = cov[triplet[0], triplet[1]] * cov[triplet[1],
                                                  triplet[2]] * cov[triplet[0], triplet[2]].conj()
        itriplets[i] = sarlab.intensity_closure(
            intensity[triplet[0]], intensity[triplet[1]], intensity[triplet[2]], norm=False, cubic=False, filter=1,  function='arctan', kappa=1)

    s = np.polyfit(itriplets, np.angle(xi), 2)[0]
    # s, grid, iter = estimate_s(np.exp(1j * np.angle(xi)),
    #                            itriplets, gridN=2000, rnge=10, gradDescent=False)

    predicted_cphase = np.exp(1j * itriplets * s).flatten()
    est_phase = np.exp(1j * (Adag @ np.angle(predicted_cphase)))

    error_coh = closures.phivec_to_coherence(
        est_phase, cov.shape[0])

    return error_coh


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
    # files = files[34:50]

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
    # clip = [500, 2700, 600, 1000, ]

    SLCs = SLCs[:, clip[0]:clip[1], clip[2]:clip[3]]
    lf = inputs.lf
    N = SLCs.shape[0]

    if inputs.platform == 'S1':
        ml_size = (7*lf, 19*lf)
        sample_size = (2 * lf, 2*lf)

    # if inputs.platform == 'S1':
    #     ml_size = (3*lf, 6*lf)
    #     sample_size = (1, 1)

    elif inputs.platform == 'UAVSAR':
        ml_size = (5 * lf, 5 * lf)
        sample_size = (2 * lf, 2 * lf)

    n = ml_size[0] * ml_size[1]  # Number of looks
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=sample_size)

    SLCs = None

    # coherence = cov.get_coherence()
    # uncorrected = coherence.copy()
    intensity = cov.get_intensity()
    cov = cov.cov

    triplets = closures.get_triplets(cov.shape[0], all=False)
    # A, rank = closures.build_A(
    #     triplets, cov)

    # U, S, Vh = np.linalg.svd(A)
    # A_dagger = np.linalg.pinv(A)

    # basis = expansion.TwoHopBasis(cov.shape[0])
    cmap, vabs = cc.cm['CET_C3'], 180
    # cmap, vabs = cc.cm['CET_C3_r'], 180
    cmap, vabs = plt.cm.seismic, 90
    # p = 40

    #
    lats = io.load_geom_from_slc(files[0], file='lat')[clip[0]:clip[1], clip[2]:clip[3]][
        ::sample_size[0], ::sample_size[1]]
    # [clip[0]:clip[1], clip[2]:clip[3]]
    lons = io.load_geom_from_slc(files[0], file='lon')[clip[0]:clip[1], clip[2]:clip[3]][
        ::sample_size[0], ::sample_size[1]]
    print(lons.dtype)

    # lat_lon_polygons = (68.624416, -149.303349)
    # lat_lon_polygons = (68.627749, -149.309161)
    # lat_lon_reference = (68.620209, -149.293053)
    # 68.608716, -149.276801

    # POIs = {'test2': (68.61790851224154, -149.30411616254935),
    #         'outcrop_e': (68.620209, -149.293053),
    #         'polygons_b': (68.623428, -149.312407),
    #         'polygins_c': (68.622557, -149.317123),
    #         'outcrop_a': (68.619671, -149.302320),
    #         'reference': (68.618584, -149.303809),
    #         'outcrop_c': (68.621911, -149.303943),
    #         'outcrop_d': (68.62081878634847, - 149.29427851856155),
    #         'outcrop_f': (68.62055573645934, -149.2942274071651),
    #         'outcrop_g': (68.62221310244045, -149.31120126598694),
    #         'outcrop_h': (68.61777651446322, -149.30408567147012),
    #         'test': (68.62167772076815, -149.30439476233917),
    #         'outcrop_b': (68.618820, -149.322446),
    #         'tundra': (68.616262, -149.310665),
    #         'tundra_b': (68.6285, -149.3048)}

    # POIs = {
    #     'reference': (68.618584, -149.303809),
    #     'Outcrop A': (68.61777651446322, -149.30408567147012),
    #     # 'outcrop_b': (68.619037, -149.320111),
    #     'Outcrop B': (68.621911, -149.303943),
    #     'Tundra B': (68.62706488016352, -149.30803114685497),
    #     'Tundra A': (68.622557, -149.317123),
    # }

    POIs = {
        'reference': (36.395973, -115.329958),
        'alluvium': (36.377915, -115.35607),
    }

    POIs = {
        'reference': (70.125940, -143.615788),
        'polygons': (70.121374, -143.670639)
    }

    for key in POIs:
        POIs[key] = latlon_to_index(
            lats, lons, POIs[key][0], POIs[key][1])
        print(POIs[key])

    def ml_eval(eval):
        # return np.mean(eval, axis=(1, 2))
        return np.mean(eval.real, axis=(1, 2)) + 1j * np.mean(eval.imag, axis=(1, 2))

    def update_triangles(coherence, x, y, key):
        # coh_slice = coherence[:, :, x, y]
        # inds = np.tril_indices(coh_slice.shape[0])

        # Cvec = coh_slice[inds[0], inds[1], :, :]
        # Cvec = np.moveaxis(Cvec, 0, -1)

        islice = intensity[x, y, :]
        vectorized = False

        basis = expansion.TwoHopBasis(coherence.shape[0])
        eval = basis.evaluate_covariance(
            coherence[:, :, x, y], compl=True, normalize=False, vectorized=vectorized)

        triangle_plot(basis, eval,
                      ax=ax['tsal'], cmap=cmap, vabs=vabs)

        basis = expansion.SmallStepBasis(coherence.shape[0])
        eval = basis.evaluate_covariance(
            coherence[:, :, x, y], compl=True, normalize=False, vectorized=vectorized)

        triangle_plot(basis, eval,
                      ax=ax['tsbl'], cmap=cmap, vabs=vabs)

        ax['tsal'].set_title('Two Hop Basis Closure Phase')
        ax['tsbl'].set_title('Small Step Basis Closure Phase')

        # Compute Timeseries relative to Reference
        coh_slice = coherence[:, :, x, y]
        coh_reference = coherence[:, :,
                                  POIs['reference'][0], POIs['reference'][1]]

        m = coh_slice.shape[0]
        coh_ts = coh_slice[:, :, np.newaxis, np.newaxis]
        coh_reference = coh_reference[:, :, np.newaxis, np.newaxis]
        m = m - 1
        coherence_m = sarlab.reduce_cov(coh_ts, keep_diag=m)
        coh_reference_m = sarlab.reduce_cov(coh_reference, keep_diag=m)
        timeseries_m = sarlab.eig_decomp(coherence_m)[0, 0]
        ts_ref_m = sarlab.eig_decomp(coh_reference_m)[0, 0]

        tsmm = np.angle(timeseries_m * ts_ref_m.conj()) * 56 / (np.pi * 4)
        # ax['tsa'].plot(tsmm, label=f"bw-{m}", color='black', linewidth=3)

        m = 2
        coherence_m = sarlab.reduce_cov(coh_ts, keep_diag=m)
        coh_reference_m = sarlab.reduce_cov(coh_reference, keep_diag=m)
        timeseries_m = sarlab.eig_decomp(coherence_m)[0, 0]
        ts_ref_m = sarlab.eig_decomp(coh_reference_m)[0, 0]

        tsmm = np.angle(timeseries_m * ts_ref_m.conj()) * 56 / (np.pi * 4)
        # ax['tsa'].plot(tsmm, linestyle='dashed',
        #                label=f"bw-{m}",  color='steelblue', linewidth=3)

        m = 5
        coherence_m = sarlab.reduce_cov(coh_ts, keep_diag=m)
        coh_reference_m = sarlab.reduce_cov(coh_reference, keep_diag=m)
        timeseries_m = sarlab.eig_decomp(coherence_m)[0, 0]
        ts_ref_m = sarlab.eig_decomp(coh_reference_m)[0, 0]

        tsmm = np.angle(timeseries_m * ts_ref_m.conj()) * 56 / (np.pi * 4)
        # ax['tsa'].plot(tsmm, linestyle=(0, (5, 1)),
        #                label=f"bw-{m}", color='tomato', linewidth=3)

        ##
        # Compute Timeseries relative to full bw
        coh_slice = coherence[:, :, x, y]

        m = coh_slice.shape[0]
        coherence_m = sarlab.reduce_cov(coh_ts, keep_diag=m)
        coh_reference_m = sarlab.reduce_cov(coh_reference, keep_diag=m)
        timeseries_full = sarlab.eig_decomp(coherence_m)[0, 0]

        # tsmm = np.angle(timeseries_m * ts_ref_m.conj()) * 56 / (np.pi * 4)
        # ax['tsa'].plot(tsmm, label=f"bw-{m}")

        m = 2
        coherence_m = sarlab.reduce_cov(coh_ts, keep_diag=m)
        coh_reference_m = sarlab.reduce_cov(coh_reference, keep_diag=m)
        timeseries_m = sarlab.eig_decomp(coherence_m)[0, 0]
        tsmm = np.angle(timeseries_m * timeseries_full.conj()
                        ) * 56 / (np.pi * 4)
        ax['tsb'].plot(
            tsmm, 'o-', label=f"bw-{m}", color='steelblue', linewidth=1.5, markersize=5)
        m = 5
        coherence_m = sarlab.reduce_cov(coh_ts, keep_diag=m)
        coh_reference_m = sarlab.reduce_cov(coh_reference, keep_diag=m)
        timeseries_m = sarlab.eig_decomp(coherence_m)[0, 0]

        tsmm = np.angle(timeseries_m * timeseries_full.conj()
                        ) * 56 / (np.pi * 4)
        ax['tsb'].plot(tsmm, 'o-',
                       label=f"bw-{m}", color='tomato', linewidth=1.5, markersize=5)

        ax['tsb'].plot((islice - islice[0]), 'o-', label=f"Intensity",
                       color='black', linewidth=1.5, markersize=5)

        if 'Tundra A' in key:
            # ax['tsa'].legend(loc='best', framealpha=0.1)
            ax['tsb'].legend(loc='best', framealpha=0.1)

        # Labels

        # ax['tsa'].set_title('Relative to Reference')
        # ax['tsa'].set_xlabel('scene')
        # ax['tsa'].set_ylabel('[$mm$]')

        ax['tsb'].set_title('Relative to bw-full')
        ax['tsb'].set_xlabel('scene')
        ax['tsb'].set_ylabel('[$mm$, dB]')

    # fig, ax = plt.subplots(nrows=2, ncols=3)

    for key in POIs:
        if 'reference' not in key:
            fig, ax = plt.subplot_mosaic([['left', 'tsb', 'tsb'],
                                          ['left', 'tsal', 'tsbl']],
                                         figsize=(6, 3), layout="constrained")

            fig.suptitle(key, fontsize=16)
            x, y = POIs[key]
            update_triangles(cov, x, y, key)
            ax['left'].set_title('Intensity \& POI')
            ax['left'].imshow(np.median(intensity[:, :, :], axis=2),
                              cmap=plt.cm.Greys_r)
            scatter = ax['left'].scatter(y, x, s=30,
                                         facecolors='none', edgecolors='r')
            reference = ax['left'].scatter(POIs['reference'][1], POIs['reference'][0], s=50,
                                           facecolors='none', edgecolors='black', marker='*')
            ax['left'].set_title('Median Intensity \& POI')

            cax = fig.add_axes((1.5, 0.300, 0.015, 0.400))
            cbar = fig.colorbar(cm.ScalarMappable(
                norm=Normalize(-vabs, vabs, clip=True), cmap=cmap), cax=cax)
            cbar.set_ticks([-vabs, -vabs/2, 0, vabs/2, vabs])
            cbarlabel = 'phase [$^{\\circ}$]'
            cax.text(2.4, 1.2, cbarlabel, ha='center',
                     va='baseline', transform=cax.transAxes)

            def onclick(event):
                if event.xdata != None and event.ydata != None:
                    # ax['tsa'].clear()
                    ax['tsb'].clear()
                    ax['tsal'].clear()
                    ax['tsbl'].clear()
                    # ax['hist'].clear()

                    x, y = int(np.floor(event.xdata)), int(
                        np.floor(event.ydata))
                    print('Lat Lon:', lats[y, x], lons[y, x])
                    update_triangles(cov, y, x, 'Tundra A')
                    scatter.set_offsets((event.xdata, event.ydata))
                    plt.draw()

            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.savefig(f'/Users/rbiessel/Documents/imnav_{key}.png',
                        dpi=300, transparent=True)
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
