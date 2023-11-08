
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime as dt
import argparse
import glob
import os
from scipy import special
import pandas as pd
from covariance import CovarianceMatrix
import rasterio
from latlon import latlon_to_index
from rasterio import logging
import closures
import library as sarlab
from scipy.stats import gaussian_kde


log = logging.getLogger()
log.setLevel(logging.ERROR)


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the merged folder.')
    args = parser.parse_args()

    return args


def main():
    inputs = readInputs()
    merged_path = inputs.path
    stack_path = os.path.expanduser(merged_path)
    stack_path = os.path.join(stack_path, './SLC/**/*.slc.full')
    print(stack_path)
    files = glob.glob(stack_path)
    files = sorted(files)

    start = 0
    stop = 30

    files = files[start:stop]
    dates = []

    safe_files = np.loadtxt(os.path.join(
        merged_path, '../SAFE_files.txt'), dtype=str)
    safe_files = sorted(safe_files)
    safe_files = safe_files[start:stop]

    print(safe_files)

    for safe_files in safe_files:
        date = safe_files.split('/')[-1].split('_')[5]
        dates.append(date)

    import isceio as io
    SLCs = io.load_stack_vrt(files)
    # inc_map = io.load_inc_from_slc(files[0])

    # SLCs = SLCs[:, 0:199, 1100:2101]
    n = 19
    # 7 x 19
    # cov = CovarianceMatrix(SLCs, ml_size=(7, 19))
    # cov = CovarianceMatrix(SLCs, ml_size=(7, 19), sample=(2, 5))
    lf = 2
    ml_size = (7*lf, 19*lf)
    resample_size = (2 * lf, 5 * lf)
    # ml_size = (1)
    n = ml_size[0] * ml_size[1]
    cov = CovarianceMatrix(SLCs, ml_size=ml_size, sample=resample_size)
    intensity = cov.get_intensity()
    coherence = cov.get_coherence()
    print(intensity.shape)

    from dataimport.smap_val_data import get_sm_ts
    sm_ts = get_sm_ts()

    lats = io.load_geom_from_slc(files[0], file='lat')[
        ::resample_size[0], ::resample_size[1]]
    lons = io.load_geom_from_slc(files[0], file='lon')[
        ::resample_size[0], ::resample_size[1]]
    print(lats.shape)
    print(intensity.shape)

    sardates = np.array([dt.strptime(date, '%Y%m%dT%H%M%S')
                         for date in dates], dtype=np.datetime64)

    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0], all=False)

    # variances = np.var(triplets, axis=1)
    # triplets = triplets[np.argsort(np.var(triplets, axis=1))]
    triplets = triplets[0:int(k)]

    A, rank = closures.build_A(triplets, coherence)
    U, S, Vh = np.linalg.svd(A)
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]
    closure_stack = np.zeros((
        len(triplets), coherence.shape[2], coherence.shape[3]), dtype=np.complex64)

    amp_triplet_stack = closure_stack.copy()
    amp_triplet_stack = amp_triplet_stack.astype(np.float64)

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        clml = (5, 5)
        closure = sarlab.multilook(closure, ml=clml)

        # mean_coherence = (np.abs(coherence[triplet[0], triplet[1]]) + np.abs(
        #     coherence[triplet[1], triplet[2]]) + np.abs(coherence[triplet[0], triplet[2]].conj())) / 3

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False)

        # fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
        # ax[0].imshow(np.angle(closure), vmin=-np.pi/2,
        #              vmax=np.pi/2, cmap=plt.cm.seismic)
        # ax[1].imshow(np.sign(amp_triplet) *
        #              np.log10(np.abs(amp_triplet)), cmap=plt.cm.seismic)
        # plt.show()
        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet

    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    for ts in sm_ts:

        index = latlon_to_index(lats, lons, ts.attrs['lat'], ts.attrs['lon'])
        intensities = intensity[index[0], index[1], :]

        data = {'Dates': sardates.astype(
            ts['Dates'].dtype), 'Power': intensities}

        df_SAR = pd.DataFrame(data=data)

        df_a = pd.merge_asof(df_SAR, ts, on="Dates",
                             tolerance=pd.Timedelta("12d"))

        # Setup intensities and intensity differences
        sm = df_a['WASM']
        raw_intensities = intensities
        smdiffs = np.tile(sm, (len(sm), 1))
        idiffs = np.tile(intensities, (len(intensities), 1))
        smdiffs = smdiffs.T - smdiffs
        idiffs = idiffs.T - idiffs

        window_amps = amp_triplet_stack[:, index[0], index[1]]
        window_closure = closure_stack[:, index[0], index[1]]

        coh = coherence[:, :, index[0], index[1]]

        plt.imshow(
            np.abs(coh), cmap=plt.cm.Greys_r, vmin=0, vmax=1)
        plt.title('Coherence')
        plt.show()

        plt.scatter(closures.coherence_to_phivec(idiffs).flatten(),
                    np.angle(closures.coherence_to_phivec(coh)).flatten())
        plt.xlabel('Amp Dif')
        plt.ylabel('Phase')
        plt.show()

        plt.scatter(closures.coherence_to_phivec(smdiffs).flatten(),
                    np.angle(closures.coherence_to_phivec(coh)).flatten())
        plt.xlabel('SM Dif')
        plt.ylabel('Phase')
        plt.show()

        fitform = 'linear'
        # estimate relationship between amplitudes and closures
        coeff, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
        ), W=None, function=fitform)

        # plt.scatter(idiffs.flatten(), np.abs(
        #     coherence[:, :, index[0], index[1]]).flatten())
        # plt.xlabel('Soil Moisture difference')
        # plt.ylabel('Coherence')
        # plt.show()

        print(ts.shape)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Power', color=color)
        ax1.plot(df_a['Dates'], df_a['Power'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('SWC', color=color)
        ax2.plot(df_a['Dates'], df_a['WASM'],
                 label='Soil Moisture', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        fig, ax = plt.subplots(ncols=2, nrows=1)

        ax[0].scatter(df_a['WASM'], df_a['Power'])
        ax[0].set_xlabel('SWC')
        ax[0].set_ylabel('SAR Power')

        xy = np.vstack(
            [window_amps.flatten(), np.angle(window_closure).flatten()])
        z = gaussian_kde(xy)(xy)

        ax[1].scatter(window_amps.flatten(), np.angle(
            window_closure).flatten(), c=z, s=10)
        ax[1].set_xlabel('Power Ratio Triplet')
        ax[1].set_ylabel('Closure Phase')
        plt.show()


main()
