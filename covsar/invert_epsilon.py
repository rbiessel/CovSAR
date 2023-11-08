from tkinter.tix import Tree
from numpy.core.arrayprint import _leading_trailing
from numpy.lib.polynomial import polyval
from pkg_resources import to_filename
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from scipy import special
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
from sm_forward import SMForward
from closures import write_closures


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
    # files = files[3:6]
    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    # clone = None
    if inputs.landcover:
        landcover = np.squeeze(rasterio.open(inputs.landcover).read())
        print('landcover:')
        print(landcover.shape)

    SLCs = io.load_stack(files)
    inc_map = io.load_inc_from_slc(files[0])

    cov = CovarianceMatrix(SLCs, ml_size=(41, 41))
    SLCs = None
    coherence = cov.get_coherence()

    intensities = sarlab.get_intensity(cov.cov)
    epsilons = sarlab.intensity_to_epsilon(intensities)
    intensities = None
    cov = None

    k = special.comb(coherence.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence.shape[0])
    print(epsilons.shape)
    forward = SMForward(1, 1, 1)
    write_closures(
        coherence, '/Users/rbiessel/Documents/InSAR/saltLakeClosures')
    return
    for sf in np.linspace(1/10, 2, 10):
        for tf in np.linspace(0, 15, 10):

            imaginary_part = np.random.rand(3) * sf

            e1 = (epsilons[:, :, 0] * sf + tf) + imaginary_part[0] * 1j
            e2 = (epsilons[:, :, 1] * sf + tf) + imaginary_part[1] * 1j
            e3 = (epsilons[:, :, 2] * sf + tf) + imaginary_part[2] * 1j

            phi_sm_12 = forward.get_phases_dezan(e1, e2, use_epsilon=True)
            phi_sm_23 = forward.get_phases_dezan(e2, e3, use_epsilon=True)
            phi_sm_13 = forward.get_phases_dezan(e1, e3, use_epsilon=True)

            closure = phi_sm_12 * phi_sm_23 * np.conj(phi_sm_13)

            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
            ax[0].imshow(np.angle(closure), vmin=-np.pi/4,
                         vmax=np.pi/4, cmap=plt.cm.seismic)
            observed = coherence[0, 1] * \
                coherence[1, 2] * coherence[0, 2].conj()

            ax[1].imshow(np.angle(observed), vmin=-np.pi/4,
                         vmax=np.pi/4, cmap=plt.cm.seismic)
            ax[2].imshow(np.sqrt((closure.real - observed.real) **
                                 2 + (closure.imag - observed.imag)**2))

            plt.show()


main()
