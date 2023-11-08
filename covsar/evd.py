from random import sample
from numpy.core.arrayprint import _leading_trailing
from numpy.lib.polynomial import polyval
import rasterio
from matplotlib import pyplot as plt
import numpy as np
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
from subset_isce_stack import subset_image


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


def write_timeseries(phi_hist_eig, dates, kappa, outputs, intensity=None, geocode=None):

    # Check if output folder exists already
    if os.path.exists(os.path.join(os.getcwd(), outputs)):
        print('Output folder already exists, clearing it')
        shutil.rmtree(os.path.join(os.getcwd(), outputs))

    print('creating output folder')
    os.mkdir(os.path.join(os.getcwd(), outputs))

    # Write out nearest neighbor pairs
    for i in range(1, phi_hist_eig.shape[2]):
        phi = phi_hist_eig[:, :, i] * phi_hist_eig[:, :, i - 1].conj()
        date_str = dates[i]
        ref_date = dates[i - 1]
        date_str = f'{ref_date}_{date_str}'
        os.mkdir(os.path.join(os.getcwd(), outputs, f'{date_str}'))
        int_path = os.path.join(os.getcwd(), outputs,
                                date_str, f'{date_str}_wrapped.int')
        io.write_image(int_path, np.angle(phi), geocode=geocode)
        if intensity is not None:
            db = intensity[:, :, i] - intensity[:, :, i - 1]
            intensity_path = os.path.join(os.getcwd(), outputs,
                                          date_str, f'{date_str}_db.int')
            io.write_image(intensity_path, db, geocode=geocode)

    kappa_path = os.path.join(os.getcwd(), outputs,
                              './temporal_coherence.int')
    io.write_image(kappa_path, np.abs(kappa), geocode=geocode)


def write_geometry(geom_path, outputs, clip=[0, -1, 0, -1], sample_size=(1, 1)):
    if os.path.exists(os.path.join(os.getcwd(), outputs)):
        print('Output folder already exists, clearing it')
        shutil.rmtree(os.path.join(os.getcwd(), outputs))

    print('creating output geometry folder')
    os.mkdir(os.path.join(os.getcwd(), outputs))

    files = ['lat', 'lon', 'hgt', 'incLocal', 'shadowMask', 'los']

    for file in files:
        try:
            path = os.path.join(geom_path, f'{file}.rdr.full')
            outpath = os.path.join(outputs, f'{file}.rdr.full')
            subset_image(path, outpath, clip[2], clip[3],
                         clip[0], clip[1], sample=sample_size)
        except:
            print(f'Could not locate file {file} -- Ignoring...')


def main():
    inputs = readInputs()
    stack_path = inputs.path
    stack_path = os.path.expanduser(stack_path)
    outputs = inputs.output
    files = glob.glob(stack_path)
    files = sorted(files)
    # files = files[4:-1]

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
    print(SLCs.shape)

    cov = CovarianceMatrix(SLCs, ml_size=(
        21, 21), sample=(7, 7))
    SLCs = None
    coherence = cov.get_coherence()
    cov = None

    phi_hist_eig = sarlab.eig_decomp(coherence)
    phi_hist_eig = phi_hist_eig.conj()
    kappa = sarlab.compute_tc(coherence, phi_hist_eig)

    write_timeseries(phi_hist_eig, dates, kappa, outputs)


if __name__ == "__main__":
    main()
