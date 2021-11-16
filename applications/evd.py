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
    files = files[1:7]

    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    # clone = None
    if inputs.landcover:
        landcover_src = rasterio.open(inputs.landcover)
        landcover = landcover_src.read()[0]

    SLCs = io.load_stack(files)
    # SLCs = SLCs[:, 100:400, 100:300]
    print(SLCs.shape)
    cov = CovarianceMatrix(SLCs, ml_size=(30, 30))
    SLCs = None
    coherence = cov.get_coherence()
    closures.write_closures(coherence, 'closures')
    return
    # intensity = cov.get_intensity()
    cov = None

    # Check if output folder exists already
    if os.path.exists(os.path.join(os.getcwd(), outputs)):
        print('Output folder already exists, clearing it')
        shutil.rmtree(os.path.join(os.getcwd(), outputs))

    print('creating output folder')
    os.mkdir(os.path.join(os.getcwd(), outputs))

    phi_hist_eig = sarlab.eig_decomp(coherence)
    phi_hist_eig = phi_hist_eig.conj()
    kappa = sarlab.compute_tc(coherence, phi_hist_eig)

    # Write out nearest neighbor pairs
    for i in range(1, phi_hist_eig.shape[2]):
        phi = phi_hist_eig[:, :, i] * phi_hist_eig[:, :, i - 1].conj()
        # db = intensity[:, :, i] - intensity[:, :, i - 1]
        date_str = dates[i]
        ref_date = dates[i - 1]
        date_str = f'{ref_date}_{date_str}'
        os.mkdir(os.path.join(os.getcwd(), outputs, f'{date_str}'))
        int_path = os.path.join(os.getcwd(), outputs,
                                date_str, f'{date_str}_wrapped.int')

        intensity_path = os.path.join(os.getcwd(), outputs,
                                      date_str, f'{date_str}_db.int')

        io.write_image(int_path, np.angle(phi))
        # io.write_image(intensity_path, db)

    kappa_path = os.path.join(os.getcwd(), outputs,
                              '../temporal_coherence.int')
    io.write_image(kappa_path, np.abs(kappa))


main()
