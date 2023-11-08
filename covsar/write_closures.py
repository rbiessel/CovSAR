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

    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    # clone = None
    if inputs.landcover:
        landcover = np.fromfile(inputs.landcover, dtype=np.int8)

    SLCs = io.load_stack(files)
    print(SLCs.shape)

    if inputs.landcover:
        landcover = np.reshape(landcover, (SLCs.shape[1], SLCs.shape[2]))

    # SLCs = SLCs[:, 450:850, 4500:5300]
    # landcover = landcover[450:850, 4500:5300]

    cov = CovarianceMatrix(SLCs, ml_size=(
        21, 21), sample=(7, 7))
    SLCs = None
    coherence = cov.get_coherence()
    closures.write_closures(coherence, 'closures')


main()
