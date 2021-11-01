import rasterio
from matplotlib import pyplot as plt
import gdal
import numpy as np
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage
import library as sarlab
from datetime import datetime as dt
import argparse
import glob
import os
import shutil
from covariance import CovarianceMatrix


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


def write_image(outfile_path, data):

    image = createImage()
    image.setWidth(data.shape[1])
    image.setLength(data.shape[0])
    image.setAccessMode('write')
    image.filename = outfile_path
    image.dataType = 'FLOAT'
    image.createImage()

    image.dump(f'{outfile_path}.xml')
    data.tofile(outfile_path)


def main():
    inputs = readInputs()
    stack_path = inputs.path
    stack_path = os.path.expanduser(stack_path)
    outputs = inputs.output
    files = glob.glob(stack_path)
    files = sorted(files)
    files = files[0:3]

    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    SLCs = None
    # clone = None
    if inputs.landcover:
        landcover_src = rasterio.open(inputs.landcover)
        landcover = landcover_src.read()[0]

    for i in range(len(files)):
        print(f'Loading SLC {i} / {len(files)}...')
        im = createImage()
        im.load(files[i] + '.xml')
        mm = im.memMap()
        if SLCs is None:
            SLCs = np.zeros(
                (len(files), mm.shape[0], mm.shape[1]), dtype=np.complex64)

        SLCs[i, :, :] = mm[:, :, 0]

    # SLCs = SLCs[:, 100:500, 200:500]

    cov = CovarianceMatrix(SLCs, ml_size=(20, 20))
    SLCs = None
    coherence = cov.get_coherence()
    cov = None

    phi_hist_eig = sarlab.eig_decomp(coherence)
    phi_hist_eig = phi_hist_eig.conj()
    kappa = sarlab.compute_tc(coherence, phi_hist_eig)
    return
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

        write_image(int_path, np.angle(phi))

    kappa_path = os.path.join(os.getcwd(), outputs,
                              '../temporal_coherence.int')
    write_image(kappa_path, np.abs(kappa))


main()
