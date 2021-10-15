import rasterio
from matplotlib import pyplot as plt
import gdal
import numpy as np
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage

from lc_filter import filter as lc_filter
import library as sarlab
from datetime import datetime as dt
import argparse
import glob
import os
import shutil


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
    # files = files[1:4]

    dates = []
    for file in files:
        date = dt.strptime(file.split('/')[-2], '%Y%m%d')
        dates.append(date)

    n = 2

    SLCs = None
    # clone = None
    if inputs.landcover:
        landcover_src = rasterio.open(inputs.landcover)
        landcover = landcover_src.read()[0]

    for i in range(len(files)):
        print('Loading SLC...')
        im = createImage()
        im.load(files[i] + '.xml')
        mm = im.memMap()
        print(mm.shape)
        if SLCs is None:
            SLCs = np.zeros(
                (len(files), mm.shape[0], mm.shape[1]), dtype=np.complex64)

        SLCs[i, :, :] = mm[:, :, 0]

    # SLCs = SLCs[:, 527:3055, 1808:8985]
    SLCs = SLCs[:, 500:1500, 1800:2400]

    cov = sarlab.get_covariance(SLCs, ml_size=20, coherence=True)
    phi_hist_eig = sarlab.eig_decomp(cov)
    cov = None
    phi_hist_eig = phi_hist_eig.conj()

    for i in range(3):
        plt.imshow(np.angle(phi_hist_eig[:, :, i]))
        plt.show()

    # Check if output folder exists already
    if os.path.exists(os.path.join(os.getcwd(), outputs)):
        print('Output folder already exists, clearing it')
        shutil.rmtree(os.path.join(os.getcwd(), outputs))

    print('creating output folder')
    os.mkdir(os.path.join(os.getcwd(), outputs))

    for i in range(phi_hist_eig.shape[2]):
        print(phi_hist_eig[:, :, i].shape)

        date_str = dates[i].strftime('%Y%m%d')
        os.mkdir(os.path.join(os.getcwd(), outputs, date_str))
        int_path = os.path.join(os.getcwd(), outputs,
                                date_str, f'{date_str}_wrapped.int')
        coh_path = os.path.join(os.getcwd(), outputs,
                                date_str, f'{date_str}_coherence.int')

        write_image(int_path, np.angle(phi_hist_eig[:, :, i]))
        write_image(coh_path, np.abs(phi_hist_eig[:, :, i]))


main()
