from matplotlib import pyplot as plt
import rasterio
import gdal
import numpy as np
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage

from datetime import datetime as dt
import argparse
import glob
import os
import shutil


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the interferogram.')
    parser.add_argument('-l', '--lat', type=str, dest='lat',
                        required=False, help='latitudes')
    parser.add_argument('-L', '--lon', type=str,
                        dest='lon', required=True, help='longitudes')
    args = parser.parse_args()

    return args


def load_image(path):
    im = createImage()
    im.load(path + '.xml')
    return np.squeeze(im.memMap().copy())


def main():
    inputs = readInputs()
    path = inputs.path
    lat = load_image(inputs.lat)
    lon = load_image(inputs.lon)

    lon_lat_rdr = np.array([lon, lat])
    print(lon_lat_rdr.shape)

    toProject = rasterio.open(inputs.path)
    xy = np.indices(np.squeeze(toProject.read()).shape)
    data = toProject.read()
    print(xy.shape)
    print(xy[:, 0, 100])

    lon_lat = np.array(toProject.transform * xy)

    print(lon_lat[:, 0, 0])

    xy = np.indices(lat.shape)
    print(xy.shape)
    xy = xy.reshape((2, xy.shape[1] * xy.shape[2]))
    print(xy.shape)
    new_file = np.zeros(lat.shape, dtype=data.dtype)

    print('Projecting Image')
    for i in range(xy.shape[1]):
        index = xy[:, i]
        print(np.argmin(
            np.abs(lon_lat - lon_lat_rdr[:, index[0], index[1]][:, np.newaxis, np.newaxis])))
        # np.abs(lon_lat - lon_lat_rdr[index]
        # data_i=np.argmin()
        # print(data_i)
        # new_file[index] = data[data_i]
    return
    mx, my = 500, 600  # coord in map units, as in question
    px = mx * x_size + x_min  # x pixel
    py = my * y_size + y_min  # y pixel

    print(lat.shape)
    print(lon.shape)


main()
