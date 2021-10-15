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
        'path', help='Path to the merged folder.')

    parser.add_argument('-l', '--lon', type=float, nargs=2,
                        dest='lon', required=False, help='Longitudnal bounds')
    parser.add_argument('-L', '--lat', type=float, nargs=2,
                        dest='lat', required=False, help='Latitudnal bounds')
    parser.add_argument('-o', '--output', type=str,
                        dest='output', required=True, help='Output folder to save stack to')
    args = parser.parse_args()

    return args


def load_image(path, name):
    im = createImage()
    im.load(os.path.join(path, 'geom_reference', f'{name}.xml'))
    return im.memMap().copy()


def main():
    inputs = readInputs()

    path = inputs.path

    dest_path = os.path.join(os.getcwd(), inputs.output)

    if os.path.exists(dest_path):
        print('output folder already exists')
        shutil.rmtree(dest_path)

    os.mkdir(dest_path)

    if inputs.lon is None or inputs.lat is None:
        print('Missing bounding coordinates')
        lat = load_image(path, 'lat.rdr.full')
        lon = load_image(path, 'lon.rdr.full')

        print(f'Largest latitude bounds: {np.min(lat)} to {np.max(lat)}')
        print(f'Largest longitude bounds: {np.min(lon)} to {np.max(lon)}')
        exit

    if inputs.lon[0] > inputs.lon[1]:
        print('Longitudes are in wrong order')

    if inputs.lat[0] > inputs.lat[1]:
        print('Latitudes are in wrong order')

    lat = load_image(path, 'lat.rdr.full')
    lon = load_image(path, 'lon.rdr.full')

    corner1 = np.abs((lat - inputs.lat[0]) * (lon - inputs.lon[0]))
    corner2 = np.abs((lat - inputs.lat[1]) * (lon - inputs.lon[1]))

    print(corner1.shape)
    index1 = np.unravel_index(
        np.argmin(corner1), (corner1.shape[0], corner1.shape[1]))
    index2 = np.unravel_index(
        np.argmin(corner2), (corner2.shape[0], corner2.shape[1]))

    print(index1[1], index1[0])
    print(index2[1], index2[0])

    y1 = index2[0]
    y2 = index1[0]

    corner1 = np.abs((lat - inputs.lat[0]) * (lon - inputs.lon[1]))
    corner2 = np.abs((lat - inputs.lat[1]) * (lon - inputs.lon[0]))

    index1 = np.unravel_index(
        np.argmin(corner1), (corner1.shape[0], corner1.shape[1]))
    index2 = np.unravel_index(
        np.argmin(corner2), (corner2.shape[0], corner2.shape[1]))

    x1 = index1[1]
    x2 = index2[1]

    lat = lat[y1:y2, x1:x2]
    lon = lon[y1:y2, x1:x2]

    print(y1, y2, x1, x2)


main()
