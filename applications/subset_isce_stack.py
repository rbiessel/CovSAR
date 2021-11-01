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
    parser.add_argument('-r', '--rows', type=int, nargs=2,
                        dest='rows', required=False, help='Longitudnal bounds')
    parser.add_argument('-c', '--columns', type=int, nargs=2,
                        dest='cols', required=False, help='Latitudnal bounds')
    parser.add_argument('-o', '--output', type=str,
                        dest='output', required=True, help='Output folder to save stack to')
    args = parser.parse_args()

    return args


def load_image(path, name):
    im = createImage()
    im.load(os.path.join(path, 'geom_reference', f'{name}.xml'))
    print(im.memMap().shape)
    return im.memMap().copy()


def subset_image(inpath, outpath, x1, x2, y1, y2):
    im = createImage()
    im.load(inpath + '.xml')
    mm = im.memMap()
    print(inpath, mm.shape)
    if 'los.rdr.full' in inpath or 'shadowMask.rdr.full' in inpath:
        data = mm[y1:y2, :, x1:x2]
        width = data.shape[2]
        height = data.shape[0]
    else:
        data = mm[y1:y2, x1:x2]
        width = data.shape[1]
        height = data.shape[0]

    im2 = im.clone()
    im2.setWidth(width)
    im2.setLength(height)
    im2.setAccessMode('write')
    im2.filename = outpath
    im2.createImage()

    im2.dump(outpath + '.xml')
    data.tofile(outpath)


def main():
    inputs = readInputs()
    path = inputs.path
    dest_path = os.path.join(os.getcwd(), inputs.output)
    base_path = path
    geom_dir = 'geom_reference'

    if os.path.exists(dest_path):
        print('output folder already exists')
        shutil.rmtree(dest_path)

    os.mkdir(dest_path)
    os.mkdir(os.path.join(dest_path, 'geom_reference'))
    os.mkdir(os.path.join(dest_path, 'SLC'))
    os.mkdir(os.path.join(dest_path, 'baselines'))

    if (inputs.lon is None or inputs.lat is None) and (inputs.rows is None or inputs.cols is None):
        print('Missing bounding coordinates')
        lat = load_image(base_path, geom_dir, 'lat.rdr.full')
        lon = load_image(base_path, geom_dir, 'lon.rdr.full')

        print(f'Largest latitude bounds: {np.min(lat)} to {np.max(lat)}')
        print(f'Largest longitude bounds: {np.min(lon)} to {np.max(lon)}')
        exit

    if (inputs.lon is not None or inputs.lat is not None) and (inputs.rows is None or inputs.cols is None):
        if inputs.lon[0] > inputs.lon[1]:
            print('Longitudes are in wrong order')

        if inputs.lat[0] > inputs.lat[1]:
            print('Latitudes are in wrong order')

        lat = load_image(base_path, 'lat.rdr.full')
        lon = load_image(base_path, 'lon.rdr.full')

        corner1 = np.abs((lat - inputs.lat[0]) * (lon - inputs.lon[0]))
        corner2 = np.abs((lat - inputs.lat[1]) * (lon - inputs.lon[1]))

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

    elif (inputs.rows is not None and inputs.cols is not None):
        x1, x2 = inputs.rows
        y1, y2 = inputs.cols

    print(x1, x2, y1, y2)
    files = ['lat.rdr.full', 'lon.rdr.full', 'hgt.rdr.full',
             'los.rdr.full', 'shadowMask.rdr.full']

    print(base_path)
    # Subset Geometry Files
    for file in files:
        in_path = os.path.join(base_path, geom_dir, file)
        out_path = os.path.join(
            dest_path, geom_dir, os.path.basename(in_path))
        subset_image(in_path, out_path, y1, y2, x1, x2)

    SLCs = glob.glob(os.path.join(base_path, 'SLC', '**/[!geo_]*.slc.full'))
    for slc in SLCs:
        date = slc.split('/')[-2]
        out_folder = os.path.join(
            dest_path, 'SLC', date)
        os.mkdir(out_folder)
        out_path = os.path.join(out_folder, os.path.basename(slc))
        subset_image(slc, out_path, y1, y2, x1, x2)


main()
