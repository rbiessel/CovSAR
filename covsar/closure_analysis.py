from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import struct
from osgeo import gdal, ogr, osr
import os
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import uniform_filter, generic_filter
import glob
import argparse
import csv
from datetime import datetime as dt
import rasterio
from rasterio.windows import Window
import scipy.stats as stats
import skimage.util as skiutil
import xml.etree.ElementTree as ET


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'closures', help='Path to Closures')
    parser.add_argument(
        'landcover', help='Path to landcover map')
    args = parser.parse_args()
    return args


def parseXML(xmlfile):

    # create element tree object
    tree = ET.parse(xmlfile)
    # get root element
    root = tree.getroot()
    # create empty list for news items
    items = {}
    # iterate news items
    for item in root.findall('.//edom'):
        print(item[0].text)
        if item[1].text is not None:
            items[item[0].text] = item[1].text.split('-')[0]

    return items


def main():
    inputs = readInputs()

    closure_paths = glob.glob(inputs.closures)
    print(closure_paths)
    closures = None
    for i in range(len(closure_paths)):
        closure = rasterio.open(closure_paths[i] + '.vrt').read()

        if closures is None:
            closures = np.zeros(
                (len(closure_paths), closure.shape[1], closure.shape[2]))

        closures[i] = closure

    print(closures.shape)

    # load landcover
    landcover = np.squeeze(rasterio.open(inputs.landcover).read())
    lcStrings = parseXML(inputs.landcover + '.xml')
    print(lcStrings)
    print(landcover.shape)

    plt.imshow(landcover)
    plt.show()
    landcover_types = np.unique(landcover)
    print(landcover_types)

    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, closures.shape[0])]

    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

    for landcover_type in landcover_types:
        try:
            stringName = lcStrings[str(landcover_type)]
        except:
            stringName = 'Unassigned'

        for i in range(closures.shape[0]):
            closure = closures[i]
            data = closure[landcover == landcover_type]
            data = data[data != 0]
            all = closure[closure != 0]
            plt.hist(data, bins=200, density=True,
                     label=f'Closure {i}, Mean: {np.round(np.mean(data), 2)}', facecolor=colors[i], alpha=0.7)
            # plt.hist(all, bins=200, density=True,
            #          facecolor='black', alpha=0.4)
        plt.title(f'Class: {landcover_type}')
        plt.legend(loc='lower left')

        plt.show()
        # variance = 1 - np.exp(-np.var(data) / 2)
        # print(f'Mean: {mean}')
        # print(f'Variance: {variance}')


main()
