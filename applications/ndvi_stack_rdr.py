import numpy as np
from mintpy.geocode import run_geocode as geocode
import glob
from datetime import datetime as dt
import pandas as pd
import rasterio as rio
from matplotlib import pyplot as plt
import os
import argparse
import json
import subprocess


class MyObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path_slc', help='Path to the interferogram.')
    parser.add_argument(
        'path_ndvi', help='Path to the ndvi.')
    args = parser.parse_args()
    return args


def main():

    args = readInputs()
    ndvi_path = os.path.join(args.path_ndvi, '')
    ndvi_paths = glob.glob(ndvi_path + '*.tif')

    print(ndvi_paths)

    latFile = os.path.join(args.path_slc, './geom_reference/lat.rdr.full')
    lonFile = os.path.join(args.path_slc, './geom_reference/lon.rdr.full')

    for file in ndvi_paths:

        date = file.split('/')[-1].replace('.tif', '')
        opath = os.path.join(args.path_slc, 'SLC', date, 'ndvi.rdr.full')
        command_string = f'geocode.py {file} --lat-file {latFile} --lon-file {lonFile} --geo2radar -o {opath}'
        print(command_string)
        subprocess.call(command_string, shell=True)
    return
    geocode()

    for path in paths:
        print(path)
        # run geocode


main()
