from matplotlib import pyplot as plt
import numpy as np

from datetime import datetime as dt
import argparse
import glob
import os
import shutil
import library as sarlab
import isceio as io


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Path to the merged folder.')
    # parser.add_argument('-l', '--lon', type=float, nargs=2,
    #                     dest='lon', required=False, help='Longitudnal bounds')
    # parser.add_argument('-L', '--lat', type=float, nargs=2,
    #                     dest='lat', required=False, help='Latitudnal bounds')
    parser.add_argument('-ro', '--rows-out', type=int, nargs=2,
                        dest='rowsout', required=True)
    parser.add_argument('-co', '--columns-out', type=int, nargs=2,
                        dest='colsout', required=True)
    parser.add_argument('-ri', '--rows-in', type=int, nargs=1,
                        dest='rowsin', required=True)
    parser.add_argument('-ci', '--columns-in', type=int, nargs=1,
                        dest='colsin', required=True)
    parser.add_argument('-cl', '--column-looks', type=int, nargs=1,
                        dest='cl', required=True)
    parser.add_argument('-rl', '--row-looks', type=int, nargs=1,
                        dest='rl', required=True)
    parser.add_argument('-o', '--output', type=str,
                        dest='output', required=True, help='Output folder to save stack to')
    parser.add_argument('-s', '--sample', type=int, nargs=2,
                        dest='sample_size', required=True, help='Output folder to save stack to')
    args = parser.parse_args()

    return args


def subset_image(inpath, outpath, cols, rows, colstart, colend, rowstart, rowend, collooks=1, rowlooks=1, sample=None):
    if '.llh' in inpath:
        cols = int(np.floor(cols / collooks))
        rows = int(np.floor(rows / rowlooks))

        im = np.fromfile(inpath, dtype=np.float32).reshape((cols, rows, 3))

        colstart = int(np.floor(colstart / collooks))
        colend = int(np.floor(colend / collooks))
        rowstart = int(np.floor(rowstart / rowlooks))
        rowend = int(np.floor(rowend / rowlooks))

        data = im[colstart:colend, rowstart:rowend, :]

        io.write_image(outpath.replace('.llh', '.lat.llh'), data[:, :, 0])
        io.write_image(outpath.replace('.llh', '.lon.llh'), data[:, :, 1])
        io.write_image(outpath.replace('.llh', '.dem.llh'), data[:, :, 2])
    else:
        im = np.fromfile(inpath, dtype=np.complex64).reshape((cols, rows))
        data = im[colstart:colend, rowstart:rowend]
        print(f'Out Shape: (cols = {data.shape[0]}, rows = {data.shape[1]})')
        if sample is not None:
            data = data[::sample[0], ::sample[1]]

        data.tofile(outpath)


def main():
    inputs = readInputs()
    path = inputs.path
    dest_path = os.path.join(os.getcwd(), inputs.output)
    base_path = path

    if os.path.exists(dest_path):
        print('output folder already exists')
        shutil.rmtree(dest_path)

    os.mkdir(dest_path)
    os.mkdir(os.path.join(dest_path, 'SLC'))

    SLCs = glob.glob(os.path.join(base_path, '*.slc'))
    lookfile = glob.glob(os.path.join(base_path, '*.llh'))[0]
    lookfile_out = os.path.join(dest_path, os.path.basename(lookfile))

    collooks = 8 / inputs.cl[0]
    rowlooks = 2 / inputs.rl[0]

    print(lookfile)
    subset_image(
        lookfile, lookfile_out, inputs.colsin[0], inputs.rowsin[0], inputs.colsout[0], inputs.colsout[1], inputs.rowsout[0], inputs.rowsout[1], collooks=collooks, rowlooks=rowlooks)
    # return
    for slc in SLCs:
        out_path = os.path.join(dest_path, os.path.basename(slc))
        print(out_path)
        subset_image(
            slc, out_path, inputs.colsin[0], inputs.rowsin[0], inputs.colsout[0], inputs.colsout[1], inputs.rowsout[0], inputs.rowsout[1])


main()
