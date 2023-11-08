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
                        dest='cl', required=False, default=2)
    parser.add_argument('-rl', '--row-looks', type=int, nargs=1,
                        dest='rl', required=False, default=8)
    parser.add_argument('-o', '--output', type=str,
                        dest='output', required=True, help='Output folder to save stack to')
    parser.add_argument('-s', '--sample', type=int, nargs=2,
                        dest='sample_size', required=False, default=(1, 1), help='Resample the image by')
    parser.add_argument('-p', '--pol', type=str, nargs=1,
                        dest='pol', required=False, default='VV', help='Desired polarization. Default VV')
    args = parser.parse_args()

    return args


def subset_image(inpath, outpath, cols, rows, colstart, colend, rowstart, rowend, collooks=1, rowlooks=1, sample=None):
    if '.llh' in inpath:
        cols = int(np.floor(cols / collooks))
        rows = int(np.floor(rows / rowlooks))

        im = np.fromfile(inpath, dtype=np.float32).reshape((rows, cols, 3))

        colstart = int(np.floor(colstart / collooks))
        colend = int(np.floor(colend / collooks))
        rowstart = int(np.floor(rowstart / rowlooks))
        rowend = int(np.floor(rowend / rowlooks))

        data = im[rowstart:rowend, colstart:colend, :]

        io.write_image(os.path.join(os.path.dirname(outpath),
                                    'lat.rdr.full'), data[:, :, 0])
        io.write_image(os.path.join(os.path.dirname(outpath),
                                    'lon.rdr.full'), data[:, :, 1])
        io.write_image(os.path.join(os.path.dirname(outpath),
                                    'hgt.rdr.full'), data[:, :, 2])
    else:
        im = np.fromfile(inpath, dtype=np.complex64).reshape((rows, cols))
        data = im[rowstart:rowend, colstart:colend]
        print(f'Out Shape: (cols = {data.shape[0]}, rows = {data.shape[1]})')
        if sample is not None:
            data = data[::sample[0], ::sample[1]]

        io.write_image(outpath, data, dtype='CFLOAT')


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
    os.mkdir(os.path.join(dest_path, 'geom_reference'))

    SLCs = glob.glob(os.path.join(base_path, '*.slc'))

    SLCs = [slc for slc in SLCs if inputs.pol[0] in slc]

    lookfile = glob.glob(os.path.join(base_path, '*.llh'))[0]
    lookfile_out = os.path.join(
        dest_path, 'geom_reference', os.path.basename(lookfile))

    collooks = 2 / inputs.cl[0]
    rowlooks = 8 / inputs.rl[0]

    print(lookfile)
    subset_image(
        lookfile, lookfile_out, inputs.colsin[0], inputs.rowsin[0], inputs.colsout[0], inputs.colsout[1], inputs.rowsout[0], inputs.rowsout[1], collooks=collooks, rowlooks=rowlooks)
    # return
    for slc in SLCs:
        date = '20' + os.path.basename(slc).split('_')[4]
        os.mkdir(os.path.join(dest_path, 'SLC', date))
        out_path = os.path.join(dest_path, 'SLC', date, os.path.basename(slc))
        print(out_path)
        subset_image(
            slc, out_path, inputs.colsin[0], inputs.rowsin[0], inputs.colsout[0], inputs.colsout[1], inputs.rowsout[0], inputs.rowsout[1])


main()
