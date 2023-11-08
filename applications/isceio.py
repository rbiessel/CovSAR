import rasterio
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage
import numpy as np
import os
from matplotlib import pyplot as plt
import library as sarlab


def write_image(outfile_path, data, geocode=None, length=None, width=None, dtype='FLOAT'):
    '''
        Write out image in ISCE format.
    '''

    image = createImage()

    if width is None:
        width = data.shape[1]
    if length is None:
        length = data.shape[0]

    print(length, width)
    image.setWidth(width)
    image.setLength(length)
    image.setAccessMode('write')
    image.filename = outfile_path
    image.dataType = dtype
    image.createImage()
    image.dump(f'{outfile_path}.xml')
    data.tofile(outfile_path)
    if geocode is not None:
        sarlab.geocode(outfile_path, geocode)


def load_file(path):
    '''
        Load generic ISCE file from path only
    '''
    im = createImage()
    im.load(path + '.xml')
    mm = im.memMap()
    return mm.copy()


def load_geom_from_slc(slc_path, file):
    inc_path = os.path.join(os.path.dirname(slc_path),
                            f'../../geom_reference/{file}.rdr.full')
    im = createImage()
    im.load(inc_path + '.xml')
    mm = im.memMap()
    if 'inc' in file:
        mm = np.swapaxes(mm, 2, 1)
    elif 'lat' or 'lon' in file:
        mm = mm[:, :, 0]
    return mm.copy()


def load_inc_from_slc(slc_path):
    inc_path = os.path.join(os.path.dirname(slc_path),
                            '../../geom_reference/incLocal.rdr.full')
    im = createImage()
    im.load(inc_path + '.xml')
    mm = im.memMap()
    mm = np.swapaxes(mm, 2, 1)
    return mm


def load_stack_vrt(files):
    '''
        Load a stack of ISCE coregistered SLCs via their VRTs using rasterio
    '''
    SLCs = None
    for i in range(len(files)):
        print(f'Loading SLC {i} / {len(files)}...')
        slc = rasterio.open(files[i] + '.vrt').read()[0]
        if SLCs is None:
            SLCs = np.zeros(
                (len(files), slc.shape[0], slc.shape[1]), dtype=np.complex64)

        SLCs[i, :, :] = slc

    return SLCs


def load_stack(files):
    '''
        Load a stack of ISCE coregistered SLCs
    '''
    SLCs = None
    for i in range(len(files)):
        print(f'Loading SLC {i} / {len(files)}...')
        im = createImage()
        im.load(files[i] + '.xml')
        mm = im.memMap()
        if SLCs is None:
            SLCs = np.zeros(
                (len(files), mm.shape[0], mm.shape[1]), dtype=np.complex64)

        SLCs[i, :, :] = mm[:, :, 0]

    return SLCs


def load_stack_ndvi(files):
    '''
        Load a stack of ISCE coregistered SLCs
    '''
    VIs = None
    # return None
    for i in range(len(files)):
        print(f'Loading NDVI {i} / {len(files)}...')
        im = createImage()
        file = files[i]
        file = '/'.join(file.split('/')[0:-1]) + '/ndvi.rdr.full'
        # print(file)

        im.load(file + '.xml')
        mm = im.memMap()
        if VIs is None:
            VIs = np.zeros(
                (len(files), mm.shape[1], mm.shape[2]), dtype=np.float32)

        VIs[i, :, :] = mm[0, :, :]

    return VIs


def load_stack_uavsar(files, rows, cols):
    '''
        Load a stack of ISCE coregistered SLCs via their VRTs using rasterio
    '''
    SLCs = np.zeros(
        (len(files), cols, rows), dtype=np.complex64)

    for i in range(len(files)):
        print(f'Loading SLC {i} / {len(files)}...')
        SLCs[i, :, :] = np.fromfile(
            files[i], dtype=np.complex64).reshape((cols, rows))

    return SLCs
