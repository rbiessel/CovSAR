import rasterio
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage
import numpy as np
import os
from matplotlib import pyplot as plt


def write_image(outfile_path, data):
    '''
        Write out image in ISCE format. Assumed to be of dtype FLOAT
    '''
    image = createImage()
    image.setWidth(data.shape[1])
    image.setLength(data.shape[0])
    image.setAccessMode('write')
    image.filename = outfile_path
    image.dataType = 'FLOAT'
    image.createImage()
    image.dump(f'{outfile_path}.xml')
    data.tofile(outfile_path)


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
