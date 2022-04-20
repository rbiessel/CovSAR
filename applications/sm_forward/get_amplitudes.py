import rasterio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter


def multilook(im, ml=(8, 2), thin=(8, 2)):
    print('Multilook shape: ', ml)
    outshape = (im.shape[0] // ml[0], im.shape[1] // ml[1])
    imf = uniform_filter(im.real, size=ml)
    print(im.dtype, imf.dtype)
    if im.dtype == np.complex64:
        imf = imf.real + 1j * uniform_filter(im.imag, size=ml)
    # imf = imf[::ml[0]//2, ::ml[1]//2].copy()[:outshape[0], :outshape[1]]
    return imf


def main():

    path = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/stack/slcs_base.vrt'

    SLC_src = rasterio.open(path)
    SLCs = SLC_src.read()

    lat, lon = 68.6168, -149.3002

    index = SLC_src.index(lon, lat)

    plt.imshow(np.abs(SLCs[0]), vmin=10, vmax=90, cmap=plt.cm.binary)
    plt.show()

    print(index)

    SLCs = SLCs[:, 4000:5000, 1000:2500]

    size = (20, 20)

    pair = (0, 4)

    interferogram = SLCs[pair[0]] * SLCs[pair[1]].conj()
    interferogram = multilook(interferogram, ml=size)

    dif = np.log(multilook(np.abs(SLCs[pair[0]]), ml=size)) - \
        np.log(multilook(np.abs(SLCs[pair[1]]), ml=size))

    # plt.imshow(dif, cmap=plt.cm.binary)
    # plt.show()

    # plt.imshow(np.angle(interferogram), cmap=plt.cm.hsv)
    # plt.show()

    # plt.scatter(dif, np.angle(interferogram), s=0.05)
    # plt.show()

    plt.hist(dif.flatten(), bins=200)
    plt.show()

    plt.hist(np.angle(interferogram).flatten(), bins=200)
    plt.show()


main()
