from this import d
import h5py
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from scipy.ndimage import median_filter


def main():
    vel_corrected_path = "/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_publi/corrected_timeseries/mintpy/vel_corrected.h5"
    vel_raw_path = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_publi/raw_timeseries/mintpy/vel_raw.h5'

    vel_raw = None
    vel_corrected = None

    red = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_publi/corrected_timeseries/mintpy/rdr_dalton_red.tif'
    green = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_publi/corrected_timeseries/mintpy/rdr_dalton_green.tif'
    blue = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_publi/corrected_timeseries/mintpy/rdr_dalton_blue.tif'

    # red_im = np.fromfile(red, dtype=np.int8).reshape((275, 230))
    # blue_im = np.fromfile(green, dtype=np.int8).reshape((275, 230))
    # green_im = np.fromfile(blue, dtype=np.int8).reshape((275, 230))

    # image = np.zeros((275, 230, 3), dtype=np.int8)
    # image[:, :, 0] = median_filter(np.abs(red_im), size=1)
    # image[:, :, 1] = median_filter(np.abs(blue_im), size=1)
    # image[:, :, 2] = median_filter(np.abs(green_im), size=1)

    # image = np.abs(image)

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].imshow(image[:, :, 0])
    # ax[1].imshow(image[:, :, 1])
    # ax[2].imshow(image[:, :, 2])
    # plt.show()

    # image[np.where(image <= 0)] = 120

    # ax = plt.subplot()
    # ax.imshow(image)
    # ax.tick_params(labelbottom=False, labelleft=False)
    # plt.tight_layout()
    # plt.savefig('/Users/rbiessel/Documents/dalton_S2.png',
    #             transparent=True, dpi=300)
    # plt.show()

    # return
    with h5py.File(vel_raw_path, "r") as f:
        vel_raw = np.array(f.get('velocity')[:, :])
    with h5py.File(vel_corrected_path, "r") as f:
        vel_corrected = np.array(f.get('velocity')[:, :])

    dif = (vel_corrected - vel_raw)  # m/yr
    dif *= 1000  # cm/yr

    ax = plt.subplot()
    im = plt.imshow(dif, cmap=plt.cm.seismic, vmin=-30, vmax=30)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.colorbar(im, ax=ax, orientation='horizontal')

    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/dalton_vel_dif.png',
                transparent=True, dpi=300)
    plt.show()


main()
