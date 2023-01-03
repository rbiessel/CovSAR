import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import colorcet as cc
from matplotlib.cm import get_cmap
import figStyle

cyclic_cm = get_cmap('cet_CET_C8')


def interpolate_phase_intensity(raw_intensities, error_coh, plot=False):
    imax = np.max(raw_intensities)
    imin = np.min(raw_intensities)

    ngrid = 100j
    grid_x, grid_y = np.mgrid[imin:imax:ngrid,
                              imin:imax:ngrid]
    points = []
    values = []
    for i in range(error_coh.shape[0]):
        for j in range(error_coh.shape[1]):
            points.append(
                (raw_intensities[i], raw_intensities[j]))
            values.append(np.angle(error_coh[i, j]))

    points = np.array(points)
    values = np.array(values)
    grid_z2 = griddata(
        points, values, (grid_x, grid_y),  method='linear')

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        im = ax[0].imshow(grid_z2.T, extent=(
            imin, imax, imin, imax), origin='lower', cmap=cyclic_cm, vmin=-np.pi/30, vmax=np.pi/30)

        ax[0].scatter(points[:, 0], points[:, 1],
                      marker='o', s=15, facecolors='none', edgecolors='black')

        ax[0].contour(grid_z2.T, colors='k', origin='lower', extent=(
            imin, imax, imin, imax))
        ax[0].set_xlabel('Backscatter Intensity (Reference)')
        ax[0].set_ylabel('Backscatter Intensity (Secondary)')

    grid_z2 = np.flip(grid_z2, axis=0)

    dif_interpolated = np.tile(
        grid_x[:, 0], (len(grid_x[:, 0]), 1))
    dif_interpolated = dif_interpolated.T - dif_interpolated
    dif_interpolated = np.flip(dif_interpolated, axis=0)

    m = 10

    gradient = (grid_z2[-m, -m] - grid_z2[-1, -1]) / (
        (dif_interpolated[-m, -m] - dif_interpolated[-1, -1]))

    lin_phase = dif_interpolated * gradient
    lin_phase = np.flip(lin_phase, axis=0)
    grid_z2 = np.flip(grid_z2, axis=0)

    with_linear_phase = grid_z2 + (1 * lin_phase.T)

    if plot:
        ax[1].imshow(with_linear_phase, extent=(
            imin, imax, imin, imax), origin='lower', cmap=cyclic_cm, vmin=-np.pi/10, vmax=np.pi/10)
        ax[1].contour(with_linear_phase, colors='k', origin='lower', extent=(
            imin, imax, imin, imax))
        ax[1].set_xlabel('Reference Intensity [$dB$]')
        ax[1].set_ylabel('Seconday Intensity [$dB$]')
        fig.tight_layout()

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax,  label='Nonlinear Phase Error (rad)')
        plt.show()

    return gradient

    # Resample phases from interpolated linearized grid

    # Return phases
