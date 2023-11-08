import logging
from this import d
from matplotlib import dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import matplotlib.patches as mpl_patches
import scipy.stats as stats
import closures
import figStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from datetime import datetime
import colorcet as cc
from matplotlib.cm import get_cmap
import interpolate_phase
import matplotlib as mpl
import covariance
import pub_pixels
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

data_root = '/Users/rbiessel/Documents/InSAR/plotData'

# colsbg = ['#19192a', '#626282', '#aaaabe', '#cbcbd7']

cmap_div = get_cmap('cet_diverging_bwr_20_95_c54')
cmap_cont = get_cmap('cet_linear_grey_0_100_c0')
cmap_cont = cc.cm.fire
cmap_cont = cc.cm.dimgray
# cmap_cont = get_cmap('cet_linear_protanopic_deuteranopic_kbw_5_98_c40')

cmap_lines = mpl.cm.get_cmap("cet_glasbey_hv")


def main():

    df = pd.read_csv('/Users/rbiessel/Documents/InSAR/vegas_weather/2021.csv')
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')

    df_DNWR = df.loc[df['STATION'] == 'USC00262243']

    pixel_paths = glob.glob(data_root + '/DNWR/p_*/')
    vegas_dates = glob.glob(
        '/Users/rbiessel/Documents/InSAR/vegas_all/DNWR/SLC/**/*.slc.full')
    vegas_dates = [file.split('/')[-1].replace('.slc.full', '')
                   for file in vegas_dates]

    vegas_dates = [datetime.strptime(date, '%Y%m%d') for date in vegas_dates]
    vegas_dates = np.array(vegas_dates)
    vegas_dates = np.sort(vegas_dates)

    # df_DNWR = df_DNWR[(df_DNWR['DATE'] >= vegas_dates.min())
    #                   & (df_DNWR['DATE'] <= vegas_dates.max())]

    print(vegas_dates)
    print(pixel_paths)
    keep = [4]

    # pixel_paths = [pixel_paths[2], pixel]
    pixel_paths = [pixel_paths[i] for i in keep]
    pixel_paths = ['/Users/rbiessel/Documents/InSAR/plotData/DNWR/p_116_54/']
    pixel_paths = [pub_pixels.pixel_paths[0]]
    plen = len(pixel_paths)
    plotlen = 3
    fig, axes = plt.subplots(ncols=plen, nrows=plotlen,
                             figsize=(7, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]}, constrained_layout=True)

    # PLOT SCATTER
    for p in range(len(pixel_paths)):
        ax = axes[0]
        ax.set_title('(a) ', loc='left')
        path = pixel_paths[p]
        label = pixel_paths[p].split('/')[-3]
        C = np.load(os.path.join(
            path, 'C_raw.np.npy'))

        C_ln_slope = np.load(os.path.join(
            path, 'C_ln_slope.np.npy'))

        # C_ln_unc = np.load(os.path.join(
        #     path, 'C_ln_unc.np.npy'))

        I = np.load(os.path.join(
            path, 'Intensities.np.npy'))

        coherence = np.zeros(C.shape, dtype=np.complex64)
        print('shape: ', coherence.shape)
        for i in range(coherence.shape[0]):
            for j in range(coherence.shape[1]):
                if j >= i:
                    coherence[i, j] = C[i, j] / \
                        np.sqrt((C[i, i] * C[j, j]))
                else:
                    coherence[i, j] = coherence[j, i].conj()
        # Phase error
        mask_upper = np.zeros(C.shape)
        mask_lower = mask_upper.copy()
        mask_upper[np.triu_indices_from(C, 1)] = 1
        mask_lower[np.tril_indices_from(C, -1)] = 1

        diag_mask = np.ones([C.shape[0]])
        diag_mask = np.diag(diag_mask)
        diag_I = np.diag(I)

        x_lims_SAR = mdates.date2num(vegas_dates)

        extent = [x_lims_SAR[0], x_lims_SAR[-1],
                  x_lims_SAR[-1], x_lims_SAR[1]]

        shared_kwargs = {
            'extent': extent,
            'origin': 'upper'
        }
        cohImg = ax.imshow(np.abs(coherence), alpha=mask_lower,
                           cmap=cmap_cont, vmin=0.6, vmax=1, **shared_kwargs)

        phiImg = ax.imshow(np.angle(C_ln_slope), alpha=mask_upper,
                           cmap=cmap_div, vmin=-np.pi/30, vmax=np.pi/30, **shared_kwargs)

        # intenImg = ax.imshow(diag_I, alpha=diag_mask,
        #                      cmap=cmap_cont, vmin=np.min(I), vmax=np.max(I), **shared_kwargs)
        ax.xaxis_date()
        ax.yaxis_date()

        date_format = mdates.DateFormatter('%Y-%m-%d')

        ax.xaxis.set_major_formatter(date_format)
        ax.yaxis.set_major_formatter(date_format)
        ax.tick_params(axis='y', rotation=15)

        ax.set_aspect('equal')

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='10%', pad=0.1)
        cbar = fig.colorbar(cohImg, aspect=5, ax=ax, fraction=0.05)
        cbar.ax.set_title(r'[$\lvert \gamma \rvert$]', pad=10)

        # # cax2 = divider.append_axes('right', size='10%', pad=0.50)
        cbar2 = fig.colorbar(phiImg, aspect=5, ax=ax, fraction=0.05)
        cbar2.ax.set_title('[$\mathrm{rad}$]', pad=10)

        # cax3 = divider.append_axes('right', size='10%', pad=0.50)
        # cbar3 = fig.colorbar(intenImg, cax=cax3)
        # cbar3.ax.set_title('[$dB$]')

        ax.set_xlabel('Reference')
        ax.set_ylabel('Secondary')

        # ax.xaxis.set_major_formatter('%m-%d-%Y')
        # ax.yaxis_date()

        ax = axes[1]
        ax.set_title('(c) ', loc='left')

        differences = np.load(os.path.join(
            path, 'dispDiff.np.npy'))

        precip = df_DNWR['PRCP'].values
        weather_dates = df_DNWR['DATE'].values

        ax2 = ax.twinx()
        l1, = ax2.plot(x_lims_SAR, differences, 'o-',
                       label='Disp. Difference', linewidth=2, color='tomato', alpha=1, markersize=4)
        ax2.set_ylabel('[$\mathrm{mm}$]')
        ax2.set_ylim([-0.7, 1.5])

        ax.set_ylabel('[$\mathrm{dB}$]')
        # ax.set_ylim([38, 43])

        l2, = ax.plot(x_lims_SAR,  I - I[0], 'o-', label='Intensity',
                      color='steelblue', alpha=1, linewidth=2, markersize=4)

        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([-2.2, 4])
        ax.axhline(0, linewidth=1, color='gray', alpha=0.2)

        [ax.axvline(_x, linewidth=1, color='gray', alpha=0.2)
         for _x in vegas_dates]

        ll = [l1, l2]
        ax.legend(ll, [ll_.get_label()
                       for ll_ in ll], loc='upper right', fontsize=10, ncol=2)

        ax.xaxis_date()
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        ax = axes[2]
        ax.set_title('(e) ', loc='left')

        x_lims_WEATHER = mdates.date2num(weather_dates)

        minTemp = df_DNWR['TMIN'].values
        minTemp_sar = minTemp[np.intersect1d(
            x_lims_WEATHER, x_lims_SAR, return_indices=True)[1]]

        l1, = ax.plot(x_lims_SAR, minTemp_sar, '-o', color='tomato',
                      label='Minimum Temperature', linewidth=2, markersize=4)
        ax.set_ylabel('[$C^{\circ}$]')
        ax2 = ax.twinx()
        l2 = ax2.bar(x_lims_WEATHER, precip, label='Precipitation',
                     width=1.9, alpha=0.8, color='steelblue')
        ax2.set_ylabel('[$\mathrm{mm}$]')
        ax2.set_ylim([0, 15])
        [ax.axvline(_x, linewidth=1, color='gray', alpha=0.2)
         for _x in vegas_dates]
        ax.axhline(0, linewidth=1, color='gray', alpha=0.2)
        ll = [l1, l2]
        ax.legend(ll, [ll_.get_label() for ll_ in ll],
                  loc='upper left', fontsize=10)
        ax.tick_params(axis='x', rotation=45)

        ax.xaxis_date()
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlim([x_lims_SAR[0], x_lims_SAR[-1]])
        plt.savefig(
            '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/vegas_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    dBs = np.tile(I, (len(I), 1))
    dBs = (dBs.T - dBs)

    dBs = dBs[np.triu_indices_from(dBs)].flatten()
    phiErrors = np.angle(
        C_ln_slope[np.triu_indices_from(C_ln_slope)].flatten())

    # phiErrors_unc = np.angle(
    #     C_ln_unc[np.triu_indices_from(C_ln_unc)].flatten())

    ax.scatter(dBs, phiErrors, alpha=0.5, color='black')
    # ax.scatter(dBs, phiErrors_unc, alpha=0.5, color='orange')

    ax.axvline(0, linewidth=1, color='gray', alpha=0.2)
    ax.axhline(0, linewidth=1, color='gray', alpha=0.2)
    ax.set_xlabel('Intensity Difference [$\mathrm{dB}$]')
    ax.set_ylabel('Predicted Phase Error [$\mathrm{rad}$]')
    ax.set_title('(a) ', loc='left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/dBPhiScatter_dnwr.png', dpi=300)
    plt.show()

    interpolate_phase.interpolate_phase_intensity(
        I, C_ln_slope, plot=True)

    # Coherence

    # handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
    #                                  lw=0, alpha=0)] * 2
    # # create the legend, supressing the blank space of the empty line symbol and the
    # # padding between symbol and label by setting handlelenght and handletextpad
    # axes[0, p].legend(handles, labels, loc='best', fontsize='medium',
    #                   fancybox=True, framealpha=0.7,
    #                   handlelength=0, handletextpad=0)

    # PLOT Difference
    # for p in range(len(pixel_paths)):
    #     ax = axes[p, 1]
    #     I = np.load(os.path.join(
    #         path, 'Intensities.np.npy'))
    #     ax.plot(I)

    # plt.tight_layout(pad=0.2, w_pad=0.8, h_pad=0.5)
    plt.show()


main()
