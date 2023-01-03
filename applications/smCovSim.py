import numpy as np
import closures
from matplotlib import pyplot as plt
import figStyle
import glob
import isceio as io
from covariance import CovarianceMatrix
from sm_forward.sm_forward import SMForward
import library as sarlab
from scipy import stats
import matplotlib.patches as mpl_patches
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
stack_path = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/slope_mtn/SLC/**/*.slc.full'
stack_path = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetB/SLC/**/*.slc.full'
forward_model = SMForward(1/10, 0.1, 0.5, 0.001)


def main():

    files = glob.glob(stack_path)
    files = sorted(files)
    # files = files[0:5]
    dates = []
    for file in files:
        date = file.split('/')[-2]
        dates.append(date)

    SLCs = io.load_stack_vrt(files)

    clip = None

    clip = [0, -1, 0, -1]

    print(SLCs.shape)
    clip = [100, 300, 100, 300]
    SLCs = SLCs[:, clip[0]:clip[1], clip[2]:clip[3]]
    lf = 4

    N = SLCs.shape[0]
    ml_size = (7*lf, 19*lf)
    n = ml_size[0] * ml_size[1]  # Number of looks
    sample_size = (2 * lf, 5*lf)
    cov = CovarianceMatrix(SLCs, ml_size=ml_size,
                           sample=sample_size)

    SLCs = None

    intensity = cov.get_intensity()
    print(intensity.shape)
    # Take some arbitrary pixel
    print(cov.cov.shape)
    coh = cov.get_coherence()

    C = coh[:, :, 5, 5]
    I = intensity[5, 5, :]

    sm = I * 10

    forward_model.set_moistures(sm)
    forward_model.plot_dielectric()

    def interfere(refi, seci):
        interferogram = refi * seci.conj()

    C_true = C.copy()

    triplets = closures.get_triplets(C.shape[0], all=False)

    closure_stack_observed = np.zeros((len(triplets)), dtype=np.complex64)
    closure_stack_simulated = np.zeros((len(triplets)), dtype=np.complex64)
    intensity_triplets = np.zeros((len(triplets)), dtype=np.float32)

    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            C[i, j] = forward_model.get_phases_dezan(sm[i], sm[j])

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = C[triplet[0], triplet[1]] * C[triplet[1],
                                                triplet[2]] * C[triplet[0], triplet[2]].conj()
        closure_observed = C_true[triplet[0], triplet[1]] * C_true[triplet[1],
                                                                   triplet[2]] * C_true[triplet[0], triplet[2]].conj()

        amp_triplet = sarlab.intensity_closure(
            I[triplet[0]], I[triplet[1]], I[triplet[2]], norm=False, cubic=False, filter=1, inc=None)

        closure_stack_simulated[i] = closure
        closure_stack_observed[i] = closure_observed

        intensity_triplets[i] = amp_triplet

    mask_upper = np.zeros(C.shape)
    mask_lower = mask_upper.copy()
    mask_upper[np.triu_indices_from(C, 1)] = 1
    mask_lower[np.tril_indices_from(C, -1)] = 1

    diag_mask = np.ones([C.shape[0]])
    diag_mask = np.diag(diag_mask)
    diag_I = np.diag(I)
    print(diag_I)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

    ax[0].set_title('Modeled Covariance (SM Only)')
    ax[0].imshow(np.abs(C), alpha=mask_lower,
                 cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].imshow(np.angle(C), alpha=mask_upper,
                 cmap=plt.cm.seismic, vmin=-np.pi/2, vmax=np.pi/2)

    ax[0].imshow(diag_I, alpha=diag_mask,
                 cmap=plt.cm.cividis, vmin=np.min(I), vmax=np.max(I))

    ax[0].imshow(np.angle(C), alpha=mask_upper,
                 cmap=plt.cm.seismic, vmin=-np.pi/2, vmax=np.pi/2)

    ax[1].set_title('Observed Covariance')
    ax[1].imshow(np.abs(C_true), alpha=mask_lower,
                 cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].imshow(np.angle(C_true), alpha=mask_upper,
                 cmap=plt.cm.seismic, vmin=-np.pi, vmax=np.pi)

    ax[1].imshow(diag_I, alpha=diag_mask,
                 cmap=plt.cm.cividis, vmin=np.min(I), vmax=np.max(I))

    # ax[2].plot(sm)
    # ax[2].set_xlabel('Time')
    # ax[2].set_ylabel('Relative Soil Moisture')

    # print(intensity_triplets.dtype)
    # r, pval = stats.pearsonr(
    #     intensity_triplets, np.angle(closure_stack_simulated))
    # ax[3].scatter(intensity_triplets, np.angle(
    #     closure_stack_simulated), s=10, color='black', alpha=0.3)

    # ax[3].set_xlabel(r'$\mathfrak{S} [$-$]  $')
    # ax[3].set_ylabel(r'$\Xi$ [$rad$]')
    # ax[3].axhline(y=0, color='k', alpha=0.15)
    # ax[3].axvline(x=0, color='k', alpha=0.15)
    # ax[3].grid(alpha=0.2)

    # labels = []
    # labels.append(f'R$^{{2}} = {{{np.round(r**2, 2)}}}$')

    # handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
    #                                  lw=0, alpha=0)] * 2
    # # create the legend, supressing the blank space of the empty line symbol and the
    # # padding between symbol and label by setting handlelenght and handletextpad

    # ax[3].legend(handles, labels, loc='best', fontsize='medium',
    #              fancybox=True, framealpha=0.7,
    #              handlelength=0, handletextpad=0)
    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/simulated_scatter.png', dpi=300)
    plt.show()

    # Apply phases via
main()
